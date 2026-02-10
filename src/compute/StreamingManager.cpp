#include "pycauset/compute/StreamingManager.hpp"
#include "pycauset/core/MemoryGovernor.hpp"
#include "pycauset/core/Nvtx.hpp"
#include "pycauset/core/DebugTrace.hpp"
#include "pycauset/matrix/DenseMatrix.hpp"
#include <cmath>
#include <iostream>
#include <algorithm>
#include <cstring>
#include <vector>

namespace pycauset {

namespace { // Helper goodies
    template<typename T> struct Tag { using type = T; };
}

StreamingManager::StreamingManager(ComputeWorker& worker)
    : worker_(worker) {}

size_t StreamingManager::calculate_block_size(size_t n, size_t element_size) {
    auto& governor = core::MemoryGovernor::instance();
    size_t available_ram = governor.get_available_system_ram();
    
    // Target using a safe chunk of available RAM (e.g., 80% of what's free/allowed)
    // We need 3 tiles in memory: A_tile, B_tile, C_tile
    // 3 * (block_size^2) * element_size = target_ram
    size_t target_ram = static_cast<size_t>(available_ram * 0.8);
    
    size_t optimal_block_size = static_cast<size_t>(std::sqrt(target_ram / (3.0 * element_size)));
    
    // Clamp
    size_t min_block = 4096; // 4K x 4K is decent chunk
    size_t block_size = std::max(min_block, optimal_block_size);
    if (block_size > n) block_size = n;
    
    // Align to 8 (SIMD friendly)
    block_size = (block_size / 8) * 8;
    if (block_size == 0) block_size = 8; // Fallback for very small N if clamp fails

    return block_size;
}

void StreamingManager::matmul(const MatrixBase& a, const MatrixBase& b, MatrixBase& c) {
    NVTX_RANGE("StreamingManager::matmul");
    
    // Basic validation
    if (a.rows() != c.rows() || b.cols() != c.cols() || a.cols() != b.rows()) {
        throw std::invalid_argument("StreamingManager::matmul dimension mismatch");
    }

    // Identify DataType and dispatch
    // Currently only supporting dense floating point for streaming
    if (a.get_matrix_type() != MatrixType::DENSE_FLOAT || 
        b.get_matrix_type() != MatrixType::DENSE_FLOAT || 
        c.get_matrix_type() != MatrixType::DENSE_FLOAT) {
         throw std::runtime_error("StreamingManager currently only supports Dense Float/Double matrices");
    }

    DataType dt = c.get_data_type();
    size_t elem_size = 0;
    if (dt == DataType::FLOAT64) elem_size = 8;
    else if (dt == DataType::FLOAT32) elem_size = 4;
    else throw std::runtime_error("StreamingManager only supports Float32/Float64");

    size_t n = a.rows(); // Assuming square-ish for blocking logic, but loops handle rectangular
    size_t block_size = calculate_block_size(n, elem_size);

    std::cout << "[PyCauset] Streaming MatMul: BlockSize=" << block_size 
              << " (" << (block_size * block_size * elem_size) / (1024.0 * 1024.0) << " MB/tile)" << std::endl;

    // TODO: Use ObjectFactory or scratch allocators instead of raw vectors + casting?
    // For now, we manually manage buffers to ensure we stay within the budget we calculated.
    // Since MatrixBase is abstract, we need concrete tile types.
    // We'll use DenseMatrix explicitly.

    auto run_streaming_loop = [&](auto type_tag) {
        using T = typename decltype(type_tag)::type;
        
        // Host buffers for tiles
        std::vector<T> a_buf(block_size * block_size);
        std::vector<T> b_buf(block_size * block_size);
        std::vector<T> c_buf(block_size * block_size);

        // Wrappers (Lightweight views if possible, but DenseMatrix constructs usually allocate)
        // Here we create "View" matrices that point to our buffers if possible,
        // OR we create new matrices and copy. 
        // DenseMatrix<T> usually owns data. 
        // We'll construct DenseMatrix wrapping existing data if constructor allows, 
        // or just use them as temp storage containers.
        // Looking at DenseMatrix, it owns std::vector usually.
        // Let's assume we copy for now to be safe and clean, or use a "Ref" type if available.
        // Since we don't have Ref types yet, we'll treat the buffers as the backing store for the tile matrices.
        
        // Actually, ComputeWorker::matmul_tile takes MatrixBase references.
        // We can implement a simple "MatrixView" or specific "TileMatrix" if needed.
        // For Phase 3, let's create concrete DenseMatrices for tiles.
        // This incurs allocation, but they are reused? 
        // No, DenseMatrix(rows, cols) allocates.
        // If we re-allocate every tile, that's overhead.
        // Ideally we want: DenseMatrix<T> tile_a(rows, cols, buffer_ptr);
        
        // Hack for Phase 3: Create DenseMatrix once, and use direct data access to fill it.
        DenseMatrix<T> tile_a(block_size, block_size);
        DenseMatrix<T> tile_b(block_size, block_size);
        DenseMatrix<T> tile_c(block_size, block_size);

        const auto* a_real = dynamic_cast<const DenseMatrix<T>*>(&a);
        const auto* b_real = dynamic_cast<const DenseMatrix<T>*>(&b);
        auto* c_real = dynamic_cast<DenseMatrix<T>*>(&c);

        const T* raw_a = a_real->data();
        const T* raw_b = b_real->data();
        T* raw_c = c_real->data();

        size_t M = a.rows();
        size_t K = a.cols();
        size_t N = b.cols();

        // Loop: i (M), j (N), k (K)
        for (size_t i = 0; i < M; i += block_size) {
            size_t ib = std::min(block_size, M - i);
            
            for (size_t j = 0; j < N; j += block_size) {
                size_t jb = std::min(block_size, N - j);
                
                // Zero C tile
                // We reuse tile_c. Since it might be bigger than ib/jb, we should ideally resize logic or ignore padding.
                // For simplicity, we zero the whole max block or just the active region.
                // memset is fastest for POD.
                std::memset(tile_c.data(), 0, block_size * block_size * sizeof(T));

                // Accumulate over K
                for (size_t k = 0; k < K; k += block_size) {
                    size_t kb = std::min(block_size, K - k);

                    // 1. Copy A block -> tile_a
                    T* ta_data = tile_a.data();
                    for (size_t r = 0; r < ib; ++r) {
                        std::memcpy(&ta_data[r * block_size], &raw_a[(i + r) * K + k], kb * sizeof(T));
                    }
                    // NOTE: Padding (if kb < block_size) is garbage but safe if worker respects dims.
                    // However, our worker usually takes MatrixBase which has dims.
                    // We should cheat and set the dims of tile_a to (ib, kb) without realloc.
                    // DenseMatrix doesn't support resize without realloc usually.
                    // We'll pass the full block and tell the worker to use (ib, kb, jb) dimensions?
                    // ComputeWorker::matmul_tile takes MatrixBase objects.
                    // If we pass a (BlockSize x BlockSize) matrix, the worker will multiply the whole thing.
                    // This creates garbage if we don't handle boundaries.
                    // FIX: Create "View" matrices for the edge cases? Or use fixed block size and pad with zeros?
                    // Padding with zeros works for A and B.
                    // If k-loop handling edge: pad A's cols with 0, B's rows with 0.
                    // This is cleaner than resizing.

                    if (kb < block_size) {
                        // Zero out the rest of the rows in tile_a to avoid NaN/garbage affecting result
                        for (size_t r = 0; r < ib; ++r) {
                            std::memset(&ta_data[r * block_size + kb], 0, (block_size - kb) * sizeof(T));
                        }
                    }
                     
                     // 1b. Copy B block -> tile_b
                    T* tb_data = tile_b.data();
                    for (size_t r = 0; r < kb; ++r) {
                        std::memcpy(&tb_data[r * block_size], &raw_b[(k + r) * N + j], jb * sizeof(T));
                    }
                    // Pad B columns if jb < block_size? No, Matmul is A*B.
                    // A=(M, K), B=(K, N).
                    // Sub-block: A_sub=(ib, kb), B_sub=(kb, jb) -> C_sub=(ib, jb).
                    // If we feed (Block, Block) to GEMM, we need valid data.
                    // Padding A cols and B rows (the K dimension) with 0 is essentially correct for accumulation.
                    // Padding A rows and B cols (M and N dimensions) requires output filtering.
                    
                    // Simple approach: Zero EVERYTHING at start of tile load.
                    // Then copy valid rect.
                    // Then run full block GEMM.
                    // Then copy valid rect out.
                    // Overhead of zeroing small padding is negligible compared to huge GEMM.
                    
                    // Re-do load strategy:
                    // Clear A tile
                    std::memset(tile_a.data(), 0, block_size * block_size * sizeof(T));
                    // Copy valid A
                    for (size_t r = 0; r < ib; ++r) {
                         std::memcpy(&tile_a.data()[r * block_size], &raw_a[(i + r) * K + k], kb * sizeof(T));
                    }

                    // Clear B tile
                    std::memset(tile_b.data(), 0, block_size * block_size * sizeof(T));
                    // Copy valid B
                    for (size_t r = 0; r < kb; ++r) {
                        std::memcpy(&tile_b.data()[r * block_size], &raw_b[(k + r) * N + j], jb * sizeof(T));
                    }

                    // 2. Dispatch
                    // C_tile = 1.0 * A_tile * B_tile + 1.0 * C_tile
                    // We use beta=1.0 explicitly to accumulate into our tile_c buffer.
                    // But wait, tile_c is just this (i, j) block sum.
                    // The first K-iteration should be beta=0, subsequent beta=1?
                    // OR we zeroed tile_c at start of K-loop, doing C_tile += A*B is fine.
                    // ComputeWorker::matmul_tile(a, b, c, alpha, beta)
                    // If we strictly accumulate in tile_c, we set alpha=1, beta=1. 
                    // (Since we memset tile_c to 0 initially).
                    
                    worker_.matmul_tile(tile_a, tile_b, tile_c, 1.0, 1.0);
                }

                // 3. Write C tile back
                for (size_t r = 0; r < ib; ++r) {
                    std::memcpy(&raw_c[(i + r) * N + j], &tile_c.data()[r * block_size], jb * sizeof(T));
                }
            }
        }
    };

    struct Float64Tag { using type = double; };
    struct Float32Tag { using type = float; };

    if (dt == DataType::FLOAT64) run_streaming_loop(Float64Tag{});
    else if (dt == DataType::FLOAT32) run_streaming_loop(Float32Tag{});
    
    // Propagate scalar properties
    c.set_scalar(a.get_scalar() * b.get_scalar());
}

void StreamingManager::elementwise(
    const MatrixBase& a, 
    const MatrixBase& b, 
    MatrixBase& c, 
    ComputeWorker::ElementwiseOp op
) {
    NVTX_RANGE("StreamingManager::elementwise");

    if (a.rows() != c.rows() || a.cols() != c.cols() || 
        b.rows() != c.rows() || b.cols() != c.cols()) {
        throw std::invalid_argument("StreamingManager::elementwise dimension mismatch");
    }

    DataType dt = c.get_data_type();
    size_t block_size = 4096; 

    auto run_loop = [&](auto type_tag) {
         using T = typename decltype(type_tag)::type;
         DenseMatrix<T> tile_a(block_size, block_size);
         DenseMatrix<T> tile_b(block_size, block_size);
         DenseMatrix<T> tile_c(block_size, block_size);
         
         // Cast to DenseMatrix (assumes dense inputs)
         const auto* a_real = dynamic_cast<const DenseMatrix<T>*>(&a);
         const auto* b_real = dynamic_cast<const DenseMatrix<T>*>(&b);
         auto* c_real = dynamic_cast<DenseMatrix<T>*>(&c);
         
         const T* raw_a = a_real ? a_real->data() : nullptr;
         const T* raw_b = b_real ? b_real->data() : nullptr;
         T* raw_c = c_real ? c_real->data() : nullptr;

         if (!raw_c) throw std::runtime_error("StreamingManager: Output matrix must be dense");
         
         size_t Rows = c.rows();
         size_t Cols = c.cols();
         size_t A_Cols = a.cols();
         size_t B_Cols = b.cols();

         auto fill_tile = [&](DenseMatrix<T>& dst, const MatrixBase& src, const T* raw_src, 
                              size_t t_rows, size_t t_cols, size_t t_stride,
                              size_t r_off, size_t c_off, size_t src_width) {
             if (raw_src) {
                 for(size_t r=0; r<t_rows; ++r) {
                     std::memcpy(dst.data() + r*t_stride, raw_src + (r_off+r)*src_width + c_off, t_cols*sizeof(T));
                 }
             } else {
                 T* d = dst.data();
                 for(size_t r=0; r<t_rows; ++r) {
                     for(size_t c=0; c<t_cols; ++c) {
                         d[r*t_stride + c] = static_cast<T>(src.get_element_as_double(r_off + r, c_off + c));
                     }
                 }
             }
         };
         
         for (size_t i = 0; i < Rows; i += block_size) {
             size_t ib = std::min(block_size, Rows - i);
             for (size_t j = 0; j < Cols; j += block_size) {
                 size_t jb = std::min(block_size, Cols - j);
                 
                 // Use exact-sized tiles for edges to avoid processing garbage (safe for DIV)
                 if (ib < block_size || jb < block_size) {
                     DenseMatrix<T> ea(ib, jb);
                     DenseMatrix<T> eb(ib, jb);
                     DenseMatrix<T> ec(ib, jb);
                     
                     fill_tile(ea, a, raw_a, ib, jb, jb, i, j, A_Cols);
                     fill_tile(eb, b, raw_b, ib, jb, jb, i, j, B_Cols);
                     
                     worker_.elementwise_tile(ea, eb, ec, op);
                     
                     for(size_t r=0; r<ib; ++r) {
                         std::memcpy(raw_c + (i+r)*Cols + j, ec.data() + r*jb, jb*sizeof(T));
                     }
                 } else {
                     fill_tile(tile_a, a, raw_a, block_size, block_size, block_size, i, j, A_Cols);
                     fill_tile(tile_b, b, raw_b, block_size, block_size, block_size, i, j, B_Cols);

                     worker_.elementwise_tile(tile_a, tile_b, tile_c, op);
                     
                     for(size_t r=0; r<ib; ++r) {
                         std::memcpy(raw_c + (i+r)*Cols + j, tile_c.data() + r*block_size, block_size*sizeof(T));
                     }
                 }
             }
         }
    };
    
    if (dt == DataType::FLOAT64) run_loop(Tag<double>{});
    else if (dt == DataType::FLOAT32) run_loop(Tag<float>{});
    else if (dt == DataType::INT32) run_loop(Tag<int32_t>{});
    else if (dt == DataType::INT64) run_loop(Tag<int64_t>{});
    else {
        // Fallback or throw? 
        // For other types, we might fall back to direct?
        // But StreamingManager usage implies we MUST stream.
        // We should support all types eventually.
        // For now, throw for unimplemented types.
        throw std::runtime_error("StreamingManager: unsupported dtype for elementwise (ints/floats only)");
    }
}

} // namespace pycauset
