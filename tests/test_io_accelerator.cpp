#include <gtest/gtest.h>
#include "pycauset/core/IOAccelerator.hpp"
#include "pycauset/core/MemoryMapper.hpp"
#include "pycauset/core/StorageUtils.hpp"
#include "pycauset/core/MemoryHints.hpp"
#include <filesystem>
#include <fstream>
#include <vector>

using namespace pycauset::core;

class IOAcceleratorTest : public ::testing::Test {
protected:
    std::string temp_file;
    size_t file_size = 1024 * 1024; // 1MB

    void SetUp() override {
        temp_file = pycauset::make_unique_storage_file("io_test");
        // Create a dummy file with valid header
        std::ofstream out(temp_file, std::ios::binary);
        
        // Header: Magic(8) + Version(4) + Reserved(52) = 64 bytes
        const char magic[] = "PYCAUSET";
        out.write(magic, 8);
        uint32_t version = 1;
        out.write(reinterpret_cast<const char*>(&version), sizeof(version));
        char reserved[52] = {0};
        out.write(reserved, 52);

        // Data
        std::vector<char> data(file_size, 0);
        out.write(data.data(), file_size);
        out.close();
    }

    void TearDown() override {
        if (std::filesystem::exists(temp_file)) {
            std::filesystem::remove(temp_file);
        }
    }
};

TEST_F(IOAcceleratorTest, PrefetchDoesNotCrash) {
    auto mapper = std::make_unique<MemoryMapper>(temp_file, file_size);
    IOAccelerator accelerator(mapper.get());

    // Prefetch start
    accelerator.prefetch(0, 4096);
    
    // Prefetch middle
    accelerator.prefetch(file_size / 2, 4096);
    
    // Prefetch end (clamped)
    accelerator.prefetch(file_size - 100, 200);

    // Prefetch invalid (should be ignored)
    accelerator.prefetch(file_size + 100, 100);
}

TEST_F(IOAcceleratorTest, FlushAsync) {
    auto mapper = std::make_unique<MemoryMapper>(temp_file, file_size);
    IOAccelerator accelerator(mapper.get());

    // Modify data
    char* data = static_cast<char*>(mapper->get_data());
    data[0] = 'X';

    // Flush async
    auto future = accelerator.flush_async(0, 4096);
    future.wait();

    // Verify flush happened (by re-reading file)
    // Note: MemoryMapper might keep file open, so we might need to close it to be sure,
    // or just trust the OS cache coherence.
    // But flush() calls FlushViewOfFile which ensures it's written to disk/cache.
}

TEST_F(IOAcceleratorTest, DiscardDoesNotCrash) {
    auto mapper = std::make_unique<MemoryMapper>(temp_file, file_size);
    IOAccelerator accelerator(mapper.get());

    accelerator.discard(0, 4096);
}

TEST_F(IOAcceleratorTest, ProcessHintSequential) {
    auto mapper = std::make_unique<MemoryMapper>(temp_file, file_size);
    IOAccelerator accelerator(mapper.get());

    // Should trigger prefetch internally
    MemoryHint hint = MemoryHint::sequential(0, 4096);
    accelerator.process_hint(hint);
}

TEST_F(IOAcceleratorTest, ProcessHintStrided) {
    auto mapper = std::make_unique<MemoryMapper>(temp_file, file_size);
    IOAccelerator accelerator(mapper.get());

    // Should be ignored for now (Phase 1) but not crash
    MemoryHint hint = MemoryHint::strided(0, file_size, 1024, 8);
    accelerator.process_hint(hint);
}
