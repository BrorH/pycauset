#include <gtest/gtest.h>
#include "pycauset/core/MemoryGovernor.hpp"

using namespace pycauset::core;

class MemoryGovernorTest : public ::testing::Test {
protected:
    void SetUp() override {
        MemoryGovernor::instance().reset_for_testing();
    }

    void TearDown() override {
        MemoryGovernor::instance().reset_for_testing();
    }
};

TEST_F(MemoryGovernorTest, SingletonInstance) {
    MemoryGovernor& gov1 = MemoryGovernor::instance();
    MemoryGovernor& gov2 = MemoryGovernor::instance();
    EXPECT_EQ(&gov1, &gov2);
}

TEST_F(MemoryGovernorTest, SystemStats) {
    MemoryGovernor& gov = MemoryGovernor::instance();
    uint64_t total = gov.get_total_system_ram();
    uint64_t avail = gov.get_available_system_ram();

    EXPECT_GT(total, 0);
    EXPECT_GT(avail, 0);
    EXPECT_LE(avail, total);
}

TEST_F(MemoryGovernorTest, CanFitInRam) {
    MemoryGovernor& gov = MemoryGovernor::instance();
    
    // Request a small amount, should succeed
    EXPECT_TRUE(gov.can_fit_in_ram(1024));

    // Request an impossible amount (Total RAM + 1TB), should fail
    // Note: This assumes the system doesn't have exabytes of RAM
    uint64_t impossible = gov.get_total_system_ram() + 1024ULL * 1024 * 1024 * 1024; 
    EXPECT_FALSE(gov.can_fit_in_ram(impossible));
}

TEST_F(MemoryGovernorTest, PinnedMemoryBudget) {
    MemoryGovernor& gov = MemoryGovernor::instance();
    
    // Set a small limit for testing
    gov.set_max_pinned_memory(1000);
    
    // Should succeed
    EXPECT_TRUE(gov.try_pin_memory(500));
    EXPECT_EQ(gov.get_pinned_memory_usage(), 500);
    
    // Should succeed (total 900)
    EXPECT_TRUE(gov.try_pin_memory(400));
    EXPECT_EQ(gov.get_pinned_memory_usage(), 900);
    
    // Should fail (total 1100 > 1000)
    EXPECT_FALSE(gov.try_pin_memory(200));
    EXPECT_EQ(gov.get_pinned_memory_usage(), 900);
    
    // Unpin
    gov.unpin_memory(500);
    EXPECT_EQ(gov.get_pinned_memory_usage(), 400);
    
    // Should succeed now
    EXPECT_TRUE(gov.try_pin_memory(200));
    EXPECT_EQ(gov.get_pinned_memory_usage(), 600);
}


#ifdef __linux__
#include <unistd.h>
#include <filesystem>
#include <fstream>

namespace {
void write_file(const std::filesystem::path& p, const std::string& text) {
    std::filesystem::create_directories(p.parent_path());
    std::ofstream(p) << text;
}
}  // namespace

// Review 3.1 C-11: the budget follows a cgroup limit (containers, Kubernetes, Slurm).
TEST(CgroupMemory, V2TightestLevelWins) {
    const auto root = std::filesystem::temp_directory_path() / ("pc_cg_v2_" + std::to_string(::getpid()));
    write_file(root / "self", "0::/job/step\n");
    write_file(root / "fs/job/memory.max", "1000000\n");  // parent: 1 MB, 600 KB used
    write_file(root / "fs/job/memory.current", "600000\n");
    write_file(root / "fs/job/memory.stat", "anon 500000\ninactive_file 100000\n");
    write_file(root / "fs/job/step/memory.max", "max\n");  // the leaf has no limit
    write_file(root / "fs/job/step/memory.current", "10\n");
    const auto cg = pycauset::core::read_cgroup_memory((root / "self").string(), (root / "fs").string());
    ASSERT_TRUE(cg.has_value());
    EXPECT_EQ(cg->limit, 1000000u);
    EXPECT_EQ(cg->usage, 500000u);  // inactive page cache does not count
    std::filesystem::remove_all(root);
}

TEST(CgroupMemory, V1AndNoLimit) {
    const auto root = std::filesystem::temp_directory_path() / ("pc_cg_v1_" + std::to_string(::getpid()));
    write_file(root / "self", "12:cpu,cpuacct:/a\n9:memory:/a\n");
    write_file(root / "fs/memory/a/memory.limit_in_bytes", "2000000\n");
    write_file(root / "fs/memory/a/memory.usage_in_bytes", "500000\n");
    write_file(root / "fs/memory/memory.limit_in_bytes", "9223372036854771712\n");  // unlimited
    auto cg = pycauset::core::read_cgroup_memory((root / "self").string(), (root / "fs").string());
    ASSERT_TRUE(cg.has_value());
    EXPECT_EQ(cg->limit, 2000000u);
    EXPECT_EQ(cg->usage, 500000u);
    write_file(root / "self2", "0::/\n");
    EXPECT_FALSE(pycauset::core::read_cgroup_memory((root / "self2").string(), (root / "fs").string()).has_value());
    std::filesystem::remove_all(root);
}
#endif
