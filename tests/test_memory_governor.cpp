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

