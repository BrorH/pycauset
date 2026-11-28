#include <gtest/gtest.h>
#include "pycauset/core/MemoryGovernor.hpp"
#include "pycauset/core/PersistentObject.hpp"
#include "pycauset/core/MemoryMapper.hpp"
#include <memory>

// Mock PersistentObject since we don't want to link the whole world
// We just need a valid pointer address.
class MockObject : public PersistentObject {
public:
    MockObject() : PersistentObject() {}
    
    // Helper to inject a real mapper
    void set_mapper(std::shared_ptr<MemoryMapper> m) {
        mapper_ = std::move(m);
        // Manually register if it's RAM
        if (mapper_->get_filename() == ":memory:") {
            storage_state_ = pycauset::core::StorageState::RAM_ONLY;
            pycauset::core::MemoryGovernor::instance().register_object(this, mapper_->get_data_size());
        }
    }

    std::unique_ptr<PersistentObject> clone() const override {
        return std::make_unique<MockObject>();
    }
};

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

TEST_F(MemoryGovernorTest, RegisterAndUnregister) {
    MemoryGovernor& gov = MemoryGovernor::instance();
    MockObject obj1;
    MockObject obj2;

    gov.register_object(&obj1, 1000);
    EXPECT_EQ(gov.get_tracked_ram_usage(), 1000);

    gov.register_object(&obj2, 2000);
    EXPECT_EQ(gov.get_tracked_ram_usage(), 3000);

    gov.unregister_object(&obj1);
    EXPECT_EQ(gov.get_tracked_ram_usage(), 2000);

    gov.unregister_object(&obj2);
    EXPECT_EQ(gov.get_tracked_ram_usage(), 0);
}

TEST_F(MemoryGovernorTest, UpdateSize) {
    MemoryGovernor& gov = MemoryGovernor::instance();
    MockObject obj1;

    gov.register_object(&obj1, 1000);
    EXPECT_EQ(gov.get_tracked_ram_usage(), 1000);

    gov.update_size(&obj1, 5000);
    EXPECT_EQ(gov.get_tracked_ram_usage(), 5000);
}

TEST_F(MemoryGovernorTest, SystemStats) {
    MemoryGovernor& gov = MemoryGovernor::instance();
    uint64_t total = gov.get_total_system_ram();
    uint64_t avail = gov.get_available_system_ram();

    EXPECT_GT(total, 0);
    EXPECT_GT(avail, 0);
    EXPECT_LE(avail, total);
}

TEST_F(MemoryGovernorTest, RequestRam) {
    MemoryGovernor& gov = MemoryGovernor::instance();
    
    // Request a small amount, should succeed
    EXPECT_TRUE(gov.request_ram(1024));

    // Request an impossible amount (Total RAM + 1TB), should fail
    // Note: This assumes the system doesn't have exabytes of RAM
    uint64_t impossible = gov.get_total_system_ram() + 1024ULL * 1024 * 1024 * 1024; 
    EXPECT_FALSE(gov.request_ram(impossible));
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

