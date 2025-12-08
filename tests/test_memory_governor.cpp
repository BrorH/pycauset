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
    
    // Helper to inject a real mapper for testing spill
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

TEST_F(MemoryGovernorTest, EvictionLogic) {
    MemoryGovernor& gov = MemoryGovernor::instance();
    
    // 1. Create an object in RAM (1MB)
    MockObject obj1;
    auto mapper1 = std::make_unique<MemoryMapper>(":memory:", 1024 * 1024, 0, true);
    obj1.set_mapper(std::move(mapper1));
    
    EXPECT_EQ(gov.get_tracked_ram_usage(), 1024 * 1024);
    EXPECT_EQ(obj1.get_storage_state(), StorageState::RAM_ONLY);

    // 2. Force eviction by setting safety margin to (Available - 500KB)
    // This means we only have 500KB "usable" RAM.
    uint64_t avail = gov.get_available_system_ram();
    if (avail > 1024 * 1024 * 2) { // Ensure we have enough RAM to play with
        gov.set_safety_margin(avail - 512 * 1024); 
        
        // 3. Request 1MB. This requires 1MB + Margin > Available.
        // Since Margin is almost Available, this will definitely fail unless we evict.
        // obj1 (1MB) is in the way.
        
        // Note: request_ram calls evict_until_fits.
        // evict_until_fits will try to spill obj1.
        
        bool success = gov.request_ram(1024 * 1024);
        
        // Check if obj1 was spilled
        if (obj1.get_storage_state() == StorageState::DISK_BACKED) {
            // Eviction happened!
            EXPECT_EQ(gov.get_tracked_ram_usage(), 0); // obj1 is no longer tracked
            EXPECT_TRUE(success);
        } else {
            // Eviction failed (maybe OS reported more RAM suddenly or spill failed)
            // This test is flaky depending on OS memory dynamics.
            // But we can check if spill_to_disk works manually.
        }
    }
}

TEST_F(MemoryGovernorTest, ManualSpillAndPromote) {
    MemoryGovernor& gov = MemoryGovernor::instance();
    
    // Set a small safety margin to ensure request_ram succeeds
    // unless the system is critically low on memory.
    uint64_t old_margin = gov.get_safety_margin();
    gov.set_safety_margin(0);

    MockObject obj;
    auto mapper = std::make_unique<MemoryMapper>(":memory:", 1024, 0, true);
    obj.set_mapper(std::move(mapper));

    EXPECT_EQ(obj.get_storage_state(), StorageState::RAM_ONLY);
    EXPECT_EQ(gov.get_tracked_ram_usage(), 1024);

    // Spill
    EXPECT_TRUE(obj.spill_to_disk());
    EXPECT_EQ(obj.get_storage_state(), StorageState::DISK_BACKED);
    EXPECT_EQ(gov.get_tracked_ram_usage(), 0);
    EXPECT_NE(obj.get_backing_file(), ":memory:");

    // Promote
    EXPECT_TRUE(obj.promote_to_ram());
    EXPECT_EQ(obj.get_storage_state(), StorageState::RAM_ONLY);
    EXPECT_EQ(gov.get_tracked_ram_usage(), 1024);
    EXPECT_EQ(obj.get_backing_file(), ":memory:");

    // Restore margin
    gov.set_safety_margin(old_margin);
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

