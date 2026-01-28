/*
 * 能量状态管理器 - 嵌入式C实现
 * 适配MSP430低功耗MCU
 */

#include <stdint.h>
#include <stdbool.h>
#include "energy_state.h"

// 能量状态定义
typedef enum {
    ENERGY_STATE_FULL,  // > 4.0V
    ENERGY_STATE_MID,   // 3.6V - 4.0V
    ENERGY_STATE_LOW    // < 3.6V
} energy_state_t;

// 配置参数
typedef struct {
    float full_threshold;    // FULL状态阈值
    float mid_threshold;     // MID状态阈值
    uint16_t full_interval;  // FULL状态间隔（分钟）
    uint16_t mid_interval;   // MID状态间隔（分钟）
    float change_threshold;  // 变化率阈值（5%）
} energy_config_t;

// 状态管理器
typedef struct {
    energy_state_t current_state;
    energy_config_t config;
    
    // 状态变量
    float last_voltage;
    float last_data_value;
    uint32_t last_transmit_time;  // 时间戳
    uint32_t low_state_start_time;
    
    // 本地存储（铁电存储器）
    uint16_t storage_index;
    sensor_data_t local_storage[MAX_LOCAL_STORAGE];
    
    // 统计
    uint32_t transmit_count[3];  // 按状态统计
    uint32_t state_changes;
} energy_manager_t;

// 初始化能量管理器
void energy_manager_init(energy_manager_t* manager, const energy_config_t* config) {
    manager->config = *config;
    manager->current_state = ENERGY_STATE_MID;
    manager->last_voltage = 0.0f;
    manager->last_data_value = 0.0f;
    manager->last_transmit_time = 0;
    manager->storage_index = 0;
    manager->state_changes = 0;
    
    for (int i = 0; i < 3; i++) {
        manager->transmit_count[i] = 0;
    }
}

// 更新能量状态
// 
// 工作流程：
// 1. 应用电压滤波减少噪声
// 2. 根据过滤后的电压确定新状态
// 3. 如果状态改变，执行转换处理
energy_state_t energy_manager_update(energy_manager_t* manager, float voltage) {
    energy_state_t old_state = manager->current_state;
    energy_state_t new_state;
    
    // 应用简单的移动平均滤波，使用3点窗口
    // 有效降低测量噪声，改进状态判断的稳定性
    static float voltage_filter[3] = {0};
    static uint8_t filter_idx = 0;
    
    voltage_filter[filter_idx] = voltage;
    filter_idx = (filter_idx + 1) % 3;
    // 计算3个最近值的平均电压
    float filtered_voltage = (voltage_filter[0] + voltage_filter[1] + voltage_filter[2]) / 3.0f;
    
    // 根据阈值确定状态
    if (filtered_voltage >= manager->config.full_threshold) {
        new_state = ENERGY_STATE_FULL;
    } else if (filtered_voltage >= manager->config.mid_threshold) {
        new_state = ENERGY_STATE_MID;
    } else {
        new_state = ENERGY_STATE_LOW;
    }
    
    // 处理状态转换
    if (new_state != old_state) {
        manager->state_changes++;
        
        // 从LOW状态切换出来时上传缓存数据
        // 这确保了低电量期间积累的数据不会丢失
        if (old_state == ENERGY_STATE_LOW && manager->storage_index > 0) {
            energy_manager_upload_storage(manager);
        }
        
        // 进入LOW状态时记录时间戳
        // 用于计算何时执行每日批量上传
        if (new_state == ENERGY_STATE_LOW) {
            manager->low_state_start_time = get_current_timestamp();
        }
        
        manager->current_state = new_state;
    }
    
    manager->last_voltage = filtered_voltage;
    return new_state;
}

// 判断是否需要传输
bool should_transmit_data(
    energy_manager_t* manager,
    uint32_t current_time,
    float current_data,
    bool force_upload
) {
    // 强制上传（如报警）
    if (force_upload) {
        return true;
    }
    
    // 根据当前状态判断
    switch (manager->current_state) {
        case ENERGY_STATE_FULL:
            return check_full_state_transmit(manager, current_time);
            
        case ENERGY_STATE_MID:
            return check_mid_state_transmit(manager, current_time, current_data);
            
        case ENERGY_STATE_LOW:
            return check_low_state_transmit(manager, current_time, current_data);
            
        default:
            return false;
    }
}

// FULL状态传输检查
// 
// FULL状态的特点：电池充足，可进行高频传输
// 策略：固定10分钟间隔
bool check_full_state_transmit(energy_manager_t* manager, uint32_t current_time) {
    // 计算距上次传输的时间间隔
    uint32_t time_diff = current_time - manager->last_transmit_time;
    // 若达到或超过设定间隔，返回true表示应该传输
    return (time_diff >= manager->config.full_interval);
}

// MID状态传输检查
// 
// MID状态的特点：电池中等，需要平衡功耗和数据新鲜度
// 策略：30分钟时间触发 或 变化率>5%事件触发
bool check_mid_state_transmit(
    energy_manager_t* manager,
    uint32_t current_time,
    float current_data
) {
    // 时间间隔检查：30分钟一次
    uint32_t time_diff = current_time - manager->last_transmit_time;
    if (time_diff >= manager->config.mid_interval) {
        return true;
    }
    
    // 变化率检查：突发变化立即传输
    // 避免last_data_value为0导致的除以零错误
    if (manager->last_data_value != 0.0f) {
        // 计算相对变化率：|当前值-上次值|/上次值
        float change_rate = fabs(current_data - manager->last_data_value) / manager->last_data_value;
        // 若变化超过5%，触发传输
        if (change_rate > manager->config.change_threshold) {
            return true;
        }
    }
    
    return false;
}

// LOW状态传输检查
// 
// LOW状态的特点：电池低电，需要最大化电池寿命
// 策略：本地存储，每日(24小时)批量上传一次
bool check_low_state_transmit(
    energy_manager_t* manager,
    uint32_t current_time,
    float current_data
) {
    // 存储到本地铁电存储器（FRAM）
    // 仅在存储未满的情况下存储，避免数据覆盖
    if (manager->storage_index < MAX_LOCAL_STORAGE) {
        manager->local_storage[manager->storage_index].timestamp = current_time;
        manager->local_storage[manager->storage_index].value = current_data;
        manager->storage_index++;
    }
    
    // 检查是否到达每日上传时间
    // 计算从进入LOW状态以来的时间
    uint32_t time_in_low = current_time - manager->low_state_start_time;
    // 1440分钟 = 24小时
    if (time_in_low >= 1440) {
        return true;  // 执行每日批量上传
    }
    
    // 在LOW状态下，最常见的返回值是false，表示只做本地存储
    return false;
}

// 记录传输事件
void record_transmission(
    energy_manager_t* manager,
    uint32_t timestamp,
    float data_value
) {
    manager->last_transmit_time = timestamp;
    manager->last_data_value = data_value;
    manager->transmit_count[manager->current_state]++;
    
    // 清除本地存储（如果已上传）
    if (manager->current_state != ENERGY_STATE_LOW) {
        manager->storage_index = 0;
    }
}

// 获取节能统计
// 
// 计算使用能量管理策略相比固定传输间隔能节省多少能量
// 基线：每10分钟传输一次（24小时共144次）
uint16_t calculate_energy_savings(const energy_manager_t* manager) {
    // 简化计算：基于传输次数估算节能
    // 假设所有传输消耗相等的能量
    uint32_t baseline_transmissions = 144;  // 10分钟间隔 × 24小时 = 144次
    
    // 统计实际传输次数（按三个状态累加）
    uint32_t actual_transmissions = 
        manager->transmit_count[ENERGY_STATE_FULL] +
        manager->transmit_count[ENERGY_STATE_MID] +
        manager->transmit_count[ENERGY_STATE_LOW];
    
    // 计算节能百分比
    if (baseline_transmissions > actual_transmissions) {
        // 百分比 = (减少的次数 / 基线次数) × 100
        return ((baseline_transmissions - actual_transmissions) * 100) / baseline_transmissions;
    }
    
    // 若实际次数超过基线（例如MID状态频繁触发），则无节能
    return 0;
}