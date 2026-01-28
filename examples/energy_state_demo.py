"""
能量状态管理器演示
生成可视化图表展示三状态规则引擎的效果
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../cloud/model'))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from algorithms.energy_manager import (
    EnergyStateManager, 
    EnergyState,
    simulate_energy_aware_transmission
)

def plot_energy_state_demo():
    """
    绘制能量状态管理演示图表
    
    生成一个包含三个子图的可视化：
    1. 电压变化与能量状态：显示电池电压如何变化，以及设备在三个状态间的转换
    2. 传感器数据与传输触发：显示数据趋势及触发传输的事件
    3. 传输间隔分布：分析不同状态下的传输频率
    
    这个演示帮助理解三状态管理策略如何动态调整传输行为以优化功耗。
    
    Returns:
        dict: 模拟结果，包含transmissions、savings、status等信息
    """
    fig = plt.figure(figsize=(15, 10))
    
    # 生成模拟数据
    np.random.seed(42)
    hours = 24  # 模拟24小时
    minutes = hours * 60  # 转换为分钟
    
    # 生成电压序列（模拟电池充放电）
    time_hours = np.linspace(0, hours, minutes)
    
    # 基础电压：3.8V，模拟昼夜模式（白天充电，晚上放电）
    # 正弦波：最高4.2V（6点），最低3.4V（18点）
    voltage = 3.8 + 0.4 * np.sin(2 * np.pi * (time_hours - 6) / 24)
    
    # 添加高频波动（半日周期，代表充放电的不均匀性）
    voltage += 0.1 * np.sin(2 * np.pi * time_hours / 12)
    # 添加随机噪声（代表测量误差和干扰）
    voltage += 0.05 * np.random.randn(minutes)
    
    # 确保电压在合理范围内
    voltage = np.clip(voltage, 3.0, 4.2)
    
    # 生成土壤湿度数据（模拟传感器数据）
    # 基础值：30%，日周期变化（雨天湿度高）
    moisture = 30 + 15 * np.sin(2 * np.pi * time_hours / 24)
    # 添加6小时周期的变化（局部天气影响）
    moisture += 5 * np.sin(2 * np.pi * time_hours / 6)
    # 添加随机噪声
    moisture += 2 * np.random.randn(minutes)
    # 限制在合理范围
    moisture = np.clip(moisture, 15, 65)
    
    # 运行能量状态管理模拟
    result = simulate_energy_aware_transmission(voltage, moisture)
    manager = result['manager']
    transmissions = result['transmissions']
    
    # 提取传输时间和状态
    transmit_times = [t['minute'] for t in transmissions]
    transmit_states = [t['state'] for t in transmissions]
    transmit_voltages = [t['voltage'] for t in transmissions]
    
    # 子图1：电压变化与状态区域
    ax1 = plt.subplot(3, 1, 1)
    
    # 绘制能量状态的背景色区域，方便直观理解各状态的电压范围
    full_region = Rectangle((0, 4.0), minutes, 0.2, alpha=0.2, color='green', label='FULL (>4.0V)')
    mid_region = Rectangle((0, 3.6), minutes, 0.4, alpha=0.2, color='orange', label='MID (3.6-4.0V)')
    low_region = Rectangle((0, 3.0), minutes, 0.6, alpha=0.2, color='red', label='LOW (<3.6V)')
    
    ax1.add_patch(full_region)
    ax1.add_patch(mid_region)
    ax1.add_patch(low_region)
    
    # 绘制电压曲线
    ax1.plot(time_hours, voltage, 'b-', linewidth=2, label='Battery Voltage')
    
    # 标记传输事件：不同状态用不同的标记形状
    # FULL：上三角△，MID：圆●，LOW：下三角▽
    state_colors = {'FULL': 'green', 'MID': 'orange', 'LOW': 'red'}
    for t_time, t_state, t_voltage in zip(transmit_times, transmit_states, transmit_voltages):
        ax1.plot(t_time/60, t_voltage, 
                marker='^' if t_state == 'FULL' else ('v' if t_state == 'LOW' else 'o'),
                color=state_colors[t_state],
                markersize=8, markeredgewidth=1, markeredgecolor='black')
    
    ax1.set_xlim(0, hours)
    ax1.set_ylim(3.0, 4.3)
    ax1.set_ylabel('Voltage (V)')
    ax1.set_title('Energy State Management: Battery Voltage & Transmission Events')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # 子图2：土壤湿度与传输触发
    ax2 = plt.subplot(3, 1, 2, sharex=ax1)
    
    ax2.plot(time_hours, moisture, 'g-', linewidth=2, label='Soil Moisture')
    
    # 标记传输事件（根据触发原因）
    for t in transmissions:
        color_map = {
            'interval_trigger': 'blue',
            'change_rate_trigger': 'red',
            'daily_batch': 'purple',
            'force_upload': 'black'
        }
        
        # 确定颜色
        color = 'blue'
        for key, value in color_map.items():
            if key in t['trigger']:
                color = value
                break
        
        ax2.plot(t['minute']/60, t['data'], 
                marker='o', color=color,
                markersize=6, markeredgewidth=1, markeredgecolor='black')
    
    # 添加变化率阈值线
    ax2.axhline(y=30 * 1.05, color='r', linestyle='--', alpha=0.5, label='5% Change Threshold')
    ax2.axhline(y=30 * 0.95, color='r', linestyle='--', alpha=0.5)
    
    ax2.set_ylabel('Moisture (%)')
    ax2.set_title('Soil Moisture & Transmission Triggers')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # 子图3：传输间隔直方图
    ax3 = plt.subplot(3, 1, 3)
    
    # 计算传输事件之间的间隔（分钟）
    if len(transmit_times) > 1:
        intervals = np.diff(transmit_times)
        
        # 按状态分组间隔，以便观察不同状态下的传输频率
        state_intervals = {'FULL': [], 'MID': [], 'LOW': []}
        for i in range(len(intervals)):
            state = transmit_states[i]
            state_intervals[state].append(intervals[i])
        
        # 绘制直方图，为每个状态使用不同的颜色
        colors = ['green', 'orange', 'red']
        labels = ['FULL State', 'MID State', 'LOW State']
        
        for idx, (state, color, label) in enumerate(zip(['FULL', 'MID', 'LOW'], colors, labels)):
            if state_intervals[state]:
                ax3.hist(state_intervals[state], bins=20, alpha=0.6, 
                        color=color, label=f'{label} (n={len(state_intervals[state])})')
        
        # 添加目标间隔线，用虚线表示预期的传输间隔
        ax3.axvline(x=10, color='green', linestyle='--', alpha=0.7, label='Target: 10min (FULL)')
        ax3.axvline(x=30, color='orange', linestyle='--', alpha=0.7, label='Target: 30min (MID)')
        ax3.axvline(x=1440, color='red', linestyle='--', alpha=0.7, label='Target: 1440min (LOW)')
    
    ax3.set_xlabel('Transmission Interval (minutes)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Distribution of Transmission Intervals by Energy State')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 100)  # 聚焦在较短间隔
    
    plt.tight_layout()
    
    # 添加统计信息文本框
    stats_text = (
        f"Energy Savings: {result['savings']['energy_saved_percent']:.1f}%\n"
        f"Total Transmissions: {result['savings']['actual_transmissions']}\n"
        f"Baseline (10min interval): {result['savings']['baseline_transmissions']}\n"
        f"FULL State: {result['savings']['transmissions_by_state'][EnergyState.FULL]}\n"
        f"MID State: {result['savings']['transmissions_by_state'][EnergyState.MID]}\n"
        f"LOW State: {result['savings']['transmissions_by_state'][EnergyState.LOW]}"
    )
    
    plt.figtext(0.02, 0.02, stats_text, fontsize=9, 
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
    
    # 保存图表
    output_dir = '../docs/images'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/energy_state_demo.png', dpi=150, bbox_inches='tight')
    print("✅ 能量状态管理图表已保存至 docs/images/energy_state_demo.png")
    
    plt.show()
    
    return result

def print_detailed_statistics(result):
    """
    打印能量状态管理器的详细统计信息
    
    这个函数分析并输出模拟结果，包括：
    - 传输统计：总次数、基线、节能效果
    - 状态统计：每个状态下的传输次数
    - 触发原因：各种触发条件的分布
    - 间隔分析：传输间隔的统计特性
    - 状态转换：状态切换次数和模式
    
    Args:
        result: simulate_energy_aware_transmission返回的结果字典
    """
    print("\n" + "="*60)
    print("能量状态管理器详细统计")
    print("="*60)
    
    transmissions = result['transmissions']
    savings = result['savings']
    
    # 按触发原因统计
    triggers = {}
    for t in transmissions:
        trigger = t['trigger']
        if 'change_rate' in trigger:
            trigger = 'change_rate'
        triggers[trigger] = triggers.get(trigger, 0) + 1
    
    print(f"\n📊 传输统计:")
    print(f"  总传输次数: {savings['actual_transmissions']}")
    print(f"  基线传输次数: {savings['baseline_transmissions']}")
    print(f"  节能: {savings['energy_saved_percent']:.1f}%")
    
    print(f"\n🎯 按状态统计:")
    for state, count in savings['transmissions_by_state'].items():
        print(f"  {state}: {count} 次传输")
    
    print(f"\n⚡ 按触发原因统计:")
    for trigger, count in triggers.items():
        percentage = count / savings['actual_transmissions'] * 100
        print(f"  {trigger}: {count} 次 ({percentage:.1f}%)")
    
    # 计算平均间隔
    if len(transmissions) > 1:
        intervals = np.diff([t['minute'] for t in transmissions])
        print(f"\n⏱️ 传输间隔分析:")
        print(f"  平均间隔: {np.mean(intervals):.1f} 分钟")
        print(f"  最小间隔: {np.min(intervals):.1f} 分钟")
        print(f"  最大间隔: {np.max(intervals):.1f} 分钟")
        print(f"  间隔标准差: {np.std(intervals):.1f} 分钟")
    
    print(f"\n🔄 状态切换次数: {result['status']['stats']['state_changes']}")
    print(f"  当前状态: {result['status']['current_state']}")
    print(f"  本地存储数据点数: {result['status']['local_storage_count']}")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    print("运行能量状态管理器演示...")
    
    # 运行演示
    result = plot_energy_state_demo()
    
    # 打印详细统计
    print_detailed_statistics(result)
    
    # 运行验收测试
    print("\n运行验收测试...")
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../cloud/model'))
    from tests.test_energy_manager import test_acceptance_criteria
    try:
        test_acceptance_criteria()
        print("🎉 演示完成！能量状态管理器运行正常。")
    except Exception as e:
        print(f"⚠️  验收测试出现异常: {e}")