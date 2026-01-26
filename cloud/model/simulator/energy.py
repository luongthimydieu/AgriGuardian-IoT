# cloud/model/simulator/energy.py
"""
能量收集与电池状态模拟器
模拟太阳能板 + 锂电池的充放电过程
"""

import numpy as np
from typing import Literal

def simulate_battery_voltage(
    days: int = 30,
    solar_peak_hours: float = 8.0,  # 每日有效光照小时数
    battery_capacity: float = 2000,  # mAh
    base_load: float = 10,  # 基础负载电流 mA
    seed: int = None
) -> np.ndarray:
    """
    模拟电池电压变化
    
    充电模型：白天(sin曲线)充电，夜间自放电
    电压范围：3.0V(欠压) ~ 4.2V(满电)
    """
    if seed is not None:
        np.random.seed(seed + 2)
    
    total_hours = days * 24
    t = np.arange(total_hours)
    
    # 太阳能充电曲线（正弦波模拟日照强度）
    day_night = np.sin(2 * np.pi * (t - 6) / 24)  # 6点日出
    solar_power = np.maximum(0, day_night) * (battery_capacity / solar_peak_hours)
    
    # 添加天气影响（阴天充电效率降低）
    weather_factor = np.random.choice([1.0, 0.6, 0.2], total_hours, p=[0.6, 0.3, 0.1])
    solar_power *= weather_factor
    
    # 充放电导致的电量变化
    charge_rate = 0.8  # 充电效率
    energy_level = np.cumsum(solar_power * charge_rate - base_load)
    energy_level = np.maximum(0, energy_level)  # 不能低于0
    
    # 将电量转换为电压（简化模型）
    soc = energy_level / battery_capacity  # 荷电状态 0-1
    voltage = 3.0 + soc * 1.2  # 3.0V ~ 4.2V
    
    return np.clip(voltage, 3.0, 4.2)


def simulate_energy_harvest(
    days: int = 30,
    seed: int = None
) -> dict:
    """
    多源能量收集模拟器
    
    返回包含三源的dict：
    - solar: 太阳能 (mW)
    - thermal: 温差能 (mW)
    - vibration: 振动能 (mW)
    """
    if seed is not None:
        np.random.seed(seed + 3)
    
    total_hours = days * 24
    t = np.arange(total_hours)
    
    # 太阳能（主导）
    solar = 100 * np.maximum(0, np.sin(2 * np.pi * (t - 6) / 24))
    
    # 温差能（夜间稍高）
    thermal = 5 + 3 * np.sin(2 * np.pi * t / 24 + np.pi)
    
    # 振动能（随机事件）
    vibration = np.random.poisson(2, total_hours) * np.random.uniform(1, 5, total_hours)
    
    return {
        'solar': solar,
        'thermal': thermal,
        'vibration': vibration
    }
