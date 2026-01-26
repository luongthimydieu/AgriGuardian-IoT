# cloud/model/simulator/soil_moisture.py
"""
土壤湿度传感器数据模拟器
模拟真实场景的降雨-渗透-蒸发过程
"""

import numpy as np
from typing import Tuple

def simulate_soil_moisture(
    days: int = 30,
    sampling_interval: int = 1,  # 小时
    base_moisture: float = 25.0,  # 基础湿度 %
    rain_events: int = 5,
    rain_intensity: Tuple[float, float] = (20, 40),  # 降雨强度范围
    evaporation_rate: float = 0.05,  # 每小时蒸发率
    noise_std: float = 2.0,  # 传感器噪声
    seed: int = None
) -> np.ndarray:
    """
    模拟土壤湿度时序数据
    
    物理模型：
    - 白天(6-18时)蒸发作用强
    - 降雨事件随机发生，持续2-6小时
    - 雨后湿度缓慢渗透和蒸发
    - 添加高斯白噪声模拟传感器误差
    
    返回: 形状为 (days * 24 // sampling_interval,) 的数组
    """
    if seed is not None:
        np.random.seed(seed)
    
    total_hours = days * 24
    time_points = np.arange(0, total_hours, sampling_interval)
    n_points = len(time_points)
    
    # 初始化湿度数组
    moisture = np.full(n_points, base_moisture, dtype=float)
    
    # 生成降雨事件
    for _ in range(rain_events):
        # 随机选择降雨开始时间（避开最后12小时）
        rain_start = np.random.randint(0, total_hours - 12)
        rain_duration = np.random.randint(2, 7)  # 持续2-6小时
        rain_amount = np.random.uniform(*rain_intensity)
        
        # 将降雨转换为数组索引
        start_idx = rain_start // sampling_interval
        end_idx = min((rain_start + rain_duration) // sampling_interval, n_points)
        
        # 降雨期间湿度快速上升（指数衰减模型）
        t = np.arange(end_idx - start_idx)
        moisture[start_idx:end_idx] += rain_amount * np.exp(-t * 0.5)
    
    # 蒸发模型：白天蒸发强，夜间蒸发弱
    day_night_cycle = np.sin(2 * np.pi * time_points / 24)
    evaporation = evaporation_rate * (1 + 0.5 * np.maximum(0, day_night_cycle))
    moisture -= evaporation
    
    # 添加传感器噪声
    moisture += np.random.normal(0, noise_std, n_points)
    
    # 限制合理范围 [10%, 90%]
    return np.clip(moisture, 10, 90)


def generate_soil_temperature(
    days: int = 30,
    base_temp: float = 20.0,  # 基础温度 °C
    amplitude: float = 8.0,  # 日温差幅度
    seed: int = None
) -> np.ndarray:
    """模拟土壤温度（正弦波 + 随机扰动）"""
    if seed is not None:
        np.random.seed(seed + 1)
    
    total_hours = days * 24
    t = np.arange(total_hours)
    
    # 温度日变化 + 季节性变化
    temp = base_temp + amplitude * np.sin(2 * np.pi * t / 24)
    temp += 3 * np.sin(2 * np.pi * t / (24 * 30))  # 月周期
    
    return temp + np.random.normal(0, 0.5, total_hours)
