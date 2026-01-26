# examples/simulator_demo.py
"""
模拟器使用示例与可视化
运行此文件可生成示例图表
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../cloud/model'))

import numpy as np
import matplotlib.pyplot as plt
from simulator.soil_moisture import simulate_soil_moisture, generate_soil_temperature
from simulator.energy import simulate_battery_voltage, simulate_energy_harvest
from simulator.weather_mock import mock_weather_api, generate_hourly_weather

def plot_soil_moisture():
    """绘制土壤湿度模拟结果"""
    plt.figure(figsize=(12, 8))
    
    # 子图1：7天湿度变化
    plt.subplot(3, 2, 1)
    moisture = simulate_soil_moisture(days=7, rain_events=3, seed=42)
    plt.plot(moisture, label='Soil Moisture')
    plt.title("7-Day Soil Moisture Simulation")
    plt.xlabel("Hour")
    plt.ylabel("Moisture (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图2：不同采样频率
    plt.subplot(3, 2, 2)
    moisture_1h = simulate_soil_moisture(days=3, sampling_interval=1, seed=123)
    moisture_6h = simulate_soil_moisture(days=3, sampling_interval=6, seed=123)
    
    plt.plot(np.arange(0, 72, 1), moisture_1h, label='1-hour interval', alpha=0.7)
    plt.plot(np.arange(0, 72, 6), moisture_6h, 'o-', label='6-hour interval')
    plt.title("Sampling Frequency Comparison")
    plt.xlabel("Hour")
    plt.ylabel("Moisture (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图3：温度变化
    plt.subplot(3, 2, 3)
    temp = generate_soil_temperature(days=2, seed=456)
    plt.plot(temp, label='Soil Temperature', color='orange')
    plt.title("2-Day Soil Temperature")
    plt.xlabel("Hour")
    plt.ylabel("Temperature (°C)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图4：电池电压
    plt.subplot(3, 2, 4)
    voltage = simulate_battery_voltage(days=5, solar_peak_hours=8, seed=789)
    plt.plot(voltage, label='Battery Voltage', color='green')
    plt.axhline(y=4.0, color='r', linestyle='--', label='FULL threshold')
    plt.axhline(y=3.6, color='orange', linestyle='--', label='MID threshold')
    plt.axhline(y=3.0, color='r', linestyle='--', label='LOW threshold')
    plt.title("5-Day Battery Voltage")
    plt.xlabel("Hour")
    plt.ylabel("Voltage (V)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图5：能量收集三源对比
    plt.subplot(3, 2, 5)
    energy = simulate_energy_harvest(days=1, seed=101)
    plt.stackplot(range(24), 
                  energy['solar'], 
                  energy['thermal'], 
                  energy['vibration'],
                  labels=['Solar', 'Thermal', 'Vibration'])
    plt.title("24-Hour Energy Harvesting")
    plt.xlabel("Hour")
    plt.ylabel("Power (mW)")
    plt.legend()
    
    # 子图6：天气API数据
    plt.subplot(3, 2, 6)
    weather = mock_weather_api(days=5, seed=202)
    dates = [d['fxDate'] for d in weather['daily']]
    temp_max = [d['tempMax'] for d in weather['daily']]
    temp_min = [d['tempMin'] for d in weather['daily']]
    
    plt.plot(dates, temp_max, 'r-o', label='Max Temp')
    plt.plot(dates, temp_min, 'b-o', label='Min Temp')
    plt.title("5-Day Weather Forecast")
    plt.xlabel("Date")
    plt.ylabel("Temperature (°C)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    plt.tight_layout()

    output_dir = '../docs/images'
    os.makedirs(output_dir, exist_ok=True)  # 自动创建目录
    plt.savefig(f'{output_dir}/simulator_demo.png', dpi=150, bbox_inches='tight')
    print("✅ 图表已保存至 docs/images/simulator_demo.png")

def print_statistics():
    """打印模拟数据统计信息"""
    print("\n=== 模拟器统计数据 ===")
    
    # 土壤湿度
    moisture = simulate_soil_moisture(days=30, seed=42)
    print(f"土壤湿度: 均值={np.mean(moisture):.2f}%, 范围=[{np.min(moisture):.1f}, {np.max(moisture):.1f}]%")
    
    # 电池电压
    voltage = simulate_battery_voltage(days=30, seed=42)
    print(f"电池电压: 均值={np.mean(voltage):.2f}V, 范围=[{np.min(voltage):.2f}, {np.max(voltage):.2f}]V")
    
    # 能量收集
    energy = simulate_energy_harvest(days=1, seed=42)
    total_energy = {k: np.sum(v) for k, v in energy.items()}
    print(f"日能量收集: {total_energy}")
    
    # 天气
    weather = mock_weather_api(days=3, seed=42)
    print(f"天气数据: {len(weather['daily'])}天预报，城市坐标={weather.get('location', '北京')}")

if __name__ == "__main__":
    plot_soil_moisture()
    print_statistics()
