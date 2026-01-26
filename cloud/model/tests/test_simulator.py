# cloud/model/tests/test_simulator.py
"""
模拟器单元测试
验证数据生成符合物理规律
"""

import pytest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))

from simulator.soil_moisture import simulate_soil_moisture
from simulator.energy import simulate_battery_voltage, simulate_energy_harvest
from simulator.weather_mock import mock_weather_api, generate_hourly_weather

class TestSoilMoistureSimulator:
    def test_output_shape(self):
        """测试输出形状正确"""
        data = simulate_soil_moisture(days=2, sampling_interval=1)
        assert len(data) == 48  # 2天 * 24小时
        
        data = simulate_soil_moisture(days=1, sampling_interval=6)
        assert len(data) == 4  # 1天 * 24小时 / 6
    
    def test_value_range(self):
        """测试湿度在合理范围内"""
        data = simulate_soil_moisture(days=10, seed=42)
        assert np.all(data >= 10) and np.all(data <= 90)
    
    def test_rain_effect(self):
        """测试降雨事件增加湿度"""
        data = simulate_soil_moisture(days=5, rain_events=3, rain_intensity=(30, 30), seed=42)
        # 应该有明显的湿度峰值
        assert np.max(data) > 40
    
    def test_reproducibility(self):
        """测试随机种子可复现"""
        data1 = simulate_soil_moisture(days=5, seed=123)
        data2 = simulate_soil_moisture(days=5, seed=123)
        assert np.allclose(data1, data2)

class TestEnergySimulator:
    def test_voltage_range(self):
        """测试电压在合理范围"""
        voltage = simulate_battery_voltage(days=3, seed=42)
        assert np.all(voltage >= 3.0) and np.all(voltage <= 4.2)
    
    def test_solar_pattern(self):
        """测试太阳能白天发电"""
        energy = simulate_energy_harvest(days=1, seed=42)
        solar = energy['solar']
        # 白天(6-18时)应该有发电
        assert np.mean(solar[6:18]) > np.mean(solar[:6])
    
    def test_multi_source(self):
        """测试三源能量都有输出"""
        energy = simulate_energy_harvest(days=2, seed=42)
        assert all(len(v) == 48 for v in energy.values())

class TestWeatherMock:
    def test_api_structure(self):
        """测试API返回结构正确"""
        weather = mock_weather_api(location="116.40,39.90", days=3, seed=42)
        assert weather["code"] == "200"
        assert len(weather["daily"]) == 3
        assert "tempMax" in weather["daily"][0]
    
    def test_hourly_interpolation(self):
        """测试小时插值正确"""
        weather = mock_weather_api(days=2, seed=42)
        hourly = generate_hourly_weather(weather, seed=42)
        assert len(hourly) == 48  # 2天 * 24小时
        
        # 降雨小时总和应等于日降雨量
        daily_rain = sum(w['precip'] for w in weather['daily'])
        assert abs(np.sum(hourly) - daily_rain) < 0.1

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
