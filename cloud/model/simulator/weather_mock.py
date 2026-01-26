# cloud/model/simulator/weather_mock.py
"""
天气API响应模拟器
模拟和风天气API返回的JSON格式数据
"""

import json
import numpy as np
from typing import Dict, Any
from datetime import datetime, timedelta

def mock_weather_api(
    location: str = "116.40,39.90",  # 北京坐标
    days: int = 3,
    seed: int = None
) -> Dict[str, Any]:
    """
    模拟和风天气API /v7/weather/3d 接口返回
    
    返回结构与实际API一致，包含：
    - code: 状态码
    - daily: 逐日天气预报
    """
    if seed is not None:
        np.random.seed(seed + 4)
    
    # 生成未来日期
    base_date = datetime.now()
    dates = [(base_date + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days)]
    
    # 模拟天气数据
    weather_types = ["晴", "多云", "阴", "小雨", "中雨"]
    weather_codes = ["CLEAR", "CLOUDY", "OVERCAST", "LIGHT_RAIN", "MODERATE_RAIN"]
    
    daily_data = []
    for i, date in enumerate(dates):
        weather_idx = np.random.choice(len(weather_types), p=[0.3, 0.3, 0.2, 0.15, 0.05])
        
        daily_data.append({
            "fxDate": date,
            "tempMax": int(np.random.uniform(20, 35)),
            "tempMin": int(np.random.uniform(10, 20)),
            "textDay": weather_types[weather_idx],
            "iconDay": 100 + weather_idx * 10,
            "textNight": weather_types[min(weather_idx + 1, len(weather_types) - 1)],
            "iconNight": 100 + min(weather_idx + 1, len(weather_types) - 1) * 10,
            "humidity": int(np.random.uniform(40, 90)),  # 湿度 %
            "precip": np.random.uniform(0, 20),  # 降雨量 mm
            "pressure": int(np.random.uniform(1000, 1020)),  # 气压 hPa
            "cloud": int(np.random.uniform(0, 100)),  # 云量 %
            "windSpeedDay": np.random.uniform(0, 15),  # 风速 km/h
            "uvIndex": int(np.random.uniform(1, 10))  # 紫外线指数
        })
    
    return {
        "code": "200",
        "updateTime": base_date.strftime("%Y-%m-%dT%H:%M%z"),
        "daily": daily_data
    }


def generate_hourly_weather(daily_data: Dict, seed: int = None) -> np.ndarray:
    """
    将日天气预报插值为小时数据
    
    用于更精细的算法测试
    """
    if seed is not None:
        np.random.seed(seed + 5)
    
    hourly_precip = []
    for day in daily_data['daily']:
        # 降雨集中在随机4-8小时窗口
        if day['precip'] > 0:
            rain_hours = np.random.randint(4, 9)
            rain_dist = np.random.dirichlet(np.ones(rain_hours)) * day['precip']
            # 将降雨分布到24小时
            padding = np.zeros(24 - rain_hours)
            hourly = np.concatenate([rain_dist, padding])
            np.random.shuffle(hourly)  # 随机打乱
        else:
            hourly = np.zeros(24)
        
        hourly_precip.append(hourly)
    
    return np.array(hourly_precip).flatten()
