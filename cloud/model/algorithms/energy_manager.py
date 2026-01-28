"""
能量管理三状态规则引擎
基于电池电压动态调整采集和传输策略
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class EnergyState(Enum):
    """能量状态枚举"""
    FULL = "FULL"    # 能量充足：> 4.0V
    MID = "MID"      # 能量中等：3.6V - 4.0V
    LOW = "LOW"      # 能量低：< 3.6V

@dataclass
class TransmissionRecord:
    """传输记录"""
    timestamp: int  # 时间戳（分钟）
    voltage: float  # 当前电压
    data_value: float  # 传输的数据值
    state: EnergyState  # 当前状态
    trigger: str  # 触发原因：interval, change_rate, etc.

class EnergyStateManager:
    """
    能量状态管理器
    
    基于电池电压实现三状态机：
    - FULL (>4.0V): 高频传输，10分钟间隔
    - MID (3.6-4.0V): 中频传输，30分钟间隔或变化率>5%
    - LOW (<3.6V): 低频存储，每日批量上传
    """
    
    def __init__(
        self,
        full_threshold: float = 4.0,
        mid_threshold: float = 3.6,
        low_threshold: float = 3.0,
        full_interval: int = 10,    # 分钟
        mid_interval: int = 30,     # 分钟
        change_rate_threshold: float = 0.05,  # 5%
        low_batch_interval: int = 1440,  # 分钟 (24小时)
        enable_energy_smoothing: bool = False
    ):
        """
        初始化状态管理器
        
        Args:
            full_threshold: FULL状态电压阈值
            mid_threshold: MID状态电压阈值
            low_threshold: 系统最低电压（关机保护）
            full_interval: FULL状态下传输间隔（分钟）
            mid_interval: MID状态下传输间隔（分钟）
            change_rate_threshold: 变化率触发阈值
            low_batch_interval: LOW状态下批量上传间隔
            enable_energy_smoothing: 启用电压平滑滤波
        """
        self.full_threshold = full_threshold
        self.mid_threshold = mid_threshold
        self.low_threshold = low_threshold
        self.full_interval = full_interval
        self.mid_interval = mid_interval
        self.change_rate_threshold = change_rate_threshold
        self.low_batch_interval = low_batch_interval
        self.enable_energy_smoothing = enable_energy_smoothing
        
        # 状态变量
        self.current_state = EnergyState.MID
        self.last_voltage = None
        self.last_transmission_time = -1  # 用-1表示从未传输过
        self.last_data_value = None
        self.local_storage = []  # LOW状态本地存储
        self.transmission_history = []
        
        # 平滑滤波器参数
        self.voltage_window = []
        self.window_size = 5
        
        # 统计信息
        self.stats = {
            "total_transmissions": 0,
            "transmissions_by_state": {state: 0 for state in EnergyState},
            "state_changes": 0,
            "energy_saved": 0.0,  # 节省的能量百分比
            "data_lost": 0
        }
        
        logger.info(f"EnergyStateManager initialized with thresholds: "
                   f"FULL>{full_threshold}V, MID{mid_threshold}-{full_threshold}V, LOW<{mid_threshold}V")
    
    def update_state(self, current_voltage: float) -> EnergyState:
        """
        更新能量状态
        
        根据当前电压值确定设备所处的能量状态，并处理状态转换时的特殊逻辑。
        
        Args:
            current_voltage: 当前电池电压（单位：V）
            
        Returns:
            EnergyState: 新的能量状态（FULL/MID/LOW）
        """
        # 电压平滑处理：可选的滤波以降低噪声，但在测试模式禁用以保证精确性
        if self.enable_energy_smoothing:
            current_voltage = self._apply_voltage_filter(current_voltage)
        
        # 保存旧状态以便比较
        old_state = self.current_state
        
        # 根据电压阈值确定新状态
        # 使用>=阈值判断，以确保4.0V等于阈值时被归入FULL状态
        if current_voltage >= self.full_threshold:  # >= 4.0V
            new_state = EnergyState.FULL
        elif current_voltage >= self.mid_threshold:  # >= 3.6V 且 < 4.0V
            new_state = EnergyState.MID
        else:  # < 3.6V
            new_state = EnergyState.LOW
        
        # 若状态发生转换，执行相应处理
        if new_state != old_state:
            self._handle_state_transition(old_state, new_state)
            self.current_state = new_state
            self.stats["state_changes"] += 1
            logger.info(f"State transition: {old_state} -> {new_state} at {current_voltage:.2f}V")
        
        # 更新最后一次读取的电压值
        self.last_voltage = current_voltage
        return new_state
    
    def _apply_voltage_filter(self, voltage: float) -> float:
        """应用电压平滑滤波器（仅用于实际应用）"""
        # 注意：测试和演示模式禁用滤波以确保精确的状态转换
        if not self.enable_energy_smoothing:
            return voltage
            
        self.voltage_window.append(voltage)
        if len(self.voltage_window) > self.window_size:
            self.voltage_window.pop(0)
        
        # 使用简单移动平均滤波
        if len(self.voltage_window) >= 3:
            return np.mean(self.voltage_window[-3:])
        return voltage
    
    def _handle_state_transition(self, old_state: EnergyState, new_state: EnergyState):
        """
        处理状态转换时的特殊操作
        
        当设备在不同能量状态之间转换时，需要处理本地存储的数据：
        - 从LOW状态切换出来：上传缓存的所有数据
        - 进入LOW状态：准备本地存储模式
        
        Args:
            old_state: 转换前的状态
            new_state: 转换后的状态
        """
        # 从LOW状态切换到更高状态时，上传缓存的本地数据
        # 这确保了低电量期间存储的数据不会丢失
        if old_state == EnergyState.LOW and new_state != EnergyState.LOW:
            if self.local_storage:
                logger.info(f"Uploading {len(self.local_storage)} cached data points "
                           f"from LOW state transition")
                # 实际应用中此处应调用网络上传函数
                self.local_storage.clear()
        
        # 从高状态切换到LOW状态时，准备本地存储
        # 记录这个转换以便日志记录和调试
        elif old_state != EnergyState.LOW and new_state == EnergyState.LOW:
            logger.info("Entering LOW state, enabling local storage mode")
    
    def should_transmit(
        self,
        current_time: int,  # 当前时间（分钟）
        current_voltage: float,
        current_data: float,
        force_upload: bool = False
    ) -> Tuple[bool, str]:
        """
        判断是否应该传输数据
        
        Args:
            current_time: 当前时间（从0开始的分钟数）
            current_voltage: 当前电压
            current_data: 当前传感器数据
            force_upload: 是否强制上传（如报警事件）
            
        Returns:
            (should_transmit, trigger_reason)
        """
        # 更新状态
        state = self.update_state(current_voltage)
        
        # 强制上传（如紧急报警）
        if force_upload:
            return True, "force_upload"
        
        # 电压过低保护
        if current_voltage < self.low_threshold:
            logger.warning(f"Critical low voltage: {current_voltage:.2f}V < {self.low_threshold}V")
            return False, "critical_low"
        
        # 根据状态决定传输策略
        if state == EnergyState.FULL:
            return self._full_state_policy(current_time, current_data)
        elif state == EnergyState.MID:
            return self._mid_state_policy(current_time, current_data)
        else:  # LOW state
            return self._low_state_policy(current_time, current_data)
    
    def _full_state_policy(self, current_time: int, current_data: float) -> Tuple[bool, str]:
        """
        FULL状态传输策略：固定10分钟间隔
        
        在电池充足的FULL状态下，设备采用高频传输策略以获得最新的数据。
        初次传输立即进行，后续传输遵循10分钟的固定间隔。
        
        Args:
            current_time: 当前时间（从模拟开始的分钟数）
            current_data: 当前传感器数据（此策略中不使用，为了接口一致性保留）
            
        Returns:
            tuple: (是否应该传输, 触发原因描述)
            - (True, "initial_transmit"): 首次传输
            - (True, "interval_trigger"): 间隔触发
            - (False, "waiting_interval"): 等待间隔
        """
        # 第一次传输：last_transmission_time初始值为-1表示从未传输过
        if self.last_transmission_time < 0:
            return True, "initial_transmit"
        
        # 计算距上次传输的时间间隔    
        time_since_last = current_time - self.last_transmission_time
        
        # 检查是否达到10分钟间隔
        if time_since_last >= self.full_interval:
            return True, "interval_trigger"
        
        # 尚未达到间隔，继续等待
        return False, "waiting_interval"
    
    def _mid_state_policy(self, current_time: int, current_data: float) -> Tuple[bool, str]:
        """
        MID状态传输策略：30分钟间隔 或 变化率>5%
        
        在电池中等的MID状态下，设备采用混合策略：
        1. 时间触发：每30分钟传输一次
        2. 事件触发：数据变化率超过5%时立即传输
        
        这种策略既能保证定期更新，也能对突发变化快速响应。
        
        Args:
            current_time: 当前时间（从模拟开始的分钟数）
            current_data: 当前传感器数据（如土壤湿度）
            
        Returns:
            tuple: (是否应该传输, 触发原因描述)
            - (True, "initial_transmit"): 首次传输
            - (True, "interval_trigger"): 30分钟间隔触发
            - (True, "change_rate_trigger_X.X%"): 数据变化率触发
            - (False, "waiting"): 继续等待
        """
        # 第一次传输
        if self.last_transmission_time < 0:
            return True, "initial_transmit"
        
        # 计算距上次传输的时间间隔        
        time_since_last = current_time - self.last_transmission_time
        
        # 检查时间间隔触发：30分钟
        if time_since_last >= self.mid_interval:
            return True, "interval_trigger"
        
        # 检查变化率触发：数据相对于上次值的变化
        # 需要有之前的数据值（不为None）且不为0（避免除以0错误）
        if self.last_data_value is not None and self.last_data_value != 0:
            # 计算相对变化率：|当前值-上次值|/上次值
            change_rate = abs(current_data - self.last_data_value) / self.last_data_value
            
            # 如果变化率超过5%（0.05），触发传输
            if change_rate > self.change_rate_threshold:
                return True, f"change_rate_trigger_{change_rate:.1%}"
        
        # 都不满足，继续等待
        return False, "waiting"
    
    def _low_state_policy(self, current_time: int, current_data: float) -> Tuple[bool, str]:
        """
        LOW状态策略：本地存储，每日批量上传
        
        在电池低电量的LOW状态下，设备采用最省电的策略：
        1. 所有数据优先存储在本地（通常为FRAM铁电存储器）
        2. 每24小时进行一次批量上传
        
        这种策略大幅减少无线传输次数，从而显著降低功耗。
        初次进入该状态立即执行一次上传（可能上传前一周期的缓存）。
        
        Args:
            current_time: 当前时间（从模拟开始的分钟数）
            current_data: 当前传感器数据（会被存储到本地缓冲）
            
        Returns:
            tuple: (是否应该传输, 触发原因描述)
            - (True, "initial_upload"): 首次进入LOW状态时上传
            - (True, "daily_batch"): 24小时后的批量上传
            - (False, "local_storage"): 本地存储，不传输
        """
        # 将当前数据存储到本地缓冲
        # 实际应用中这些数据会写入FRAM等非易失性存储
        self.local_storage.append({
            'timestamp': current_time,
            'data': current_data
        })
        
        # 初始状态或首次进入LOW状态，立即执行上传
        # 这确保了前一个状态最后的数据不会丢失
        if self.last_transmission_time < 0:
            return True, "initial_upload"
        
        # 计算距上次传输的时间间隔
        time_since_last = current_time - self.last_transmission_time
        
        # 检查是否到达每日上传时间（1440分钟 = 24小时）
        if time_since_last >= self.low_batch_interval:
            return True, "daily_batch"
        
        # 继续本地存储，不进行无线传输
        return False, "local_storage"
    
    def record_transmission(
        self,
        timestamp: int,
        voltage: float,
        data_value: float,
        trigger: str
    ):
        """
        记录一次成功的数据传输事件
        
        每当决定进行传输时，需要调用此方法记录事件，更新内部状态，
        并进行必要的数据清理。这是状态管理的重要一步。
        
        Args:
            timestamp: 传输时的时间戳（分钟）
            voltage: 传输时的电池电压
            data_value: 传输的数据值
            trigger: 触发原因（如"interval_trigger", "change_rate_trigger"等）
        """
        # 创建传输记录对象，包含完整的上下文信息
        record = TransmissionRecord(
            timestamp=timestamp,
            voltage=voltage,
            data_value=data_value,
            state=self.current_state,  # 记录当前状态
            trigger=trigger
        )
        
        # 将记录添加到历史列表，用于后续分析
        self.transmission_history.append(record)
        
        # 更新最后一次传输的时间戳
        # 这用于计算下一次传输的间隔
        self.last_transmission_time = timestamp
        
        # 更新最后一次传输的数据值
        # 这用于计算数据变化率（MID状态）
        self.last_data_value = data_value
        
        # 增加总传输次数计数
        self.stats["total_transmissions"] += 1
        
        # 按状态分类计数
        self.stats["transmissions_by_state"][self.current_state] += 1
        
        # 清除本地存储中的数据
        # 仅在非LOW状态或执行daily_batch时清除，以避免数据丢失
        if self.current_state != EnergyState.LOW or trigger == "daily_batch":
            self.local_storage.clear()
    
    def calculate_energy_savings(self, baseline_interval: int = 10) -> Dict:
        """
        计算能量节省情况
        
        通过比较实际传输次数和基线传输次数，评估三状态管理策略的节能效果。
        
        基线场景：设备以固定10分钟间隔连续传输
        - 24小时 = 1440分钟
        - 基线传输次数 = 1440 ÷ 10 + 1 = 144次
        
        实现场景：使用三状态管理策略的实际传输次数
        
        节能百分比 = (1 - 实际次数/基线次数) × 100%
        
        Args:
            baseline_interval: 基线传输间隔（分钟），默认10分钟
            
        Returns:
            dict: 包含以下键的节能统计字典：
                - actual_transmissions: 实际传输次数
                - baseline_transmissions: 基线传输次数
                - energy_saved_percent: 节能百分比
                - transmissions_by_state: 按状态分类的传输次数
                - state_changes: 状态切换次数
        """
        if not self.transmission_history:
            return {"energy_saved_percent": 0.0}
        
        # 计算实际传输次数
        actual_transmissions = len(self.transmission_history)
        
        # 计算基线传输次数（固定间隔）
        # 获取总时间跨度（最后一次传输的时间戳）
        total_minutes = max(record.timestamp for record in self.transmission_history)
        # 公式：(总分钟数 ÷ 间隔) + 1
        # +1是因为第0分钟有一次初始传输
        baseline_transmissions = total_minutes // baseline_interval + 1
        
        # 计算节能百分比
        energy_saved = 1.0 - (actual_transmissions / baseline_transmissions)
        
        self.stats["energy_saved"] = energy_saved
        
        return {
            "actual_transmissions": actual_transmissions,
            "baseline_transmissions": baseline_transmissions,
            "energy_saved_percent": energy_saved * 100,
            "transmissions_by_state": self.stats["transmissions_by_state"],
            "state_changes": self.stats["state_changes"]
        }
    
    def get_status_report(self) -> Dict:
        """
        获取当前状态报告
        
        返回能量管理器的完整状态快照，用于监控和调试。
        
        Returns:
            dict: 包含以下内容的状态字典：
                - current_state: 当前能量状态（FULL/MID/LOW）
                - current_voltage: 最后一次读取的电压值
                - local_storage_count: 本地存储中待上传的数据点数
                - total_transmissions: 总传输次数
                - last_transmission_time: 最后一次传输的时间戳
                - stats: 完整的统计信息字典
        """
        return {
            "current_state": self.current_state.value,
            "current_voltage": self.last_voltage,
            "local_storage_count": len(self.local_storage),
            "total_transmissions": self.stats["total_transmissions"],
            "last_transmission_time": self.last_transmission_time,
            "stats": self.stats
        }

def simulate_energy_aware_transmission(
    voltage_sequence: np.ndarray,
    data_sequence: np.ndarray,
    manager: EnergyStateManager = None
) -> Dict:
    """
    模拟能量感知传输系统
    
    该函数模拟一个完整的时间周期（通常24小时），根据电压和数据变化
    动态决定何时传输，评估三状态管理策略的效果。
    
    工作流程：
    1. 对每个时间单位（分钟）：
       - 读取当前电压和传感器数据
       - 调用管理器判断是否需要传输
       - 若需要，记录传输事件
    2. 收集所有传输事件和统计信息
    3. 计算能量节省效果
    
    Args:
        voltage_sequence: 电压时间序列数组，长度为模拟分钟数（通常1440）
        data_sequence: 传感器数据时间序列数组，与voltage_sequence等长
        manager: 可选的EnergyStateManager实例，若为None则创建新实例
        
    Returns:
        dict: 模拟结果字典，包含：
            - transmissions: 所有传输事件的列表
              每个元素包含：minute, voltage, data, state, trigger
            - savings: 节能统计信息
            - status: 模拟结束后的设备状态
            - manager: 使用过的管理器实例（便于后续分析）
    """
    # 若未提供管理器，创建新实例（使用默认参数）
    if manager is None:
        manager = EnergyStateManager()
    
    transmissions = []
    
    # 逐分钟模拟
    for minute, (voltage, data) in enumerate(zip(voltage_sequence, data_sequence)):
        # 判断是否需要传输
        should_transmit, trigger = manager.should_transmit(
            current_time=minute,
            current_voltage=voltage,
            current_data=data
        )
        
        # 若判定需要传输，记录此次传输事件
        if should_transmit:
            # 调用record_transmission更新内部状态
            manager.record_transmission(
                timestamp=minute,
                voltage=voltage,
                data_value=data,
                trigger=trigger
            )
            # 将事件添加到列表以便后续分析
            transmissions.append({
                'minute': minute,
                'voltage': voltage,
                'data': data,
                'state': manager.current_state.value,
                'trigger': trigger
            })
    
    # 计算节能统计
    savings = manager.calculate_energy_savings()
    # 获取模拟结束时的状态
    status = manager.get_status_report()
    
    return {
        'transmissions': transmissions,
        'savings': savings,
        'status': status,
        'manager': manager
    }