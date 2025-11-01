"""
二自由度悬架系统求解器
基于物理模型：
- m2: 簧载质量（车身质量）
- m1: 非簧载质量（车轮质量）
- k2: 悬架刚度
- k1: 轮胎刚度
- c: 悬架阻尼
- x0: 路面激励
- x1: 非簧载质量位移
- x2: 簧载质量位移

动力学方程：
m2*ẍ2 + c(ẋ2 - ẋ1) + k2(x2 - x1) = 0
m1*ẍ1 + c(ẋ1 - ẋ2) + k2(x1 - x2) + k1(x1 - x0) = 0
"""

import numpy as np
from scipy.integrate import odeint
from scipy.linalg import eig


class SuspensionSolver:
    """悬架系统求解器"""

    def __init__(self, m1, m2, k1, k2, c, road_excitation=None):
        """
        参数:
            m1: 非簧载质量 (kg)
            m2: 簧载质量 (kg)
            k1: 轮胎刚度 (N/m)
            k2: 悬架刚度 (N/m)
            c: 悬架阻尼系数 (N·s/m)
            road_excitation: 路面激励对象
        """
        self.m1 = m1
        self.m2 = m2
        self.k1 = k1
        self.k2 = k2
        self.c = c
        self.road_excitation = road_excitation

        # 组装系统矩阵
        self._assemble_matrices()

        # 计算固有频率和阻尼比
        self._calculate_natural_properties()

    def _assemble_matrices(self):
        """组装质量、刚度、阻尼矩阵"""
        # 质量矩阵 M
        self.M = np.array([
            [self.m1, 0],
            [0, self.m2]
        ])

        # 刚度矩阵 K
        self.K = np.array([
            [self.k1 + self.k2, -self.k2],
            [-self.k2, self.k2]
        ])

        # 阻尼矩阵 C
        self.C = np.array([
            [self.c, -self.c],
            [-self.c, self.c]
        ])

    def _calculate_natural_properties(self):
        """计算固有频率和模态阻尼比"""
        # 求解广义特征值问题: K*Φ = λ*M*Φ
        eigenvalues, eigenvectors = eig(self.K, self.M)

        # 固有频率 (Hz)
        self.natural_frequencies = np.sqrt(np.real(eigenvalues)) / (2 * np.pi)
        self.natural_frequencies = np.sort(self.natural_frequencies)

        # 固有圆频率 (rad/s)
        self.omega_n = 2 * np.pi * self.natural_frequencies

        # 计算模态阻尼比
        self.damping_ratios = []
        for omega in self.omega_n:
            # 对于比例阻尼，模态阻尼比可以近似计算
            zeta = self.c / (2 * np.sqrt(self.k2 * self.m2))
            self.damping_ratios.append(zeta)
        self.damping_ratios = np.array(self.damping_ratios)

    def _dynamics_equations(self, state, t):
        """
        状态空间形式的动力学方程
        state = [x1, x2, v1, v2]
        """
        x1, x2, v1, v2 = state

        # 获取路面激励
        if self.road_excitation is not None:
            x0 = self.road_excitation.get_displacement(t)
            v0 = self.road_excitation.get_velocity(t)
        else:
            x0 = 0
            v0 = 0

        # 动力学方程
        # m1*a1 + c(v1 - v2) + k2(x1 - x2) + k1(x1 - x0) = 0
        a1 = (-self.c * (v1 - v2) - self.k2 * (x1 - x2) - self.k1 * (x1 - x0)) / self.m1

        # m2*a2 + c(v2 - v1) + k2(x2 - x1) = 0
        a2 = (-self.c * (v2 - v1) - self.k2 * (x2 - x1)) / self.m2

        return [v1, v2, a1, a2]

    def solve_dynamic(self, time_span, initial_state=None, n_points=1000):
        """
        求解动力学响应

        参数:
            time_span: (t0, tf) 时间跨度
            initial_state: [x1_0, x2_0, v1_0, v2_0] 初始状态
            n_points: 时间点数量

        返回:
            results: dict 包含时间历程数据
        """
        # 初始条件
        if initial_state is None:
            initial_state = [0, 0, 0, 0]  # 静止状态

        # 时间数组
        t = np.linspace(time_span[0], time_span[1], n_points)

        # 求解ODE
        solution = odeint(self._dynamics_equations, initial_state, t)

        # 提取结果
        x1 = solution[:, 0]  # 非簧载质量位移
        x2 = solution[:, 1]  # 簧载质量位移
        v1 = solution[:, 2]  # 非簧载质量速度
        v2 = solution[:, 3]  # 簧载质量速度

        # 计算加速度
        a1 = np.gradient(v1, t)
        a2 = np.gradient(v2, t)

        # 计算路面激励
        if self.road_excitation is not None:
            x0 = np.array([self.road_excitation.get_displacement(ti) for ti in t])
        else:
            x0 = np.zeros_like(t)

        # 计算悬架行程和轮胎变形
        suspension_travel = x2 - x1  # 悬架行程
        tire_deflection = x1 - x0  # 轮胎变形

        # 计算力
        suspension_force = self.k2 * suspension_travel + self.c * (v2 - v1)
        tire_force = self.k1 * tire_deflection

        # 组织结果
        results = {
            'time': t,
            'x1': x1,  # 非簧载质量位移
            'x2': x2,  # 簧载质量位移
            'v1': v1,  # 非簧载质量速度
            'v2': v2,  # 簧载质量速度
            'a1': a1,  # 非簧载质量加速度
            'a2': a2,  # 簧载质量加速度
            'x0': x0,  # 路面激励
            'suspension_travel': suspension_travel,
            'tire_deflection': tire_deflection,
            'suspension_force': suspension_force,
            'tire_force': tire_force,
            'natural_frequencies': self.natural_frequencies,
            'damping_ratios': self.damping_ratios,
            'parameters': {
                'm1': self.m1,
                'm2': self.m2,
                'k1': self.k1,
                'k2': self.k2,
                'c': self.c
            }
        }

        return results

    def frequency_response(self, freq_range=(0.1, 50), n_points=500):
        """
        计算频率响应函数

        参数:
            freq_range: (f_min, f_max) 频率范围 (Hz)
            n_points: 频率点数量

        返回:
            frequencies: 频率数组
            H_x2: 簧载质量位移传递函数
            H_a2: 簧载质量加速度传递函数
        """
        frequencies = np.linspace(freq_range[0], freq_range[1], n_points)
        omega = 2 * np.pi * frequencies

        H_x2 = np.zeros(n_points, dtype=complex)
        H_a2 = np.zeros(n_points, dtype=complex)

        for i, w in enumerate(omega):
            # 频域传递函数
            # H(ω) = (K - ω²M + iωC)^(-1) * F
            A = self.K - w ** 2 * self.M + 1j * w * self.C

            # 输入为路面激励在轮胎处
            F = np.array([self.k1, 0])

            try:
                X = np.linalg.solve(A, F)
                H_x2[i] = X[1]  # 簧载质量位移
                H_a2[i] = -w ** 2 * X[1]  # 簧载质量加速度
            except:
                H_x2[i] = 0
                H_a2[i] = 0

        return frequencies, np.abs(H_x2), np.abs(H_a2)

    def calculate_comfort_index(self, results):
        """
        计算舒适性指标

        参数:
            results: solve_dynamic() 返回的结果

        返回:
            comfort_metrics: dict 舒适性指标
        """
        a2 = results['a2']
        suspension_travel = results['suspension_travel']
        tire_deflection = results['tire_deflection']

        # RMS加速度（舒适性指标）
        rms_acceleration = np.sqrt(np.mean(a2 ** 2))

        # 最大加速度
        max_acceleration = np.max(np.abs(a2))

        # 悬架动行程利用率
        max_suspension_travel = np.max(np.abs(suspension_travel))

        # 轮胎动载荷系数
        tire_dynamic_load = np.max(np.abs(results['tire_force']))
        tire_static_load = self.m1 * 9.81
        dynamic_load_coefficient = tire_dynamic_load / tire_static_load

        comfort_metrics = {
            'rms_acceleration': rms_acceleration,
            'max_acceleration': max_acceleration,
            'max_suspension_travel': max_suspension_travel,
            'dynamic_load_coefficient': dynamic_load_coefficient,
            'comfort_rating': self._rate_comfort(rms_acceleration)
        }

        return comfort_metrics

    def _rate_comfort(self, rms_acc):
        """根据ISO 2631标准评估舒适性"""
        if rms_acc < 0.315:
            return "非常舒适"
        elif rms_acc < 0.63:
            return "舒适"
        elif rms_acc < 1.0:
            return "一般舒适"
        elif rms_acc < 1.6:
            return "不舒适"
        else:
            return "非常不舒适"