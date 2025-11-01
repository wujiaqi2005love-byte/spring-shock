"""
路面激励模型模块
用于汽车悬架系统的动力学分析

支持的路面类型：
1. 正弦波路面（周期性路面）
2. 随机路面（ISO标准）
3. 单次冲击路面（路面突起）
4. 梯形路面（路沿石）
5. 自定义路面（用户数据）
"""

import numpy as np
from scipy import signal
from scipy.interpolate import interp1d


class RoadExcitation:
    """路面激励基类"""

    def __init__(self, vehicle_speed=20.0):
        """
        参数:
            vehicle_speed: 车速 (m/s)
        """
        self.vehicle_speed = vehicle_speed

    def get_displacement(self, t):
        """
        获取时刻t的路面位移

        参数:
            t: 时间 (s) 或时间数组

        返回:
            路面位移 (m)
        """
        raise NotImplementedError

    def get_velocity(self, t):
        """
        获取时刻t的路面速度

        参数:
            t: 时间 (s)

        返回:
            路面速度 (m/s)
        """
        # 默认使用数值微分
        dt = 1e-4
        return (self.get_displacement(t + dt) - self.get_displacement(t)) / dt


class SineRoadExcitation(RoadExcitation):
    """正弦波路面激励"""

    def __init__(self, amplitude=0.05, wavelength=5.0, vehicle_speed=20.0):
        """
        参数:
            amplitude: 路面幅值 (m)
            wavelength: 波长 (m)
            vehicle_speed: 车速 (m/s)
        """
        super().__init__(vehicle_speed)
        self.amplitude = amplitude
        self.wavelength = wavelength
        self.frequency = vehicle_speed / wavelength  # Hz
        self.omega = 2 * np.pi * self.frequency  # rad/s

    def get_displacement(self, t):
        """正弦波位移: z = A*sin(ω*t)"""
        return self.amplitude * np.sin(self.omega * t)

    def get_velocity(self, t):
        """正弦波速度: v = A*ω*cos(ω*t)"""
        return self.amplitude * self.omega * np.cos(self.omega * t)


class RandomRoadExcitation(RoadExcitation):
    """随机路面激励（基于ISO标准）"""

    # ISO路面等级分类（功率谱密度，单位：10^-6 m^3）
    ROAD_CLASSES = {
        'A': 16,  # 优良路面
        'B': 64,  # 良好路面
        'C': 256,  # 一般路面
        'D': 1024,  # 较差路面
        'E': 4096  # 很差路面
    }

    def __init__(self, road_class='C', vehicle_speed=20.0, duration=10.0,
                 n_samples=1000, random_seed=None):
        """
        参数:
            road_class: 路面等级 'A', 'B', 'C', 'D', 'E'
            vehicle_speed: 车速 (m/s)
            duration: 持续时间 (s)
            n_samples: 采样点数
            random_seed: 随机种子（用于可重复性）
        """
        super().__init__(vehicle_speed)
        self.road_class = road_class
        self.duration = duration
        self.n_samples = n_samples

        if random_seed is not None:
            np.random.seed(random_seed)

        # 生成路面位移时间历程
        self._generate_road_profile()

    def _generate_road_profile(self):
        """使用谐波叠加法生成随机路面"""
        # 参考空间频率范围
        n0 = 0.1  # 参考空间频率 (cycle/m)
        n_min = 0.01  # 最小空间频率
        n_max = 10.0  # 最大空间频率

        # 路面功率谱密度
        Gq = self.ROAD_CLASSES[self.road_class] * 1e-6  # m^3

        # 频率离散
        n_freqs = 1000
        n_array = np.logspace(np.log10(n_min), np.log10(n_max), n_freqs)

        # 功率谱密度函数：Gq(n) = Gq(n0) * (n/n0)^(-w)
        w = 2.0  # 频率指数
        Gq_n = Gq * (n_array / n0) ** (-w)

        # 时间数组
        self.time_array = np.linspace(0, self.duration, self.n_samples)

        # 路面位移（谐波叠加）
        displacement = np.zeros(self.n_samples)

        dn = (np.log10(n_max) - np.log10(n_min)) / n_freqs
        for i, n in enumerate(n_array):
            # 角频率
            omega = 2 * np.pi * n * self.vehicle_speed
            # 随机相位
            phi = np.random.uniform(0, 2 * np.pi)
            # 幅值
            amplitude = np.sqrt(2 * Gq_n[i] * dn * n)
            # 叠加
            displacement += amplitude * np.sin(omega * self.time_array + phi)

        self.displacement_array = displacement

        # 创建插值函数
        self.interp_func = interp1d(self.time_array, self.displacement_array,
                                    kind='cubic', fill_value='extrapolate')

    def get_displacement(self, t):
        """获取插值后的路面位移"""
        return self.interp_func(t)


class ImpulseRoadExcitation(RoadExcitation):
    """单次冲击路面激励（路面凸起、减速带等）"""

    def __init__(self, height=0.1, width=0.5, position=1.0, vehicle_speed=20.0):
        """
        参数:
            height: 凸起高度 (m)
            width: 凸起宽度 (m)
            position: 凸起位置（距起点距离，m）
            vehicle_speed: 车速 (m/s)
        """
        super().__init__(vehicle_speed)
        self.height = height
        self.width = width
        self.position = position
        self.t_start = position / vehicle_speed  # 开始时间
        self.t_duration = width / vehicle_speed  # 持续时间

    def get_displacement(self, t):
        """
        半正弦脉冲: z = H*sin(π*(t-t0)/T) for t0 < t < t0+T
        """
        t = np.atleast_1d(t)
        z = np.zeros_like(t)

        # 判断是否在凸起区间
        mask = (t >= self.t_start) & (t <= self.t_start + self.t_duration)

        # 计算相对时间
        t_rel = t[mask] - self.t_start

        # 半正弦函数
        z[mask] = self.height * np.sin(np.pi * t_rel / self.t_duration)

        return z if z.size > 1 else z[0]


class StepRoadExcitation(RoadExcitation):
    """梯形路面激励（路沿石、台阶）"""

    def __init__(self, height=0.15, ramp_length=0.2, position=1.0, vehicle_speed=20.0):
        """
        参数:
            height: 台阶高度 (m)
            ramp_length: 斜坡长度 (m)（0表示垂直台阶）
            position: 台阶位置（距起点距离，m）
            vehicle_speed: 车速 (m/s)
        """
        super().__init__(vehicle_speed)
        self.height = height
        self.ramp_length = ramp_length
        self.position = position
        self.t_start = position / vehicle_speed
        self.t_ramp = ramp_length / vehicle_speed

    def get_displacement(self, t):
        """梯形函数"""
        t = np.atleast_1d(t)
        z = np.zeros_like(t)

        if self.t_ramp > 0:
            # 有斜坡
            mask1 = (t >= self.t_start) & (t < self.t_start + self.t_ramp)
            mask2 = t >= self.t_start + self.t_ramp

            # 斜坡段
            t_rel = t[mask1] - self.t_start
            z[mask1] = self.height * (t_rel / self.t_ramp)

            # 平台段
            z[mask2] = self.height
        else:
            # 垂直台阶
            mask = t >= self.t_start
            z[mask] = self.height

        return z if z.size > 1 else z[0]


class CustomRoadExcitation(RoadExcitation):
    """自定义路面激励（用户提供数据）"""

    def __init__(self, time_data, displacement_data, vehicle_speed=20.0):
        """
        参数:
            time_data: 时间数组 (s)
            displacement_data: 位移数组 (m)
            vehicle_speed: 车速 (m/s)
        """
        super().__init__(vehicle_speed)
        self.time_data = np.array(time_data)
        self.displacement_data = np.array(displacement_data)

        # 创建插值函数
        self.interp_func = interp1d(self.time_data, self.displacement_data,
                                    kind='cubic', fill_value='extrapolate')

    def get_displacement(self, t):
        """获取插值后的路面位移"""
        return self.interp_func(t)


class CompositeRoadExcitation(RoadExcitation):
    """组合路面激励（多种路面叠加）"""

    def __init__(self, excitations, vehicle_speed=20.0):
        """
        参数:
            excitations: 路面激励对象列表
            vehicle_speed: 车速 (m/s)
        """
        super().__init__(vehicle_speed)
        self.excitations = excitations

    def get_displacement(self, t):
        """叠加所有路面激励"""
        total = 0.0
        for excitation in self.excitations:
            total += excitation.get_displacement(t)
        return total

    def get_velocity(self, t):
        """叠加所有路面速度"""
        total = 0.0
        for excitation in self.excitations:
            total += excitation.get_velocity(t)
        return total


# 便捷函数
def create_road_excitation(road_type, **kwargs):
    """
    工厂函数：创建路面激励对象

    参数:
        road_type: 路面类型
            - 'sine': 正弦波路面
            - 'random': 随机路面
            - 'impulse': 单次冲击
            - 'step': 梯形路面
            - 'custom': 自定义路面
        **kwargs: 对应路面类型的参数

    返回:
        RoadExcitation对象
    """
    road_classes = {
        'sine': SineRoadExcitation,
        'random': RandomRoadExcitation,
        'impulse': ImpulseRoadExcitation,
        'step': StepRoadExcitation,
        'custom': CustomRoadExcitation
    }

    if road_type not in road_classes:
        raise ValueError(f"未知的路面类型: {road_type}")

    return road_classes[road_type](**kwargs)