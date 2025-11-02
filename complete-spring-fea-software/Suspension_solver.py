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

    def __init__(self, m1, m2, k1, k2, c, road_excitation=None, vehicle_params=None):
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

        # 可选的整车参数字典（来自 GUI）
        # 例如包含轴载、安装几何、最大行程等。若提供，则 solver 会在结果中包含整车相关字段。
        self.vehicle_params = vehicle_params or {}

        # 组装系统矩阵
        self._assemble_matrices()

        # 如果提供整车参数（vehicle_mass, wheelbase, track_width），构建耦合 7-DOF 模型
        self.multi_body = False
        vp = self.vehicle_params
        if vp and all(k in vp for k in ('vehicle_mass', 'wheelbase', 'track_width')):
            try:
                self._assemble_multi_body()
                self.multi_body = True
            except Exception:
                # 若多体构建失败，回退到简单两个自由度角模型
                self.multi_body = False

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

    # ---------------- 多体 7-DOF 模型组装 ----------------
    def _assemble_multi_body(self):
        """
        组装 7-DOF 模型质量、刚度、阻尼矩阵。
        坐标顺序（广义坐标 q）:
          [z_uw_FL, z_uw_FR, z_uw_RL, z_uw_RR, zs, theta, phi]
        其中 zs 为车身质心竖向位移，theta 为俯仰（绕 y 轴），phi 为横倾（绕 x 轴）。
        """
        vp = self.vehicle_params

        # corner order: FL, FR, RL, RR
        L = float(vp.get('wheelbase', 2.6))
        T = float(vp.get('track_width', 1.6))
        # wheel x positions relative to CG (front positive)
        x_front = L / 2.0
        x_rear = -L / 2.0
        y_right = T / 2.0
        y_left = -T / 2.0

        self.wheel_positions = np.array([
            [x_front, y_left],   # FL (front-left)
            [x_front, y_right],  # FR
            [x_rear,  y_left],   # RL
            [x_rear,  y_right]   # RR
        ], dtype=float)

        # unsprung mass per corner (from GUI unsprung_mass) or fallback to m1
        m_uw = float(vp.get('unsprung_mass', getattr(self, 'm1', 40.0)))
        # tire stiffness k1 and suspension stiffness k2 per corner
        k_tire = float(self.k1)
        k_s = float(self.k2)
        c_s = float(self.c)

        # total vehicle sprung mass (approx) = vehicle_mass - 4*m_uw
        vehicle_mass = float(vp.get('vehicle_mass', 1500.0))
        Ms = max(vehicle_mass - 4.0 * m_uw, 1.0)

        # approximate rotational inertias
        I_pitch = Ms * (L ** 2) / 12.0
        I_roll = Ms * (T ** 2) / 12.0

        # Mass matrix (7x7)
        M = np.zeros((7, 7), dtype=float)
        for i in range(4):
            M[i, i] = m_uw
        M[4, 4] = Ms
        M[5, 5] = I_pitch
        M[6, 6] = I_roll

        # Stiffness K and damping C
        K = np.zeros((7, 7), dtype=float)
        C = np.zeros((7, 7), dtype=float)

        # assemble contributions from each corner
        for i in range(4):
            xi, yi = self.wheel_positions[i]

            # unsprung diagonal: tire + suspension
            K[i, i] += k_tire + k_s
            C[i, i] += 0.0 + c_s

            # coupling between unsprung i and sprung DOFs
            # unsprung row to sprung zs/theta/phi (negative k_s)
            K[i, 4] += -k_s
            K[i, 5] += -k_s * xi
            K[i, 6] += -k_s * yi
            C[i, 4] += -c_s
            C[i, 5] += -c_s * xi
            C[i, 6] += -c_s * yi

            # sprung rows
            K[4, 4] += k_s
            K[4, 5] += k_s * xi
            K[4, 6] += k_s * yi
            C[4, 4] += c_s
            C[4, 5] += c_s * xi
            C[4, 6] += c_s * yi

            K[5, 4] += k_s * xi
            K[5, 5] += k_s * (xi ** 2)
            K[5, 6] += k_s * (xi * yi)
            C[5, 4] += c_s * xi
            C[5, 5] += c_s * (xi ** 2)
            C[5, 6] += c_s * (xi * yi)

            K[6, 4] += k_s * yi
            K[6, 5] += k_s * (xi * yi)
            K[6, 6] += k_s * (yi ** 2)
            C[6, 4] += c_s * yi
            C[6, 5] += c_s * (xi * yi)
            C[6, 6] += c_s * (yi ** 2)

            # coupling sprung to unsprung (symmetric)
            K[4, i] += -k_s
            K[5, i] += -k_s * xi
            K[6, i] += -k_s * yi
            C[4, i] += -c_s
            C[5, i] += -c_s * xi
            C[6, i] += -c_s * yi

        # save matrices
        self.M_multi = M
        self.K_multi = K
        self.C_multi = C
        self.k_tire = k_tire
        self.c_tire = 0.0
        self.m_uw = m_uw
        self.Ms = Ms

    # ---------------- 静态平衡计算（sag） ----------------
    def _compute_static_equilibrium(self):
        """
        计算静态平衡位移 q_static，使 K * q_static = -W，其中 W 为重力载荷向量 (向下为正)
        返回 q_static（multi-body 返回长度7的向量；2-DOF 返回长度2向量）
        假设 vehicle_params 中的 axle loads（若存在）表示前轴/后轴载荷（单位 kg），按前/后轴各自分配到左右车轮。
        """
        g = 9.81

        # multi-body
        if self.multi_body:
            # 如果用户提供了 axle loads，我们将其解释为前/后轴承载的簧载质量（kg）
            # 假设: 'empty_axle_load' -> 前轴簧载质量 (kg)，'full_axle_load' -> 后轴簧载质量 (kg)
            # 这些将被分配到左右车轮各半，用于构造施加在轮端的静载。
            vp = self.vehicle_params
            front_axle = vp.get('empty_axle_load', None)
            rear_axle = vp.get('full_axle_load', None)

            if front_axle is not None and rear_axle is not None and front_axle > 0 and rear_axle > 0:
                # 将簧载分配到每轮（kg）
                front_per_wheel = float(front_axle) / 2.0
                rear_per_wheel = float(rear_axle) / 2.0
                wheel_sprung_mass = np.array([front_per_wheel, front_per_wheel, rear_per_wheel, rear_per_wheel], dtype=float)

                # 作用在 unsprung DOF 的静载 = unsprung self-weight + 分配到该轮的簧载
                W = np.zeros(7, dtype=float)
                for i in range(4):
                    W[i] = (self.m_uw + wheel_sprung_mass[i]) * g
                # 将簧载视为已分配到轮端，簧载在簧端（sprung DOF）上不再施加额外重力
                W[4] = 0.0
                W[5] = 0.0
                W[6] = 0.0

                try:
                    q_static = np.linalg.solve(self.K_multi, -W)
                    return q_static
                except Exception:
                    return np.zeros(7, dtype=float)

            # 否则采用总簧载 Ms 按整车作用的传统做法
            W = np.zeros(7, dtype=float)
            for i in range(4):
                W[i] = self.m_uw * g
            W[4] = self.Ms * g
            W[5] = 0.0
            W[6] = 0.0

            try:
                q_static = np.linalg.solve(self.K_multi, -W)
                return q_static
            except Exception:
                return np.zeros(7, dtype=float)

        # 2-DOF 简化模型
        else:
            # mass vector m1 (unsprung), m2 (sprung)
            W = np.array([self.m1 * g, self.m2 * g], dtype=float)
            try:
                q_static = np.linalg.solve(self.K, -W)
                return q_static
            except Exception:
                return np.zeros(2, dtype=float)

    # ---------------- 多体动力学求解 ----------------
    def _multi_body_equations(self, y, t):
        """状态向量 y = [q, qdot]，q 长度 7"""
        n = 7
        q = y[:n]
        qdot = y[n:]

        # external force vector due to road input (only unsprung rows)
        if self.road_excitation is not None:
            x0 = self.road_excitation.get_displacement(t)
            v0 = self.road_excitation.get_velocity(t)
            # assume same road input for all wheels for now
            road_disp = np.full(4, x0, dtype=float)
            road_vel = np.full(4, v0, dtype=float)
        else:
            road_disp = np.zeros(4, dtype=float)
            road_vel = np.zeros(4, dtype=float)

        # compute external force vector F (size 7)
        F = np.zeros(7, dtype=float)
        # unsprung eq external: k_tire * x0 + c_tire * v0
        for i in range(4):
            F[i] = self.k_tire * road_disp[i] + self.c_tire * road_vel[i]

        # compute qdd: M^-1 (F - C qdot - K q)
        try:
            rhs = F - (self.C_multi.dot(qdot) + self.K_multi.dot(q))
            qdd = np.linalg.solve(self.M_multi, rhs)
        except Exception:
            qdd = np.zeros(7, dtype=float)

        return np.concatenate([qdot, qdd])

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
            # 若未提供 initial_state，则使用静态平衡点作为初始位移，速度为零
            q_static = self._compute_static_equilibrium()
            if not self.multi_body:
                # q_static 长度 2 -> [x1, x2]，v 初始为 0
                initial_state = np.concatenate([q_static, np.zeros(2, dtype=float)])
            else:
                # multi-body: q_static 长度7，构造 14 维初始状态
                initial_state = np.concatenate([q_static, np.zeros(7, dtype=float)])

        # 时间数组
        t = np.linspace(time_span[0], time_span[1], n_points)

        if not self.multi_body:
            # 简化二自由度角模型（原有实现）
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

            # 组织结果（兼容原结构）
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
                    'c': self.c,
                    'vehicle_params': self.vehicle_params
                }
            }

            # 如果未启用 multi_body，但仍然传入 vehicle_params，则提供简单复制视图
            if self.vehicle_params:
                try:
                    n_pts = len(t)
                    full_x1 = np.tile(x1.reshape(n_pts, 1), (1, 4))
                    full_x2 = np.tile(x2.reshape(n_pts, 1), (1, 4))
                    full_a2 = np.tile(a2.reshape(n_pts, 1), (1, 4))
                    results['full_vehicle'] = {
                        'per_wheel': {'x1': full_x1, 'x2': full_x2, 'a2': full_a2},
                        'body_x2_mean': np.mean(full_x2, axis=1),
                        'vehicle_params': self.vehicle_params
                    }
                except Exception:
                    pass

            return results

        # ---------------- multi_body 解算 ----------------
        # 状态 dim = 14 (7 q + 7 qdot)
        y0 = np.asarray(initial_state, dtype=float)
        solution = odeint(self._multi_body_equations, y0, t)

        n = 7
        q = solution[:, :n]
        qdot = solution[:, n:]

        # q order: [z_uw_FL, z_uw_FR, z_uw_RL, z_uw_RR, zs, theta, phi]
        unsprung_z = q[:, 0:4]
        zs = q[:, 4]
        theta = q[:, 5]
        phi = q[:, 6]

        # velocities
        unsprung_v = qdot[:, 0:4]
        zs_v = qdot[:, 4]
        theta_v = qdot[:, 5]
        phi_v = qdot[:, 6]

        # accelerations (numerical)
        unsprung_a = np.gradient(unsprung_v, t, axis=0)
        zs_a = np.gradient(zs_v, t)

        # road
        if self.road_excitation is not None:
            road = np.array([self.road_excitation.get_displacement(ti) for ti in t])
        else:
            road = np.zeros(len(t))

        # compute suspension travel per wheel: z_sprung_point - z_uw
        pts = self.wheel_positions
        sprung_point = np.zeros_like(unsprung_z)
        for i in range(4):
            xi, yi = pts[i]
            sprung_point[:, i] = zs + theta * xi + phi * yi

        suspension_travel = sprung_point - unsprung_z
        tire_deflection = unsprung_z - road.reshape(-1, 1)

        # forces
        suspension_force = self.k2 * suspension_travel + self.c * ( ( (zs_v.reshape(-1,1) + theta_v.reshape(-1,1)*pts[:,0] + phi_v.reshape(-1,1)*pts[:,1]) ) - unsprung_v )
        # Note: above v mapping uses pts arrays shape; compute approximate sprung point velocities properly
        # For simplicity compute sprung point velocity per wheel
        sprung_v_pts = np.zeros_like(unsprung_v)
        for i in range(4):
            xi, yi = pts[i]
            sprung_v_pts[:, i] = zs_v + theta_v * xi + phi_v * yi
        suspension_force = self.k2 * suspension_travel + self.c * (sprung_v_pts - unsprung_v)

        tire_force = self.k1 * tire_deflection

        results = {
            'time': t,
            'x0': road,
            'unsprung_z': unsprung_z,
            'unsprung_v': unsprung_v,
            'unsprung_a': unsprung_a,
            'body_zs': zs,
            'body_zs_v': zs_v,
            'body_zs_a': zs_a,
            'body_theta': theta,
            'body_phi': phi,
            'suspension_travel': suspension_travel,
            'tire_deflection': tire_deflection,
            'suspension_force': suspension_force,
            'tire_force': tire_force,
            'natural_frequencies': self.natural_frequencies,
            'damping_ratios': self.damping_ratios,
            'parameters': {
                'vehicle_mass': self.vehicle_params.get('vehicle_mass'),
                'wheelbase': self.vehicle_params.get('wheelbase'),
                'track_width': self.vehicle_params.get('track_width'),
                'm_uw': self.m_uw,
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

        H_x = np.zeros(n_points, dtype=complex)
        H_a = np.zeros(n_points, dtype=complex)

        if not self.multi_body:
            # 原有 2-DOF 频响
            for i, w in enumerate(omega):
                A = self.K - w ** 2 * self.M + 1j * w * self.C
                F = np.array([self.k1, 0])
                try:
                    X = np.linalg.solve(A, F)
                    H_x[i] = X[1]
                    H_a[i] = -w ** 2 * X[1]
                except Exception:
                    H_x[i] = 0
                    H_a[i] = 0
        else:
            # multi-body: input is road displacement at wheels -> equivalent force vector F = [k_tire]*4
            for i, w in enumerate(omega):
                A = self.K_multi - w ** 2 * self.M_multi + 1j * w * self.C_multi
                F = np.zeros(7, dtype=complex)
                F[0:4] = self.k_tire
                try:
                    X = np.linalg.solve(A, F)
                    # body heave index 4
                    H_x[i] = X[4]
                    H_a[i] = -w ** 2 * X[4]
                except Exception:
                    H_x[i] = 0
                    H_a[i] = 0

        return frequencies, np.abs(H_x), np.abs(H_a)

    def calculate_comfort_index(self, results):
        """
        计算舒适性指标

        参数:
            results: solve_dynamic() 返回的结果

        返回:
            comfort_metrics: dict 舒适性指标
        """
        # 兼容多体/单角结果
        if 'body_zs_a' in results:
            a_body = results['body_zs_a']
            rms_acceleration = np.sqrt(np.mean(a_body ** 2))
            max_acceleration = np.max(np.abs(a_body))
            max_suspension_travel = np.max(np.abs(results['suspension_travel']))

            # tire_force shape may be (n_points,4)
            tf = results.get('tire_force')
            if tf is None:
                tire_dynamic_load = 0.0
            else:
                try:
                    tire_dynamic_load = np.max(np.abs(tf))
                except Exception:
                    tire_dynamic_load = float(np.max(np.abs(tf)))

            # approximate static load per wheel
            if hasattr(self, 'Ms'):
                tire_static_load = (self.Ms * 9.81) / 4.0
            else:
                tire_static_load = (self.m1 * 9.81)
            dynamic_load_coefficient = tire_dynamic_load / max(1e-6, tire_static_load)

        else:
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