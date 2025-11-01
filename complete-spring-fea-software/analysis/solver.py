import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.integrate import odeint


class FEMSolver:
    """有限元求解器，支持静态分析和带阻尼的动力学分析"""

    def __init__(self, mesh, material, boundary_conditions, damping_config=None):
        """
        参数:
            mesh: {'nodes': ndarray (n_nodes,3), 'elements': list/ndarray of index lists, 'type': 'triangle'/'tetrahedron'}
            material: 对象，必须实现 get_elastic_matrix() 返回 6x6 (弹性矩阵 D)
            boundary_conditions: 对象，需包含:
                - node_forces: ndarray (3*n_nodes,) 总力向量
                - apply_boundary_conditions(K_csr, f) -> reduced_K, reduced_f
                - expand_displacement(reduced_u) -> full_u (3*n_nodes,)
            damping_config: dict, 阻尼配置
                {
                    'type': 'rayleigh' | 'modal' | 'proportional',
                    'alpha': 瑞利阻尼质量系数,
                    'beta': 瑞利阻尼刚度系数,
                    'damping_ratio': 阻尼比 (用于模态阻尼),
                    'viscous_coeff': 粘性阻尼系数 (用于比例阻尼)
                }
        """
        self.mesh = mesh
        self.material = material
        self.bc = boundary_conditions
        self.damping_config = damping_config or {}

        self.nodes = np.asarray(mesh['nodes'], dtype=np.float64)
        self.elements = list(mesh['elements'])
        self.element_type = mesh['type']

        self.stiffness_matrix = None
        self.mass_matrix = None
        self.damping_matrix = None
        self.force_vector = None
        self.displacement = None
        self.velocity = None
        self.acceleration = None
        self.stresses = None

        # 获取弹性矩阵（假设返回 6x6 全矩阵）
        self.d_matrix = np.asarray(material.get_elastic_matrix(), dtype=np.float64)

        # 密度（从材料获取）
        self.density = getattr(material, 'rho', 7850.0)

    # ---------------- 组装刚度矩阵 ----------------
    def assemble_stiffness_matrix(self):
        """组装整体刚度矩阵和力向量"""
        n_nodes = len(self.nodes)
        n_dofs = 3 * n_nodes

        # 初始化刚度矩阵（稀疏矩阵）
        self.stiffness_matrix = lil_matrix((n_dofs, n_dofs), dtype=np.float64)

        # 初始化力向量
        if getattr(self.bc, 'node_forces', None) is None:
            self.force_vector = np.zeros(n_dofs, dtype=np.float64)
        else:
            self.force_vector = np.asarray(self.bc.node_forces, dtype=np.float64).copy()
            if self.force_vector.size != n_dofs:
                raise ValueError(f"bc.node_forces 大小应为 {n_dofs}，但得到 {self.force_vector.size}")

        # 遍历单元并装配
        for element_id, element_nodes in enumerate(self.elements):
            element_nodes = np.asarray(element_nodes, dtype=int)
            coords = self.nodes[element_nodes]

            if self.element_type == 'triangle':
                ke = self.calculate_triangle_stiffness(coords)
            else:  # tetrahedron
                ke = self.calculate_tetrahedron_stiffness(coords)

            dofs = np.array([[3 * i, 3 * i + 1, 3 * i + 2] for i in element_nodes]).flatten()
            # 组装
            for i_local in range(len(dofs)):
                for j_local in range(len(dofs)):
                    self.stiffness_matrix[dofs[i_local], dofs[j_local]] += ke[i_local, j_local]

    # ---------------- 组装质量矩阵 ----------------
    def assemble_mass_matrix(self):
        """组装整体一致质量矩阵"""
        n_nodes = len(self.nodes)
        n_dofs = 3 * n_nodes

        self.mass_matrix = lil_matrix((n_dofs, n_dofs), dtype=np.float64)

        for element_id, element_nodes in enumerate(self.elements):
            element_nodes = np.asarray(element_nodes, dtype=int)
            coords = self.nodes[element_nodes]

            if self.element_type == 'triangle':
                me = self.calculate_triangle_mass(coords)
            else:  # tetrahedron
                me = self.calculate_tetrahedron_mass(coords)

            dofs = np.array([[3 * i, 3 * i + 1, 3 * i + 2] for i in element_nodes]).flatten()
            # 组装
            for i_local in range(len(dofs)):
                for j_local in range(len(dofs)):
                    self.mass_matrix[dofs[i_local], dofs[j_local]] += me[i_local, j_local]

    # ---------------- 组装阻尼矩阵 ----------------
    def assemble_damping_matrix(self):
        """根据配置组装阻尼矩阵"""
        if not self.damping_config:
            # 无阻尼配置，创建零阻尼矩阵
            n_dofs = 3 * len(self.nodes)
            self.damping_matrix = lil_matrix((n_dofs, n_dofs), dtype=np.float64)
            return

        damping_type = self.damping_config.get('type', 'rayleigh')

        if damping_type == 'rayleigh':
            # 瑞利阻尼: C = α*M + β*K
            alpha = self.damping_config.get('alpha', 0.0)
            beta = self.damping_config.get('beta', 0.0)

            if self.mass_matrix is None:
                self.assemble_mass_matrix()
            if self.stiffness_matrix is None:
                self.assemble_stiffness_matrix()

            self.damping_matrix = alpha * self.mass_matrix + beta * self.stiffness_matrix

        elif damping_type == 'proportional':
            # 比例阻尼: C = c * I (简化模型)
            c = self.damping_config.get('viscous_coeff', 0.0)
            n_dofs = 3 * len(self.nodes)
            self.damping_matrix = lil_matrix((n_dofs, n_dofs), dtype=np.float64)
            for i in range(n_dofs):
                self.damping_matrix[i, i] = c

        elif damping_type == 'modal':
            # 模态阻尼（需要特征值分析，这里简化为瑞利阻尼）
            damping_ratio = self.damping_config.get('damping_ratio', 0.05)
            # 假设两个特征频率
            omega1 = self.damping_config.get('omega1', 10.0)  # rad/s
            omega2 = self.damping_config.get('omega2', 100.0)  # rad/s

            # 计算瑞利系数
            alpha = 2 * damping_ratio * omega1 * omega2 / (omega1 + omega2)
            beta = 2 * damping_ratio / (omega1 + omega2)

            if self.mass_matrix is None:
                self.assemble_mass_matrix()
            if self.stiffness_matrix is None:
                self.assemble_stiffness_matrix()

            self.damping_matrix = alpha * self.mass_matrix + beta * self.stiffness_matrix

    # ---------------- 三角形单元刚度矩阵 ----------------
    def calculate_triangle_stiffness(self, coords):
        """计算三角形单元刚度矩阵 (9x9)"""
        x1, y1 = coords[0, 0], coords[0, 1]
        x2, y2 = coords[1, 0], coords[1, 1]
        x3, y3 = coords[2, 0], coords[2, 1]

        area = 0.5 * abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))
        if area <= 1e-14:
            raise ValueError("三角形单元面积接近 0，检查网格质量或节点顺序。")

        b1 = y2 - y3
        b2 = y3 - y1
        b3 = y1 - y2
        c1 = x3 - x2
        c2 = x1 - x3
        c3 = x2 - x1

        B2d = (1.0 / (2.0 * area)) * np.array([
            [b1, 0, b2, 0, b3, 0],
            [0, c1, 0, c2, 0, c3],
            [c1, b1, c2, b2, c3, b3]
        ])

        D2d = self.d_matrix[:3, :3]
        ke2d = area * (B2d.T @ D2d @ B2d)

        idx2d = [0, 1, 3, 4, 6, 7]
        ke9 = np.zeros((9, 9), dtype=np.float64)
        ke9[np.ix_(idx2d, idx2d)] = ke2d

        return ke9

    # ---------------- 三角形单元质量矩阵 ----------------
    def calculate_triangle_mass(self, coords):
        """计算三角形单元一致质量矩阵 (9x9)"""
        x1, y1 = coords[0, 0], coords[0, 1]
        x2, y2 = coords[1, 0], coords[1, 1]
        x3, y3 = coords[2, 0], coords[2, 1]

        area = 0.5 * abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))
        if area <= 1e-14:
            raise ValueError("三角形单元面积接近 0")

        # 假设单位厚度
        thickness = 1.0
        mass = self.density * area * thickness

        # 一致质量矩阵（每个节点质量均分，对角线优势）
        me2d = (mass / 12.0) * np.array([
            [2, 0, 1, 0, 1, 0],
            [0, 2, 0, 1, 0, 1],
            [1, 0, 2, 0, 1, 0],
            [0, 1, 0, 2, 0, 1],
            [1, 0, 1, 0, 2, 0],
            [0, 1, 0, 1, 0, 2]
        ])

        idx2d = [0, 1, 3, 4, 6, 7]
        me9 = np.zeros((9, 9), dtype=np.float64)
        me9[np.ix_(idx2d, idx2d)] = me2d

        return me9

    # ---------------- 四面体单元刚度矩阵 ----------------
    def calculate_tetrahedron_stiffness(self, coords):
        """计算四面体单元刚度矩阵 (12x12)"""
        p1, p2, p3, p4 = coords
        v = self.calculate_tetrahedron_volume(p1, p2, p3, p4)
        if v <= 1e-18:
            raise ValueError("四面体体积接近 0")

        A = np.array([
            [1.0, p1[0], p1[1], p1[2]],
            [1.0, p2[0], p2[1], p2[2]],
            [1.0, p3[0], p3[1], p3[2]],
            [1.0, p4[0], p4[1], p4[2]]
        ], dtype=np.float64)

        invA = np.linalg.inv(A)
        grads = invA[1:4, :]

        B = np.zeros((6, 12), dtype=np.float64)
        for i in range(4):
            bi, ci, di = grads[0, i], grads[1, i], grads[2, i]
            B[0, 3 * i + 0] = bi
            B[1, 3 * i + 1] = ci
            B[2, 3 * i + 2] = di
            B[3, 3 * i + 0] = ci
            B[3, 3 * i + 1] = bi
            B[4, 3 * i + 1] = di
            B[4, 3 * i + 2] = ci
            B[5, 3 * i + 0] = di
            B[5, 3 * i + 2] = bi

        ke = v * (B.T @ self.d_matrix @ B)
        return ke

    # ---------------- 四面体单元质量矩阵 ----------------
    def calculate_tetrahedron_mass(self, coords):
        """计算四面体单元一致质量矩阵 (12x12)"""
        p1, p2, p3, p4 = coords
        v = self.calculate_tetrahedron_volume(p1, p2, p3, p4)
        if v <= 1e-18:
            raise ValueError("四面体体积接近 0")

        mass = self.density * v

        # 一致质量矩阵（简化：每个节点质量均分）
        # 更精确的实现需要形函数积分
        me = np.zeros((12, 12), dtype=np.float64)
        node_mass = mass / 20.0  # 一致质量矩阵系数

        for i in range(4):
            for j in range(4):
                coeff = 2.0 if i == j else 1.0
                for k in range(3):
                    me[3 * i + k, 3 * j + k] = coeff * node_mass

        return me

    def calculate_tetrahedron_volume(self, p1, p2, p3, p4):
        """计算四面体体积"""
        return abs(np.dot(np.cross(p2 - p1, p3 - p1), (p4 - p1))) / 6.0

    # ---------------- 静态求解 ----------------
    def solve(self):
        """静态求解：K*u = F"""
        if self.stiffness_matrix is None:
            self.assemble_stiffness_matrix()

        reduced_k, reduced_f = self.bc.apply_boundary_conditions(
            self.stiffness_matrix.tocsr(),
            self.force_vector
        )

        reduced_u = spsolve(reduced_k, reduced_f)
        self.displacement = self.bc.expand_displacement(reduced_u)
        self.calculate_stresses()
        von_mises = self.calculate_von_mises_stress()

        return {
            'displacement': self.displacement,
            'stresses': self.stresses,
            'von_mises': von_mises
        }

    # ---------------- 动力学求解 ----------------
    def solve_dynamic(self, time_span, initial_displacement=None, initial_velocity=None,
                      time_dependent_force=None, road_excitation=None,
                      base_nodes=None, n_time_points=100):
        """
        动力学求解: M*a + C*v + K*u = F(t)

        参数:
            time_span: (t0, tf) 时间跨度
            initial_displacement: 初始位移 (n_dofs,)
            initial_velocity: 初始速度 (n_dofs,)
            time_dependent_force: 函数 f(t) 返回力向量 (n_dofs,)
            road_excitation: RoadExcitation对象，路面激励模型
            base_nodes: 受路面激励影响的节点索引列表
            n_time_points: 时间输出点数量

        返回:
            dict 包含时间历程数据
        """
        if self.stiffness_matrix is None:
            self.assemble_stiffness_matrix()
        if self.mass_matrix is None:
            self.assemble_mass_matrix()
        if self.damping_matrix is None:
            self.assemble_damping_matrix()

        n_dofs = 3 * len(self.nodes)

        # 初始条件
        if initial_displacement is None:
            initial_displacement = np.zeros(n_dofs)
        if initial_velocity is None:
            initial_velocity = np.zeros(n_dofs)

        # 应用边界条件
        K_csr = self.stiffness_matrix.tocsr()
        M_csr = self.mass_matrix.tocsr()
        C_csr = self.damping_matrix.tocsr()

        mask = self.bc.get_stiffness_matrix_mask()
        K_reduced = K_csr[mask][:, mask]
        M_reduced = M_csr[mask][:, mask]
        C_reduced = C_csr[mask][:, mask]

        # 简化的初始条件（只取自由度）
        u0 = initial_displacement[mask]
        v0 = initial_velocity[mask]
        y0 = np.concatenate([u0, v0])

        # 时间步
        t_eval = np.linspace(time_span[0], time_span[1], n_time_points)

        # 处理路面激励
        if road_excitation is not None and base_nodes is not None:
            # 将路面激励转换为基础位移约束
            # 这里使用力激励的方式模拟路面输入
            def get_road_force(t):
                """根据路面激励计算等效力"""
                road_disp = road_excitation.get_displacement(t)
                road_vel = road_excitation.get_velocity(t)

                # 创建基础激励力向量
                F_road = np.zeros(n_dofs)

                # 对每个基础节点施加激励力
                # F = K*u_base + C*v_base (基础激励产生的等效力)
                for node_idx in base_nodes:
                    for dof in range(3):
                        global_dof = 3 * node_idx + dof
                        # 根据方向（通常是Z方向）施加激励
                        if dof == 2:  # Z方向
                            # 简化模型：用弹簧-阻尼器连接基础
                            k_tire = 200000.0  # 轮胎刚度 (N/m)
                            c_tire = 1000.0  # 轮胎阻尼 (N·s/m)

                            # 当前节点位移和速度
                            u_node = initial_displacement[global_dof] if isinstance(t, float) else 0
                            v_node = initial_velocity[global_dof] if isinstance(t, float) else 0

                            # 轮胎产生的力
                            F_road[global_dof] = (k_tire * (road_disp - u_node) +
                                                  c_tire * (road_vel - v_node))

                return F_road
        else:
            get_road_force = None

        # 定义状态空间方程
        def equations_of_motion(y, t):
            u = y[:len(u0)]
            v = y[len(u0):]

            # 计算外力
            if time_dependent_force is not None:
                F = time_dependent_force(t)
            else:
                F = self.force_vector.copy()

            # 添加路面激励力
            if get_road_force is not None:
                F_road_full = get_road_force(t)
                F += F_road_full

            F_reduced = F[mask]

            # M*a = F - C*v - K*u
            # a = M^-1 * (F - C*v - K*u)
            rhs = F_reduced - C_reduced.dot(v) - K_reduced.dot(u)
            a = spsolve(M_reduced, rhs)

            return np.concatenate([v, a])

        # 求解ODE
        solution = odeint(equations_of_motion, y0, t_eval)

        # 提取位移和速度
        n_free = len(u0)
        displacements = solution[:, :n_free]
        velocities = solution[:, n_free:]

        # 扩展到完整自由度
        full_displacements = []
        full_velocities = []
        for i in range(len(t_eval)):
            full_u = self.bc.expand_displacement(displacements[i])
            full_v = self.bc.expand_displacement(velocities[i])
            full_displacements.append(full_u)
            full_velocities.append(full_v)

        # 保存最后时刻的结果用于应力计算
        self.displacement = full_displacements[-1]
        self.velocity = full_velocities[-1]
        self.calculate_stresses()
        von_mises = self.calculate_von_mises_stress()

        # 准备返回结果
        results = {
            'time': t_eval,
            'displacement_history': np.array(full_displacements),
            'velocity_history': np.array(full_velocities),
            'displacement': self.displacement,
            'velocity': self.velocity,
            'stresses': self.stresses,
            'von_mises': von_mises
        }

        # 如果有路面激励，记录路面位移历程
        if road_excitation is not None:
            road_displacement_history = np.array([road_excitation.get_displacement(t)
                                                  for t in t_eval])
            results['road_displacement_history'] = road_displacement_history
            results['road_excitation'] = road_excitation

        return results

    # ---------------- 计算单元应力 ----------------
    def calculate_stresses(self):
        """计算每个单元的应力"""
        n_elements = len(self.elements)
        self.stresses = np.zeros((n_elements, 6), dtype=np.float64)

        for element_id, element_nodes in enumerate(self.elements):
            element_nodes = np.asarray(element_nodes, dtype=int)
            coords = self.nodes[element_nodes]
            u_element = np.concatenate([self.displacement[3 * i:3 * i + 3] for i in element_nodes])

            if self.element_type == 'triangle':
                x1, y1 = coords[0, 0], coords[0, 1]
                x2, y2 = coords[1, 0], coords[1, 1]
                x3, y3 = coords[2, 0], coords[2, 1]
                area = 0.5 * abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))
                if area <= 1e-14:
                    raise ValueError("三角形单元面积接近 0")
                b1 = y2 - y3
                b2 = y3 - y1
                b3 = y1 - y2
                c1 = x3 - x2
                c2 = x1 - x3
                c3 = x2 - x1
                B2d = (1.0 / (2.0 * area)) * np.array([
                    [b1, 0, b2, 0, b3, 0],
                    [0, c1, 0, c2, 0, c3],
                    [c1, b1, c2, b2, c3, b3]
                ])
                u2d = u_element[[0, 1, 3, 4, 6, 7]]
                stress2d = self.d_matrix[:3, :3] @ (B2d @ u2d)
                self.stresses[element_id, :] = np.array([stress2d[0], stress2d[1], 0.0, stress2d[2], 0.0, 0.0])
            else:
                p1, p2, p3, p4 = coords
                v = self.calculate_tetrahedron_volume(p1, p2, p3, p4)
                if v <= 1e-18:
                    raise ValueError("四面体体积接近 0")
                A = np.array([
                    [1.0, p1[0], p1[1], p1[2]],
                    [1.0, p2[0], p2[1], p2[2]],
                    [1.0, p3[0], p3[1], p3[2]],
                    [1.0, p4[0], p4[1], p4[2]]
                ], dtype=np.float64)
                invA = np.linalg.inv(A)
                grads = invA[1:4, :]
                B = np.zeros((6, 12), dtype=np.float64)
                for i in range(4):
                    bi, ci, di = grads[0, i], grads[1, i], grads[2, i]
                    B[0, 3 * i + 0] = bi
                    B[1, 3 * i + 1] = ci
                    B[2, 3 * i + 2] = di
                    B[3, 3 * i + 0] = ci
                    B[3, 3 * i + 1] = bi
                    B[4, 3 * i + 1] = di
                    B[4, 3 * i + 2] = ci
                    B[5, 3 * i + 0] = di
                    B[5, 3 * i + 2] = bi
                stress = self.d_matrix @ (B @ u_element)
                self.stresses[element_id, :] = stress

    # ---------------- Von Mises 应力 ----------------
    def calculate_von_mises_stress(self):
        """计算每个单元的Von Mises应力"""
        if self.stresses is None:
            return None

        von_mises = np.zeros(len(self.stresses), dtype=np.float64)
        for i, stress in enumerate(self.stresses):
            s11, s22, s33, s12, s23, s13 = stress
            von_mises[i] = np.sqrt(
                0.5 * ((s11 - s22) ** 2 + (s22 - s33) ** 2 + (s33 - s11) ** 2) +
                3.0 * (s12 ** 2 + s23 ** 2 + s13 ** 2)
            )
        return von_mises