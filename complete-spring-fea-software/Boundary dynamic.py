"""
动力学分析边界条件 - 适用于路面激励分析
与静态分析的边界条件区别：
1. 不需要固定约束（路面激励是基础激励）
2. 不需要静态载荷
3. 需要指定受激励的节点
"""

import numpy as np
from analysis.boundary import BoundaryConditions

class DynamicBoundaryConditions:
    """动力学分析边界条件 - 用于路面激励分析"""

    def __init__(self, mesh, excitation_nodes=None):
        """
        初始化动力学边界条件

        参数:
            mesh: 网格数据
            excitation_nodes: 受路面激励影响的节点索引列表（如底部节点）
        """
        self.mesh = mesh
        self.nodes = mesh['nodes']
        self.elements = mesh['elements']

        # 动力学分析不需要静态载荷
        self.node_forces = np.zeros(3 * len(self.nodes))

        # 受激励节点
        self.excitation_nodes = excitation_nodes if excitation_nodes is not None else []

        # 动力学分析通常不需要固定约束（自由振动或基础激励）
        self.fixed_dofs = []

    def auto_detect_excitation_nodes(self, direction='z'):
        """
        自动检测受路面激励的节点（通常是底部节点）

        参数:
            direction: 'x', 'y', 或 'z'，表示激励方向
        """
        direction_map = {'x': 0, 'y': 1, 'z': 2}
        axis = direction_map.get(direction.lower(), 2)

        # 找到该方向坐标最小的节点（底部节点）
        node_coords = self.nodes[:, axis]
        min_coord = np.min(node_coords)
        tolerance = 1e-6

        # 选择所有接近最小坐标的节点
        self.excitation_nodes = np.where(np.abs(node_coords - min_coord) < tolerance)[0]

        return self.excitation_nodes

    def set_excitation_nodes(self, node_indices):
        """
        手动设置受激励节点

        参数:
            node_indices: 节点索引数组或列表
        """
        self.excitation_nodes = np.array(node_indices, dtype=int)

    def get_stiffness_matrix_mask(self):
        """
        生成刚度矩阵的掩码
        动力学分析通常不需要固定约束，返回全True掩码
        """
        n_dofs = 3 * len(self.nodes)
        mask = np.ones(n_dofs, dtype=bool)

        # 如果有固定自由度（特殊情况）
        if len(self.fixed_dofs) > 0:
            mask[self.fixed_dofs] = False

        return mask

    def apply_boundary_conditions(self, stiffness_matrix, force_vector):
        """
        应用边界条件（动力学分析版本）

        参数:
            stiffness_matrix: 刚度矩阵
            force_vector: 力向量

        返回:
            处理后的刚度矩阵和力向量
        """
        mask = self.get_stiffness_matrix_mask()

        if len(self.fixed_dofs) > 0:
            # 如果有固定约束，应用掩码
            reduced_stiffness = stiffness_matrix[mask][:, mask]
            reduced_force = force_vector[mask]
        else:
            # 没有固定约束，直接返回（自由振动）
            reduced_stiffness = stiffness_matrix
            reduced_force = force_vector

        return reduced_stiffness, reduced_force

    def expand_displacement(self, reduced_displacement):
        """
        扩展位移向量

        参数:
            reduced_displacement: 简化的位移向量

        返回:
            完整的位移向量
        """
        if len(self.fixed_dofs) > 0:
            mask = self.get_stiffness_matrix_mask()
            full_displacement = np.zeros(3 * len(self.nodes))
            full_displacement[mask] = reduced_displacement
            return full_displacement
        else:
            # 没有约束，直接返回
            return reduced_displacement


class CombinedBoundaryConditions(BoundaryConditions):
    """
    组合边界条件 - 同时支持静态和动力学分析
    继承原有的静态边界条件，添加动力学功能
    """

    def __init__(self, mesh, load_magnitude=1000.0, load_direction=[0, 0, 1],
                 analysis_type='static', excitation_nodes=None):
        """
        初始化组合边界条件

        参数:
            mesh: 网格数据
            load_magnitude: 载荷大小（静态分析用）
            load_direction: 载荷方向（静态分析用）
            analysis_type: 'static' 或 'dynamic'
            excitation_nodes: 受激励节点（动力学分析用）
        """
        super().__init__(mesh, load_magnitude, load_direction)

        self.analysis_type = analysis_type
        self.excitation_nodes = excitation_nodes if excitation_nodes is not None else []

    def set_analysis_type(self, analysis_type):
        """设置分析类型"""
        self.analysis_type = analysis_type

    def auto_detect_excitation_nodes(self, direction='z'):
        """自动检测受激励节点"""
        direction_map = {'x': 0, 'y': 1, 'z': 2}
        axis = direction_map.get(direction.lower(), 2)

        node_coords = self.nodes[:, axis]
        min_coord = np.min(node_coords)
        tolerance = 1e-6

        self.excitation_nodes = np.where(np.abs(node_coords - min_coord) < tolerance)[0]
        return self.excitation_nodes

    def apply_for_dynamic_analysis(self):
        """
        为动力学分析配置边界条件
        清除静态载荷，设置为自由振动或基础激励
        """
        self.analysis_type = 'dynamic'

        # 动力学分析不需要静态载荷
        self.node_forces = np.zeros(3 * len(self.nodes))

        # 如果是路面激励，通常不需要固定约束
        # 但可以保留原有的约束设置以支持半主动悬架等特殊情况

    def apply_boundary_conditions(self, stiffness_matrix, force_vector):
        """
        根据分析类型应用不同的边界条件
        """
        if self.analysis_type == 'dynamic' and len(self.fixed_dofs) == 0:
            # 动力学分析且无固定约束 - 自由振动
            return stiffness_matrix, force_vector
        else:
            # 静态分析或有约束的动力学分析
            return super().apply_boundary_conditions(stiffness_matrix, force_vector)



from analysis.boundary import BoundaryConditions