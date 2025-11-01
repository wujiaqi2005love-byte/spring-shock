import os
import numpy as np
from PyQt5.QtWidgets import (QMainWindow, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QLineEdit, QFileDialog, QComboBox,
                             QDoubleSpinBox, QProgressBar, QMessageBox, QGroupBox,
                             QCheckBox, QSpinBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

# 导入各个模块
from material.material import Material
from meshing.mesher import Mesher
from analysis.boundary import BoundaryConditions
from analysis.solver import FEMSolver
from Road_excitation import RoadExcitation  # 导入路面激励类
from visualization.plotter import ResultsPlotter


class AnalysisThread(QThread):
    """后台分析线程"""
    progress_updated = pyqtSignal(int)
    analysis_finished = pyqtSignal(dict)

    def __init__(self, mesh, material, boundary_conditions, damping_config=None,
                 analysis_type='static', time_span=None, road_excitation=None,
                 excitation_nodes=None):
        super().__init__()
        self.mesh = mesh
        self.material = material
        self.boundary_conditions = boundary_conditions
        self.damping_config = damping_config
        self.analysis_type = analysis_type
        self.time_span = time_span or (0, 1.0)
        self.road_excitation = road_excitation
        self.excitation_nodes = excitation_nodes

    def run(self):
        try:
            solver = FEMSolver(self.mesh, self.material, self.boundary_conditions,
                               self.damping_config)
            self.progress_updated.emit(30)

            if self.analysis_type == 'static':
                # 静态分析
                solver.assemble_stiffness_matrix()
                self.progress_updated.emit(60)
                results = solver.solve()
            elif self.analysis_type == 'dynamic_road':
                # 带路面激励的动力学分析
                solver.assemble_stiffness_matrix()
                self.progress_updated.emit(40)
                solver.assemble_mass_matrix()
                self.progress_updated.emit(50)
                solver.assemble_damping_matrix()
                self.progress_updated.emit(60)
                results = solver.solve_dynamic(
                    self.time_span,
                    road_excitation=self.road_excitation,
                    base_nodes=self.excitation_nodes
                )
            else:
                # 普通动力学分析
                solver.assemble_stiffness_matrix()
                self.progress_updated.emit(40)
                solver.assemble_mass_matrix()
                self.progress_updated.emit(50)
                solver.assemble_damping_matrix()
                self.progress_updated.emit(60)
                results = solver.solve_dynamic(self.time_span)

            self.progress_updated.emit(100)
            self.analysis_finished.emit(results)
        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n\n{traceback.format_exc()}"
            self.analysis_finished.emit({"error": error_msg})


class MainWindow(QMainWindow):
    """主窗口类"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("汽车减震系统有限元分析软件（支持阻尼和路面激励）")
        self.setGeometry(100, 100, 1300, 900)

        # 数据存储
        self.step_file = None
        self.mesh = None
        self.material = None
        self.boundary_conditions = None
        self.results = None
        self.damping_config = None
        self.road_excitation = None

        # 创建主部件和布局
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        # 创建标签页控件
        self.tabs = QTabWidget()
        self.main_layout.addWidget(self.tabs)

        # 创建各个标签页
        self.create_import_tab()
        self.create_material_tab()
        self.create_meshing_tab()
        self.create_boundary_tab()
        self.create_damping_tab()
        self.create_road_excitation_tab()  # 新增：路面激励标签页
        self.create_analysis_tab()
        self.create_results_tab()

        # 创建导航按钮
        self.nav_layout = QHBoxLayout()
        self.prev_btn = QPushButton("上一步")
        self.next_btn = QPushButton("下一步")
        self.nav_layout.addWidget(self.prev_btn)
        self.nav_layout.addWidget(self.next_btn)
        self.main_layout.addLayout(self.nav_layout)

        # 连接信号和槽
        self.prev_btn.clicked.connect(self.prev_tab)
        self.next_btn.clicked.connect(self.next_tab)
        self.import_btn.clicked.connect(self.import_stl)
        self.generate_mesh_btn.clicked.connect(self.generate_mesh)
        self.set_boundary_btn.clicked.connect(self.set_boundary_conditions)
        self.run_analysis_btn.clicked.connect(self.run_analysis)
        self.plot_mesh_btn.clicked.connect(self.plot_mesh)
        self.plot_displacement_btn.clicked.connect(self.plot_displacement)
        self.plot_stress_btn.clicked.connect(self.plot_stress)
        self.plot_stress_disp_btn.clicked.connect(self.plot_stress_displacement)
        self.plot_time_history_btn.clicked.connect(self.plot_time_history)

        # 阻尼相关信号
        self.enable_damping_check.stateChanged.connect(self.toggle_damping_options)
        self.damping_type_combo.currentTextChanged.connect(self.update_damping_params)

        # 边界条件相关信号
        self.enable_boundary_check.stateChanged.connect(self.toggle_boundary_options)

        # 路面激励相关信号
        self.enable_road_check.stateChanged.connect(self.toggle_road_options)
        self.road_type_combo.currentTextChanged.connect(self.update_road_params)
        self.analysis_type_combo.currentTextChanged.connect(self.update_analysis_requirements)

        # 初始化按钮状态
        self.update_nav_buttons()

        # 初始化边界条件选项状态(默认启用)
        self.toggle_boundary_options(Qt.Checked)

    def create_import_tab(self):
        """创建模型导入标签页"""
        self.import_tab = QWidget()
        layout = QVBoxLayout(self.import_tab)

        self.file_path_edit = QLineEdit()
        self.file_path_edit.setReadOnly(True)
        self.import_btn = QPushButton("导入STEP文件")

        file_layout = QHBoxLayout()
        file_layout.addWidget(self.file_path_edit)
        file_layout.addWidget(self.import_btn)

        self.model_info = QLabel("未导入模型")
        self.model_info.setAlignment(Qt.AlignCenter)
        self.model_info.setStyleSheet("font-size: 14px; margin-top: 20px;")

        layout.addLayout(file_layout)
        layout.addWidget(self.model_info)
        layout.addStretch()

        self.tabs.addTab(self.import_tab, "1. STEP模型导入")

    def create_material_tab(self):
        """创建材料属性标签页"""
        self.material_tab = QWidget()
        layout = QVBoxLayout(self.material_tab)

        group = QGroupBox("材料属性")
        group_layout = QVBoxLayout(group)

        # 弹性模量
        em_layout = QHBoxLayout()
        em_layout.addWidget(QLabel("弹性模量 (Pa):"))
        self.elastic_modulus = QDoubleSpinBox()
        self.elastic_modulus.setRange(1e9, 1e12)
        self.elastic_modulus.setValue(2e11)
        self.elastic_modulus.setSuffix(" Pa")
        self.elastic_modulus.setDecimals(2)
        em_layout.addWidget(self.elastic_modulus)
        group_layout.addLayout(em_layout)

        # 泊松比
        pr_layout = QHBoxLayout()
        pr_layout.addWidget(QLabel("泊松比:"))
        self.poisson_ratio = QDoubleSpinBox()
        self.poisson_ratio.setRange(0.0, 0.5)
        self.poisson_ratio.setValue(0.3)
        pr_layout.addWidget(self.poisson_ratio)
        group_layout.addLayout(pr_layout)

        # 密度
        den_layout = QHBoxLayout()
        den_layout.addWidget(QLabel("密度 (kg/m³):"))
        self.density = QDoubleSpinBox()
        self.density.setRange(1000, 10000)
        self.density.setValue(7850)
        den_layout.addWidget(self.density)
        group_layout.addLayout(den_layout)

        # 屈服强度
        ys_layout = QHBoxLayout()
        ys_layout.addWidget(QLabel("屈服强度 (Pa):"))
        self.yield_strength = QDoubleSpinBox()
        self.yield_strength.setRange(1e6, 1e9)
        self.yield_strength.setValue(250e6)
        self.yield_strength.setSuffix(" Pa")
        group_layout.addLayout(ys_layout)

        group.setLayout(group_layout)
        layout.addWidget(group)
        layout.addStretch()

        self.tabs.addTab(self.material_tab, "2. 材料属性")

    def create_meshing_tab(self):
        """创建网格划分标签页"""
        self.meshing_tab = QWidget()
        layout = QVBoxLayout(self.meshing_tab)

        mesh_param_layout = QVBoxLayout()

        # 网格类型
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("网格类型:"))
        self.mesh_type = QComboBox()
        self.mesh_type.addItems(["triangle", "tetrahedron"])
        self.mesh_type.setCurrentText("tetrahedron")
        type_layout.addWidget(self.mesh_type)
        mesh_param_layout.addLayout(type_layout)

        # 网格大小
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("网格大小 (m):"))
        self.mesh_size = QDoubleSpinBox()
        self.mesh_size.setRange(0.001, 1.0)
        self.mesh_size.setValue(0.05)
        self.mesh_size.setDecimals(4)
        size_layout.addWidget(self.mesh_size)
        mesh_param_layout.addLayout(size_layout)

        # 生成网格按钮
        self.generate_mesh_btn = QPushButton("生成网格")
        mesh_param_layout.addWidget(self.generate_mesh_btn)

        # 网格信息
        self.mesh_info = QLabel("未生成网格")
        mesh_param_layout.addWidget(self.mesh_info)

        layout.addLayout(mesh_param_layout)
        layout.addStretch()

        self.tabs.addTab(self.meshing_tab, "3. 网格划分")

    def create_boundary_tab(self):
        """创建边界条件标签页"""
        self.boundary_tab = QWidget()
        layout = QVBoxLayout(self.boundary_tab)

        # 启用边界条件复选框
        self.enable_boundary_check = QCheckBox("启用边界条件")
        self.enable_boundary_check.setChecked(True)  # 默认启用
        self.enable_boundary_check.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(self.enable_boundary_check)

        # 边界条件配置组
        self.boundary_config_group = QWidget()
        boundary_config_layout = QVBoxLayout(self.boundary_config_group)

        # 载荷设置
        load_group = QGroupBox("载荷设置")
        load_layout = QVBoxLayout(load_group)

        # 载荷大小
        mag_layout = QHBoxLayout()
        mag_layout.addWidget(QLabel("载荷大小 (N):"))
        self.load_magnitude = QDoubleSpinBox()
        self.load_magnitude.setRange(1, 1e6)
        self.load_magnitude.setValue(1000)
        mag_layout.addWidget(self.load_magnitude)
        load_layout.addLayout(mag_layout)

        # 载荷方向
        dir_layout = QHBoxLayout()
        dir_layout.addWidget(QLabel("载荷方向:"))
        self.load_direction = QComboBox()
        self.load_direction.addItems(["X轴", "Y轴", "Z轴"])
        self.load_direction.setCurrentText("Z轴")
        dir_layout.addWidget(self.load_direction)
        load_layout.addLayout(dir_layout)

        load_group.setLayout(load_layout)
        boundary_config_layout.addWidget(load_group)

        # 设置按钮
        self.set_boundary_btn = QPushButton("应用边界条件")
        boundary_config_layout.addWidget(self.set_boundary_btn)

        # 将配置组添加到主布局
        self.boundary_config_group.setLayout(boundary_config_layout)
        layout.addWidget(self.boundary_config_group)

        layout.addStretch()

        self.tabs.addTab(self.boundary_tab, "4. 边界条件")

    def create_damping_tab(self):
        """创建阻尼配置标签页"""
        self.damping_tab = QWidget()
        layout = QVBoxLayout(self.damping_tab)

        # 启用阻尼复选框
        self.enable_damping_check = QCheckBox("启用阻尼分析")
        self.enable_damping_check.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(self.enable_damping_check)

        # 阻尼参数组
        self.damping_group = QGroupBox("阻尼参数")
        damping_layout = QVBoxLayout(self.damping_group)

        # 阻尼类型
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("阻尼类型:"))
        self.damping_type_combo = QComboBox()
        self.damping_type_combo.addItems(["瑞利阻尼", "比例阻尼", "模态阻尼"])
        type_layout.addWidget(self.damping_type_combo)
        damping_layout.addLayout(type_layout)

        # 瑞利阻尼参数
        self.rayleigh_widget = QWidget()
        rayleigh_layout = QVBoxLayout(self.rayleigh_widget)

        alpha_layout = QHBoxLayout()
        alpha_layout.addWidget(QLabel("α (质量系数):"))
        self.alpha_spinbox = QDoubleSpinBox()
        self.alpha_spinbox.setRange(0, 100)
        self.alpha_spinbox.setValue(0.1)
        self.alpha_spinbox.setDecimals(4)
        alpha_layout.addWidget(self.alpha_spinbox)
        rayleigh_layout.addLayout(alpha_layout)

        beta_layout = QHBoxLayout()
        beta_layout.addWidget(QLabel("β (刚度系数):"))
        self.beta_spinbox = QDoubleSpinBox()
        self.beta_spinbox.setRange(0, 1)
        self.beta_spinbox.setValue(0.001)
        self.beta_spinbox.setDecimals(6)
        beta_layout.addWidget(self.beta_spinbox)
        rayleigh_layout.addLayout(beta_layout)

        damping_layout.addWidget(self.rayleigh_widget)

        # 比例阻尼参数
        self.proportional_widget = QWidget()
        proportional_layout = QVBoxLayout(self.proportional_widget)

        visc_layout = QHBoxLayout()
        visc_layout.addWidget(QLabel("粘性系数 c:"))
        self.viscous_coeff_spinbox = QDoubleSpinBox()
        self.viscous_coeff_spinbox.setRange(0, 10000)
        self.viscous_coeff_spinbox.setValue(100)
        self.viscous_coeff_spinbox.setDecimals(2)
        visc_layout.addWidget(self.viscous_coeff_spinbox)
        proportional_layout.addLayout(visc_layout)

        damping_layout.addWidget(self.proportional_widget)
        self.proportional_widget.hide()

        # 模态阻尼参数
        self.modal_widget = QWidget()
        modal_layout = QVBoxLayout(self.modal_widget)

        ratio_layout = QHBoxLayout()
        ratio_layout.addWidget(QLabel("阻尼比 ζ:"))
        self.damping_ratio_spinbox = QDoubleSpinBox()
        self.damping_ratio_spinbox.setRange(0, 1)
        self.damping_ratio_spinbox.setValue(0.05)
        self.damping_ratio_spinbox.setDecimals(4)
        ratio_layout.addWidget(self.damping_ratio_spinbox)
        modal_layout.addLayout(ratio_layout)

        omega1_layout = QHBoxLayout()
        omega1_layout.addWidget(QLabel("第一频率 ω₁ (rad/s):"))
        self.omega1_spinbox = QDoubleSpinBox()
        self.omega1_spinbox.setRange(0.1, 1000)
        self.omega1_spinbox.setValue(10.0)
        self.omega1_spinbox.setDecimals(2)
        omega1_layout.addWidget(self.omega1_spinbox)
        modal_layout.addLayout(omega1_layout)

        omega2_layout = QHBoxLayout()
        omega2_layout.addWidget(QLabel("第二频率 ω₂ (rad/s):"))
        self.omega2_spinbox = QDoubleSpinBox()
        self.omega2_spinbox.setRange(0.1, 1000)
        self.omega2_spinbox.setValue(100.0)
        self.omega2_spinbox.setDecimals(2)
        omega2_layout.addWidget(self.omega2_spinbox)
        modal_layout.addLayout(omega2_layout)

        damping_layout.addWidget(self.modal_widget)
        self.modal_widget.hide()

        # 轮胎参数（用于路面激励分析）
        tire_group = QGroupBox("轮胎参数（用于路面激励分析）")
        tire_layout = QVBoxLayout(tire_group)

        tire_k_layout = QHBoxLayout()
        tire_k_layout.addWidget(QLabel("轮胎刚度 (N/m):"))
        self.tire_stiffness_spinbox = QDoubleSpinBox()
        self.tire_stiffness_spinbox.setRange(1e4, 1e7)
        self.tire_stiffness_spinbox.setValue(2e5)
        self.tire_stiffness_spinbox.setDecimals(0)
        tire_k_layout.addWidget(self.tire_stiffness_spinbox)
        tire_layout.addLayout(tire_k_layout)

        tire_c_layout = QHBoxLayout()
        tire_c_layout.addWidget(QLabel("轮胎阻尼 (N·s/m):"))
        self.tire_damping_spinbox = QDoubleSpinBox()
        self.tire_damping_spinbox.setRange(100, 10000)
        self.tire_damping_spinbox.setValue(1000)
        self.tire_damping_spinbox.setDecimals(0)
        tire_c_layout.addWidget(self.tire_damping_spinbox)
        tire_layout.addLayout(tire_c_layout)

        damping_layout.addWidget(tire_group)

        self.damping_group.setLayout(damping_layout)
        self.damping_group.setEnabled(False)
        layout.addWidget(self.damping_group)

        # 时间参数（用于动态分析）
        self.time_group = QGroupBox("动态分析时间参数")
        time_layout = QVBoxLayout(self.time_group)

        duration_layout = QHBoxLayout()
        duration_layout.addWidget(QLabel("分析时长 (s):"))
        self.time_duration_spinbox = QDoubleSpinBox()
        self.time_duration_spinbox.setRange(0.01, 100)
        self.time_duration_spinbox.setValue(5.0)
        self.time_duration_spinbox.setDecimals(3)
        duration_layout.addWidget(self.time_duration_spinbox)
        time_layout.addLayout(duration_layout)

        steps_layout = QHBoxLayout()
        steps_layout.addWidget(QLabel("时间步数:"))
        self.time_steps_spinbox = QSpinBox()
        self.time_steps_spinbox.setRange(50, 2000)
        self.time_steps_spinbox.setValue(200)
        steps_layout.addWidget(self.time_steps_spinbox)
        time_layout.addLayout(steps_layout)

        self.time_group.setLayout(time_layout)
        self.time_group.setEnabled(False)
        layout.addWidget(self.time_group)

        layout.addStretch()

        self.tabs.addTab(self.damping_tab, "5. 阻尼配置")

    def create_road_excitation_tab(self):
        """创建路面激励配置标签页"""
        self.road_tab = QWidget()
        layout = QVBoxLayout(self.road_tab)

        # 启用路面激励复选框
        self.enable_road_check = QCheckBox("启用路面激励（汽车减震系统分析）")
        self.enable_road_check.setStyleSheet("font-size: 14px; font-weight: bold; color: #0066cc;")
        layout.addWidget(self.enable_road_check)

        # 路面激励参数组
        self.road_group = QGroupBox("路面激励参数")
        road_layout = QVBoxLayout(self.road_group)

        # 激励类型
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("激励类型:"))
        self.road_type_combo = QComboBox()
        self.road_type_combo.addItems([
            "简谐激励（正弦波）",
            "减速带/凸起",
            "随机路面",
            "扫频激励",
            "ISO标准随机路面"
        ])
        type_layout.addWidget(self.road_type_combo)
        road_layout.addLayout(type_layout)

        # ============ 简谐激励参数 ============
        self.harmonic_widget = QWidget()
        harmonic_layout = QVBoxLayout(self.harmonic_widget)

        amp_layout = QHBoxLayout()
        amp_layout.addWidget(QLabel("振幅 A (m):"))
        self.harmonic_amplitude_spinbox = QDoubleSpinBox()
        self.harmonic_amplitude_spinbox.setRange(0.001, 0.5)
        self.harmonic_amplitude_spinbox.setValue(0.05)
        self.harmonic_amplitude_spinbox.setDecimals(4)
        amp_layout.addWidget(self.harmonic_amplitude_spinbox)
        harmonic_layout.addLayout(amp_layout)

        freq_layout = QHBoxLayout()
        freq_layout.addWidget(QLabel("频率 f (Hz):"))
        self.harmonic_frequency_spinbox = QDoubleSpinBox()
        self.harmonic_frequency_spinbox.setRange(0.1, 50)
        self.harmonic_frequency_spinbox.setValue(2.0)
        self.harmonic_frequency_spinbox.setDecimals(2)
        freq_layout.addWidget(self.harmonic_frequency_spinbox)
        harmonic_layout.addLayout(freq_layout)

        road_layout.addWidget(self.harmonic_widget)

        # ============ 减速带参数 ============
        self.bump_widget = QWidget()
        bump_layout = QVBoxLayout(self.bump_widget)

        height_layout = QHBoxLayout()
        height_layout.addWidget(QLabel("凸起高度 (m):"))
        self.bump_height_spinbox = QDoubleSpinBox()
        self.bump_height_spinbox.setRange(0.01, 0.5)
        self.bump_height_spinbox.setValue(0.1)
        self.bump_height_spinbox.setDecimals(3)
        height_layout.addWidget(self.bump_height_spinbox)
        bump_layout.addLayout(height_layout)

        length_layout = QHBoxLayout()
        length_layout.addWidget(QLabel("凸起长度 (m):"))
        self.bump_length_spinbox = QDoubleSpinBox()
        self.bump_length_spinbox.setRange(0.1, 5.0)
        self.bump_length_spinbox.setValue(0.5)
        self.bump_length_spinbox.setDecimals(2)
        length_layout.addWidget(self.bump_length_spinbox)
        bump_layout.addLayout(length_layout)

        velocity_layout = QHBoxLayout()
        velocity_layout.addWidget(QLabel("车速 (m/s):"))
        self.bump_velocity_spinbox = QDoubleSpinBox()
        self.bump_velocity_spinbox.setRange(1, 50)
        self.bump_velocity_spinbox.setValue(10.0)
        self.bump_velocity_spinbox.setDecimals(1)
        velocity_layout.addWidget(self.bump_velocity_spinbox)
        bump_layout.addLayout(velocity_layout)

        start_layout = QHBoxLayout()
        start_layout.addWidget(QLabel("开始时间 (s):"))
        self.bump_start_spinbox = QDoubleSpinBox()
        self.bump_start_spinbox.setRange(0, 10)
        self.bump_start_spinbox.setValue(0.5)
        self.bump_start_spinbox.setDecimals(2)
        start_layout.addWidget(self.bump_start_spinbox)
        bump_layout.addLayout(start_layout)

        road_layout.addWidget(self.bump_widget)
        self.bump_widget.hide()

        # ============ 随机路面参数 ============
        self.random_widget = QWidget()
        random_layout = QVBoxLayout(self.random_widget)

        std_layout = QHBoxLayout()
        std_layout.addWidget(QLabel("标准差 σ (m):"))
        self.random_std_spinbox = QDoubleSpinBox()
        self.random_std_spinbox.setRange(0.001, 0.1)
        self.random_std_spinbox.setValue(0.02)
        self.random_std_spinbox.setDecimals(4)
        std_layout.addWidget(self.random_std_spinbox)
        random_layout.addLayout(std_layout)

        seed_layout = QHBoxLayout()
        seed_layout.addWidget(QLabel("随机种子:"))
        self.random_seed_spinbox = QSpinBox()
        self.random_seed_spinbox.setRange(0, 10000)
        self.random_seed_spinbox.setValue(42)
        seed_layout.addWidget(self.random_seed_spinbox)
        random_layout.addLayout(seed_layout)

        road_layout.addWidget(self.random_widget)
        self.random_widget.hide()

        # ============ 扫频激励参数 ============
        self.swept_widget = QWidget()
        swept_layout = QVBoxLayout(self.swept_widget)

        swept_amp_layout = QHBoxLayout()
        swept_amp_layout.addWidget(QLabel("振幅 (m):"))
        self.swept_amplitude_spinbox = QDoubleSpinBox()
        self.swept_amplitude_spinbox.setRange(0.001, 0.2)
        self.swept_amplitude_spinbox.setValue(0.03)
        self.swept_amplitude_spinbox.setDecimals(4)
        swept_amp_layout.addWidget(self.swept_amplitude_spinbox)
        swept_layout.addLayout(swept_amp_layout)

        f_start_layout = QHBoxLayout()
        f_start_layout.addWidget(QLabel("起始频率 (Hz):"))
        self.swept_f_start_spinbox = QDoubleSpinBox()
        self.swept_f_start_spinbox.setRange(0.1, 100)
        self.swept_f_start_spinbox.setValue(1.0)
        self.swept_f_start_spinbox.setDecimals(2)
        f_start_layout.addWidget(self.swept_f_start_spinbox)
        swept_layout.addLayout(f_start_layout)

        f_end_layout = QHBoxLayout()
        f_end_layout.addWidget(QLabel("结束频率 (Hz):"))
        self.swept_f_end_spinbox = QDoubleSpinBox()
        self.swept_f_end_spinbox.setRange(0.1, 100)
        self.swept_f_end_spinbox.setValue(20.0)
        self.swept_f_end_spinbox.setDecimals(2)
        f_end_layout.addWidget(self.swept_f_end_spinbox)
        swept_layout.addLayout(f_end_layout)

        road_layout.addWidget(self.swept_widget)
        self.swept_widget.hide()

        # ============ ISO标准路面参数 ============
        self.iso_widget = QWidget()
        iso_layout = QVBoxLayout(self.iso_widget)

        class_layout = QHBoxLayout()
        class_layout.addWidget(QLabel("路面等级:"))
        self.iso_class_combo = QComboBox()
        self.iso_class_combo.addItems(["A (很好)", "B (好)", "C (一般)", "D (差)", "E (很差)"])
        self.iso_class_combo.setCurrentText("C (一般)")
        class_layout.addWidget(self.iso_class_combo)
        iso_layout.addLayout(class_layout)

        iso_vel_layout = QHBoxLayout()
        iso_vel_layout.addWidget(QLabel("车速 (m/s):"))
        self.iso_velocity_spinbox = QDoubleSpinBox()
        self.iso_velocity_spinbox.setRange(1, 50)
        self.iso_velocity_spinbox.setValue(20.0)
        self.iso_velocity_spinbox.setDecimals(1)
        iso_vel_layout.addWidget(self.iso_velocity_spinbox)
        iso_layout.addLayout(iso_vel_layout)

        iso_seed_layout = QHBoxLayout()
        iso_seed_layout.addWidget(QLabel("随机种子:"))
        self.iso_seed_spinbox = QSpinBox()
        self.iso_seed_spinbox.setRange(0, 10000)
        self.iso_seed_spinbox.setValue(42)
        iso_seed_layout.addWidget(self.iso_seed_spinbox)
        iso_layout.addLayout(iso_seed_layout)

        road_layout.addWidget(self.iso_widget)
        self.iso_widget.hide()

        self.road_group.setLayout(road_layout)
        self.road_group.setEnabled(False)
        layout.addWidget(self.road_group)

        # 说明文本
        info_label = QLabel(
            "提示：路面激励将模拟轮胎与路面的接触，适用于汽车减震系统分析。\n"
            "系统将自动在最高点的节点施加路面位移激励。"
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #666; font-size: 11px; margin-top: 10px;")
        layout.addWidget(info_label)

        layout.addStretch()

        self.tabs.addTab(self.road_tab, "6. 路面激励")

    def create_analysis_tab(self):
        """创建分析求解标签页"""
        self.analysis_tab = QWidget()
        layout = QVBoxLayout(self.analysis_tab)

        # 分析类型选择
        analysis_type_layout = QHBoxLayout()
        analysis_type_layout.addWidget(QLabel("分析类型:"))
        self.analysis_type_combo = QComboBox()
        self.analysis_type_combo.addItems(["静态分析", "动态分析", "路面激励分析"])
        analysis_type_layout.addWidget(self.analysis_type_combo)
        layout.addLayout(analysis_type_layout)

        # 分析要求说明
        self.analysis_requirements_label = QLabel()
        self.analysis_requirements_label.setWordWrap(True)
        self.analysis_requirements_label.setStyleSheet(
            "background-color: #f0f0f0; padding: 10px; border-radius: 5px; margin: 10px 0;"
        )
        self.update_analysis_requirements()
        layout.addWidget(self.analysis_requirements_label)

        self.run_analysis_btn = QPushButton("运行分析")
        self.run_analysis_btn.setStyleSheet("font-size: 16px; padding: 10px;")

        self.progress_bar = QProgressBar()
        self.analysis_status = QLabel("等待分析...")

        layout.addWidget(self.run_analysis_btn)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.analysis_status)
        layout.addStretch()

        self.tabs.addTab(self.analysis_tab, "7. 分析求解")

    def create_results_tab(self):
        """创建结果可视化标签页"""
        self.results_tab = QWidget()
        layout = QVBoxLayout(self.results_tab)

        # 结果按钮
        btn_layout1 = QHBoxLayout()
        btn_layout2 = QHBoxLayout()

        self.plot_mesh_btn = QPushButton("显示网格")
        self.plot_displacement_btn = QPushButton("显示位移分布")
        self.plot_stress_btn = QPushButton("显示应力云图")
        self.plot_stress_disp_btn = QPushButton("应力-位移关系")
        self.plot_time_history_btn = QPushButton("时间历程曲线")

        btn_layout1.addWidget(self.plot_mesh_btn)
        btn_layout1.addWidget(self.plot_displacement_btn)
        btn_layout1.addWidget(self.plot_stress_btn)
        btn_layout2.addWidget(self.plot_stress_disp_btn)
        btn_layout2.addWidget(self.plot_time_history_btn)

        # 结果信息
        self.results_info = QLabel("尚未进行分析，无结果可显示")
        self.results_info.setWordWrap(True)

        layout.addLayout(btn_layout1)
        layout.addLayout(btn_layout2)
        layout.addWidget(self.results_info)
        layout.addStretch()

        self.tabs.addTab(self.results_tab, "8. 结果可视化")

    def toggle_damping_options(self, state):
        """切换阻尼选项的启用状态"""
        enabled = (state == Qt.Checked)
        self.damping_group.setEnabled(enabled)
        self.time_group.setEnabled(enabled or self.enable_road_check.isChecked())

    def toggle_road_options(self, state):
        """切换路面激励选项的启用状态"""
        enabled = (state == Qt.Checked)
        self.road_group.setEnabled(enabled)
        self.time_group.setEnabled(enabled or self.enable_damping_check.isChecked())

        if enabled:
            # 如果启用路面激励，自动启用阻尼
            if not self.enable_damping_check.isChecked():
                QMessageBox.information(
                    self, "提示",
                    "路面激励分析需要阻尼模型，已自动启用阻尼分析。"
                )
                self.enable_damping_check.setChecked(True)

    def toggle_boundary_options(self, state):
        """切换边界条件配置的启用状态"""
        enabled = (state == Qt.Checked)
        self.boundary_config_group.setEnabled(enabled)

    def update_damping_params(self, damping_type):
        """根据阻尼类型更新参数界面"""
        self.rayleigh_widget.hide()
        self.proportional_widget.hide()
        self.modal_widget.hide()

        if damping_type == "瑞利阻尼":
            self.rayleigh_widget.show()
        elif damping_type == "比例阻尼":
            self.proportional_widget.show()
        elif damping_type == "模态阻尼":
            self.modal_widget.show()

    def update_road_params(self, road_type):
        """根据路面类型更新参数界面"""
        self.harmonic_widget.hide()
        self.bump_widget.hide()
        self.random_widget.hide()
        self.swept_widget.hide()
        self.iso_widget.hide()

        if road_type == "简谐激励（正弦波）":
            self.harmonic_widget.show()
        elif road_type == "减速带/凸起":
            self.bump_widget.show()
        elif road_type == "随机路面":
            self.random_widget.show()
        elif road_type == "扫频激励":
            self.swept_widget.show()
        elif road_type == "ISO标准随机路面":
            self.iso_widget.show()

    def update_analysis_requirements(self):
        """更新分析要求说明"""
        analysis_type = self.analysis_type_combo.currentText()

        if analysis_type == "静态分析":
            text = ("📊 静态分析：\n"
                    "• 不考虑时间效应\n"
                    "• 不需要启用阻尼\n"
                    "• 计算结构在静载荷下的平衡状态")
        elif analysis_type == "动态分析":
            text = ("⏱ 动态分析：\n"
                    "• 需要启用阻尼分析\n"
                    "• 考虑结构的惯性和阻尼效应\n"
                    "• 可以观察振动响应")
        else:  # 路面激励分析
            text = ("🚗 路面激励分析（汽车减震系统）：\n"
                    "• 需要启用阻尼分析\n"
                    "• 需要配置路面激励参数\n"
                    "• 模拟车辆通过不同路面的动态响应\n"
                    "• 可评估减震性能和乘坐舒适性")

        self.analysis_requirements_label.setText(text)

    def prev_tab(self):
        """切换到上一个标签页"""
        current_index = self.tabs.currentIndex()
        if current_index > 0:
            self.tabs.setCurrentIndex(current_index - 1)
            self.update_nav_buttons()

    def next_tab(self):
        """切换到下一个标签页"""
        current_index = self.tabs.currentIndex()
        if current_index < self.tabs.count() - 1:
            self.tabs.setCurrentIndex(current_index + 1)
            self.update_nav_buttons()

    def update_nav_buttons(self):
        """更新导航按钮状态"""
        current_index = self.tabs.currentIndex()
        self.prev_btn.setEnabled(current_index > 0)
        self.next_btn.setEnabled(current_index < self.tabs.count() - 1)

    def import_stl(self):
        """导入STL文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择STEP文件", "", "STEP Files (*.step *.stp)"
        )
        if file_path:
            self.step_file = file_path
            self.file_path_edit.setText(file_path)
            self.model_info.setText(f"STEP模型导入成功: {os.path.basename(file_path)}")
            QMessageBox.information(self, "成功", "STEP模型导入成功")

    def generate_mesh(self):
        """生成网格"""
        if not self.step_file:
            QMessageBox.warning(self, "警告", "请先导入STEP模型")
            return
        try:
            element_type = self.mesh_type.currentText()
            mesh_size = self.mesh_size.value()
            self.mesher = Mesher()
            self.mesh = self.mesher.generate_mesh(
                self.step_file,
                element_type=element_type,
                mesh_size=mesh_size
            )
            self.mesh_info.setText(
                f"网格生成成功: 节点数={len(self.mesh['nodes'])}, "
                f"单元数={len(self.mesh['elements'])}, 类型={element_type}"
            )
            QMessageBox.information(self, "成功", "网格生成成功")
        except Exception as e:
            self.mesh_info.setText(f"网格生成失败: {str(e)}")
            QMessageBox.critical(self, "错误", f"网格生成失败: {str(e)}")

    def set_boundary_conditions(self):
        """设置边界条件"""
        if not self.mesh:
            QMessageBox.warning(self, "警告", "请先生成网格")
            return

        try:
            # 创建材料对象
            self.material = Material(
                e=self.elastic_modulus.value(),
                nu=self.poisson_ratio.value(),
                rho=self.density.value(),
                sigma_y=self.yield_strength.value()
            )

            # 检查是否启用边界条件
            if self.enable_boundary_check.isChecked():
                # 获取载荷方向向量
                dir_text = self.load_direction.currentText()
                if dir_text == "X轴":
                    direction = [1, 0, 0]
                elif dir_text == "Y轴":
                    direction = [0, 1, 0]
                else:  # Z轴
                    direction = [0, 0, 1]

                # 创建边界条件
                self.boundary_conditions = BoundaryConditions(
                    self.mesh,
                    load_magnitude=self.load_magnitude.value(),
                    load_direction=direction
                )

                # 自动检测固定面和载荷面
                self.boundary_conditions.auto_detect_fixed_and_load_faces()

            # 配置阻尼
            if self.enable_damping_check.isChecked():
                damping_type_text = self.damping_type_combo.currentText()

                if damping_type_text == "瑞利阻尼":
                    self.damping_config = {
                        'type': 'rayleigh',
                        'alpha': self.alpha_spinbox.value(),
                        'beta': self.beta_spinbox.value(),
                        'tire_stiffness': self.tire_stiffness_spinbox.value(),
                        'tire_damping': self.tire_damping_spinbox.value(),
                        'n_steps': self.time_steps_spinbox.value()
                    }
                elif damping_type_text == "比例阻尼":
                    self.damping_config = {
                        'type': 'proportional',
                        'viscous_coeff': self.viscous_coeff_spinbox.value(),
                        'tire_stiffness': self.tire_stiffness_spinbox.value(),
                        'tire_damping': self.tire_damping_spinbox.value(),
                        'n_steps': self.time_steps_spinbox.value()
                    }
                elif damping_type_text == "模态阻尼":
                    self.damping_config = {
                        'type': 'modal',
                        'damping_ratio': self.damping_ratio_spinbox.value(),
                        'omega1': self.omega1_spinbox.value(),
                        'omega2': self.omega2_spinbox.value(),
                        'tire_stiffness': self.tire_stiffness_spinbox.value(),
                        'tire_damping': self.tire_damping_spinbox.value(),
                        'n_steps': self.time_steps_spinbox.value()
                    }
            else:
                self.damping_config = None

            # 配置路面激励
            if self.enable_road_check.isChecked():
                road_type_text = self.road_type_combo.currentText()

                if road_type_text == "简谐激励（正弦波）":
                    excitation_params = {
                        'amplitude': self.harmonic_amplitude_spinbox.value(),
                        'frequency': self.harmonic_frequency_spinbox.value(),
                        'phase': 0.0
                    }
                    self.road_excitation = RoadExcitation('harmonic', excitation_params)

                elif road_type_text == "减速带/凸起":
                    excitation_params = {
                        'height': self.bump_height_spinbox.value(),
                        'length': self.bump_length_spinbox.value(),
                        'velocity': self.bump_velocity_spinbox.value(),
                        'start_time': self.bump_start_spinbox.value()
                    }
                    self.road_excitation = RoadExcitation('bump', excitation_params)

                elif road_type_text == "随机路面":
                    excitation_params = {
                        'std': self.random_std_spinbox.value(),
                        'seed': self.random_seed_spinbox.value()
                    }
                    self.road_excitation = RoadExcitation('random', excitation_params)

                elif road_type_text == "扫频激励":
                    excitation_params = {
                        'amplitude': self.swept_amplitude_spinbox.value(),
                        'f_start': self.swept_f_start_spinbox.value(),
                        'f_end': self.swept_f_end_spinbox.value(),
                        'duration': self.time_duration_spinbox.value()
                    }
                    self.road_excitation = RoadExcitation('swept_sine', excitation_params)

                elif road_type_text == "ISO标准随机路面":
                    road_class = self.iso_class_combo.currentText()[0]  # 获取等级字母
                    excitation_params = {
                        'road_class': road_class,
                        'velocity': self.iso_velocity_spinbox.value(),
                        'seed': self.iso_seed_spinbox.value()
                    }
                    self.road_excitation = RoadExcitation('iso_random', excitation_params)
            else:
                self.road_excitation = None

            # 如果未启用边界条件,创建一个空的边界条件对象
            if not self.enable_boundary_check.isChecked():
                self.boundary_conditions = BoundaryConditions(
                    self.mesh,
                    load_magnitude=0,
                    load_direction=[0, 0, 1]
                )

            success_msg = "配置完成:\n"
            if self.enable_boundary_check.isChecked():
                success_msg += "✓ 边界条件已设置\n"
            else:
                success_msg += "○ 边界条件未启用\n"

            if self.enable_damping_check.isChecked():
                success_msg += "✓ 阻尼已配置\n"

            if self.enable_road_check.isChecked():
                success_msg += "✓ 路面激励已配置"

            QMessageBox.information(self, "成功", success_msg)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"设置失败: {str(e)}")

    def run_analysis(self):
        """运行有限元分析"""
        if not all([self.mesh, self.material]):
            QMessageBox.warning(self, "警告", "请先完成网格生成和材料设置")
            return

        if not self.boundary_conditions:
            QMessageBox.warning(self, "警告", "请先应用边界条件（可以不启用但需要点击应用按钮）")
            return

        # 检查分析类型与配置的一致性
        analysis_type_text = self.analysis_type_combo.currentText()

        if analysis_type_text == "动态分析" and not self.enable_damping_check.isChecked():
            QMessageBox.warning(self, "警告", "动态分析需要启用阻尼！")
            return

        if analysis_type_text == "路面激励分析":
            if not self.enable_road_check.isChecked():
                QMessageBox.warning(self, "警告", "路面激励分析需要启用路面激励配置！")
                return
            if not self.enable_damping_check.isChecked():
                QMessageBox.warning(self, "警告", "路面激励分析需要启用阻尼配置！")
                return

        try:
            # 禁用按钮
            self.run_analysis_btn.setEnabled(False)
            self.analysis_status.setText("正在进行有限元分析...")
            self.progress_bar.setValue(0)

            # 确定分析类型
            if analysis_type_text == "静态分析":
                analysis_type = 'static'
                time_span = None
            elif analysis_type_text == "路面激励分析":
                analysis_type = 'dynamic_road'
                time_span = (0, self.time_duration_spinbox.value())
            else:
                analysis_type = 'dynamic'
                time_span = (0, self.time_duration_spinbox.value())

            # 根据复选框状态决定是否使用边界条件
            boundary_conditions = self.boundary_conditions if self.enable_boundary_check.isChecked() else None

            # 创建并启动分析线程
            self.analysis_thread = AnalysisThread(
                self.mesh, self.material, boundary_conditions,
                self.damping_config, analysis_type, time_span,
                self.road_excitation, None  # excitation_nodes自动检测
            )
            self.analysis_thread.progress_updated.connect(self.update_progress)
            self.analysis_thread.analysis_finished.connect(self.on_analysis_finished)
            self.analysis_thread.start()
        except Exception as e:
            self.analysis_status.setText(f"分析失败: {str(e)}")
            self.run_analysis_btn.setEnabled(True)

    def update_progress(self, value):
        """更新进度条"""
        self.progress_bar.setValue(value)

    def on_analysis_finished(self, results):
        """分析完成处理"""
        self.run_analysis_btn.setEnabled(True)

        if "error" in results:
            self.analysis_status.setText(f"分析失败")
            QMessageBox.critical(self, "错误", f"分析失败:\n{results['error']}")
        else:
            self.results = results
            self.analysis_status.setText("分析完成成功")

            von_mises = results.get('von_mises', None)
            if von_mises is None or len(von_mises) == 0:
                self.results_info.setText("分析完成，但未计算出有效应力结果。")
            else:
                max_disp = np.max(np.linalg.norm(results['displacement'].reshape(-1, 3), axis=1))
                max_stress = np.max(von_mises)

                if 'time' in results:
                    # 动态分析结果
                    info_text = f"动态分析完成:\n"
                    info_text += f"• 最终时刻最大位移: {max_disp:.6e} m\n"
                    info_text += f"• 最终时刻最大应力: {max_stress:.2e} Pa\n"
                    info_text += f"• 时间步数: {len(results['time'])}\n"

                    if 'road_displacement_history' in results:
                        road_max = np.max(np.abs(results['road_displacement_history']))
                        info_text += f"• 路面激励最大幅值: {road_max:.4f} m\n"
                        info_text += f"• 激励节点: {results.get('excitation_nodes', [])}"

                    self.results_info.setText(info_text)
                else:
                    # 静态分析结果
                    self.results_info.setText(
                        f"静态分析完成:\n"
                        f"• 最大位移: {max_disp:.6e} m\n"
                        f"• 最大应力: {max_stress:.2e} Pa"
                    )
                QMessageBox.information(self, "成功", "有限元分析完成！")

            # 自动切换到结果标签页
            self.tabs.setCurrentIndex(7)

    def plot_mesh(self):
        """显示网格"""
        if not self.mesh:
            QMessageBox.warning(self, "警告", "请先生成网格")
            return

        try:
            plotter = ResultsPlotter(self.mesh, {})
            plotter.plot_mesh()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"无法显示网格: {str(e)}")

    def plot_displacement(self):
        """显示位移分布"""
        if not self.results:
            QMessageBox.warning(self, "警告", "请先完成分析")
            return

        try:
            plotter = ResultsPlotter(self.mesh, self.results)
            plotter.plot_displacement()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"无法显示位移分布: {str(e)}")

    def plot_stress(self):
        """显示应力云图"""
        if not self.results:
            QMessageBox.warning(self, "警告", "请先完成分析")
            return

        try:
            plotter = ResultsPlotter(self.mesh, self.results)
            plotter.plot_stress(self.results['von_mises'])
        except Exception as e:
            QMessageBox.critical(self, "错误", f"无法显示应力云图: {str(e)}")

    def plot_stress_displacement(self):
        """显示应力-位移关系"""
        if not self.results:
            QMessageBox.warning(self, "警告", "请先完成分析")
            return

        try:
            plotter = ResultsPlotter(self.mesh, self.results)
            plotter.plot_stress_vs_displacement()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"无法显示应力-位移关系: {str(e)}")

    def plot_time_history(self):
        """显示时间历程曲线"""
        if not self.results:
            QMessageBox.warning(self, "警告", "请先完成分析")
            return

        if 'time' not in self.results:
            QMessageBox.warning(self, "警告", "静态分析没有时间历程数据")
            return

        try:
            import matplotlib.pyplot as plt

            time = self.results['time']

            # 创建图形
            fig, axes = plt.subplots(3, 1, figsize=(10, 10))
            fig.suptitle('时间历程响应', fontsize=14, fontweight='bold')

            # 位移历程（取最大位移节点）
            disp_history = self.results['displacement_history']
            disp_norms = np.linalg.norm(disp_history.reshape(len(time), -1, 3), axis=2)
            max_node = np.argmax(np.max(disp_norms, axis=0))

            axes[0].plot(time, disp_norms[:, max_node], 'b-', linewidth=2)
            axes[0].set_ylabel('位移 (m)', fontsize=12)
            axes[0].set_title(f'节点{max_node}的位移响应', fontsize=11)
            axes[0].grid(True, alpha=0.3)

            # 速度历程
            if 'velocity_history' in self.results:
                vel_history = self.results['velocity_history']
                vel_norms = np.linalg.norm(vel_history.reshape(len(time), -1, 3), axis=2)
                axes[1].plot(time, vel_norms[:, max_node], 'g-', linewidth=2)
                axes[1].set_ylabel('速度 (m/s)', fontsize=12)
                axes[1].set_title(f'节点{max_node}的速度响应', fontsize=11)
                axes[1].grid(True, alpha=0.3)

            # 路面激励 or 加速度
            if 'road_displacement_history' in self.results:
                road_disp = self.results['road_displacement_history']
                axes[2].plot(time, road_disp, 'r-', linewidth=2, label='路面激励')
                axes[2].plot(time, disp_norms[:, max_node], 'b--', linewidth=1.5,
                             label='车身响应', alpha=0.7)
                axes[2].set_ylabel('位移 (m)', fontsize=12)
                axes[2].set_xlabel('时间 (s)', fontsize=12)
                axes[2].set_title('路面激励与车身响应对比', fontsize=11)
                axes[2].legend()
                axes[2].grid(True, alpha=0.3)
            elif 'acceleration_history' in self.results:
                acc_history = self.results['acceleration_history']
                acc_norms = np.linalg.norm(acc_history.reshape(len(time), -1, 3), axis=2)
                axes[2].plot(time, acc_norms[:, max_node], 'm-', linewidth=2)
                axes[2].set_ylabel('加速度 (m/s²)', fontsize=12)
                axes[2].set_xlabel('时间 (s)', fontsize=12)
                axes[2].set_title(f'节点{max_node}的加速度响应', fontsize=11)
                axes[2].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()

        except Exception as e:
            QMessageBox.critical(self, "错误", f"无法显示时间历程: {str(e)}")