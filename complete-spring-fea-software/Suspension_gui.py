"""
悬架系统分析GUI
用于输入悬架参数、选择路面激励并进行动力学分析
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QGroupBox, QLabel, QDoubleSpinBox, QPushButton,
                             QComboBox, QTabWidget, QTextEdit, QSplitter,
                             QApplication, QMessageBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from Suspension_solver import SuspensionSolver
from Road_excitation import (SineRoadExcitation, RandomRoadExcitation,
                             ImpulseRoadExcitation, StepRoadExcitation)


class AnalysisThread(QThread):
    """后台计算线程"""
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, solver, time_span, n_points):
        super().__init__()
        self.solver = solver
        self.time_span = time_span
        self.n_points = n_points

    def run(self):
        try:
            results = self.solver.solve_dynamic(self.time_span, n_points=self.n_points)
            comfort = self.solver.calculate_comfort_index(results)
            results['comfort_metrics'] = comfort
            self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))


class SuspensionAnalysisWindow(QMainWindow):
    """悬架系统分析主窗口"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("汽车悬架系统动力学分析")
        self.setGeometry(100, 100, 1400, 900)

        self.results = None
        self.solver = None

        self._init_ui()

    def _init_ui(self):
        """初始化界面"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)

        # 左侧参数输入区
        left_panel = self._create_input_panel()

        # 右侧结果显示区
        right_panel = self._create_results_panel()

        # 使用分割器
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)

        main_layout.addWidget(splitter)

    def _create_input_panel(self):
        """创建参数输入面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # 1. 质量参数组
        mass_group = QGroupBox("质量参数")
        mass_layout = QVBoxLayout()

        # m1 - 非簧载质量
        m1_layout = QHBoxLayout()
        m1_layout.addWidget(QLabel("m₁ 非簧载质量 (kg):"))
        self.m1_spin = QDoubleSpinBox()
        self.m1_spin.setRange(10, 200)
        self.m1_spin.setValue(40)
        self.m1_spin.setDecimals(1)
        m1_layout.addWidget(self.m1_spin)
        mass_layout.addLayout(m1_layout)

        # m2 - 簧载质量
        m2_layout = QHBoxLayout()
        m2_layout.addWidget(QLabel("m₂ 簧载质量 (kg):"))
        self.m2_spin = QDoubleSpinBox()
        self.m2_spin.setRange(100, 2000)
        self.m2_spin.setValue(320)
        self.m2_spin.setDecimals(1)
        m2_layout.addWidget(self.m2_spin)
        mass_layout.addLayout(m2_layout)

        mass_group.setLayout(mass_layout)
        layout.addWidget(mass_group)

        # 2. 刚度参数组
        stiffness_group = QGroupBox("刚度参数")
        stiffness_layout = QVBoxLayout()

        # k1 - 轮胎刚度
        k1_layout = QHBoxLayout()
        k1_layout.addWidget(QLabel("k₁ 轮胎刚度 (N/m):"))
        self.k1_spin = QDoubleSpinBox()
        self.k1_spin.setRange(10000, 500000)
        self.k1_spin.setValue(190000)
        self.k1_spin.setDecimals(0)
        k1_layout.addWidget(self.k1_spin)
        stiffness_layout.addLayout(k1_layout)

        # k2 - 悬架刚度
        k2_layout = QHBoxLayout()
        k2_layout.addWidget(QLabel("k₂ 悬架刚度 (N/m):"))
        self.k2_spin = QDoubleSpinBox()
        self.k2_spin.setRange(1000, 100000)
        self.k2_spin.setValue(16000)
        self.k2_spin.setDecimals(0)
        k2_layout.addWidget(self.k2_spin)
        stiffness_layout.addLayout(k2_layout)

        stiffness_group.setLayout(stiffness_layout)
        layout.addWidget(stiffness_group)

        # 3. 阻尼参数组
        damping_group = QGroupBox("阻尼参数（比例阻尼）")
        damping_layout = QVBoxLayout()

        # c - 悬架阻尼系数
        c_layout = QHBoxLayout()
        c_layout.addWidget(QLabel("c 阻尼系数 (N·s/m):"))
        self.c_spin = QDoubleSpinBox()
        self.c_spin.setRange(100, 10000)
        self.c_spin.setValue(1000)
        self.c_spin.setDecimals(0)
        c_layout.addWidget(self.c_spin)
        damping_layout.addLayout(c_layout)

        # 阻尼比显示（实时计算）
        self.damping_ratio_label = QLabel("阻尼比: --")
        damping_layout.addWidget(self.damping_ratio_label)

        # 连接信号以实时更新阻尼比
        self.c_spin.valueChanged.connect(self._update_damping_ratio)
        self.k2_spin.valueChanged.connect(self._update_damping_ratio)
        self.m2_spin.valueChanged.connect(self._update_damping_ratio)

        damping_group.setLayout(damping_layout)
        layout.addWidget(damping_group)

        # 4. 路面激励参数组
        road_group = QGroupBox("路面激励")
        road_layout = QVBoxLayout()

        # 路面类型选择
        road_type_layout = QHBoxLayout()
        road_type_layout.addWidget(QLabel("路面类型:"))
        self.road_type_combo = QComboBox()
        self.road_type_combo.addItems([
            "正弦波路面",
            "随机路面 (ISO)",
            "单次冲击",
            "梯形路面",
            "无激励"
        ])
        self.road_type_combo.currentTextChanged.connect(self._on_road_type_changed)
        road_type_layout.addWidget(self.road_type_combo)
        road_layout.addLayout(road_type_layout)

        # 路面参数（动态显示）
        self.road_params_widget = QWidget()
        self.road_params_layout = QVBoxLayout(self.road_params_widget)
        road_layout.addWidget(self.road_params_widget)

        # 车速
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("车速 (m/s):"))
        self.speed_spin = QDoubleSpinBox()
        self.speed_spin.setRange(1, 50)
        self.speed_spin.setValue(20)
        self.speed_spin.setDecimals(1)
        speed_layout.addWidget(self.speed_spin)
        road_layout.addLayout(speed_layout)

        road_group.setLayout(road_layout)
        layout.addWidget(road_group)

        # 初始化路面参数界面
        self._on_road_type_changed("正弦波路面")

        # 5. 仿真参数组
        sim_group = QGroupBox("仿真参数")
        sim_layout = QVBoxLayout()

        # 仿真时长
        duration_layout = QHBoxLayout()
        duration_layout.addWidget(QLabel("仿真时长 (s):"))
        self.duration_spin = QDoubleSpinBox()
        self.duration_spin.setRange(0.1, 100)
        self.duration_spin.setValue(5.0)
        self.duration_spin.setDecimals(1)
        duration_layout.addWidget(self.duration_spin)
        sim_layout.addLayout(duration_layout)

        # 采样点数
        points_layout = QHBoxLayout()
        points_layout.addWidget(QLabel("采样点数:"))
        self.points_spin = QDoubleSpinBox()
        self.points_spin.setRange(100, 10000)
        self.points_spin.setValue(1000)
        self.points_spin.setDecimals(0)
        points_layout.addWidget(self.points_spin)
        sim_layout.addLayout(points_layout)

        sim_group.setLayout(sim_layout)
        layout.addWidget(sim_group)

        # 6. 预设参数按钮
        preset_layout = QHBoxLayout()
        preset_comfort = QPushButton("舒适型")
        preset_comfort.clicked.connect(lambda: self._load_preset('comfort'))
        preset_sport = QPushButton("运动型")
        preset_sport.clicked.connect(lambda: self._load_preset('sport'))
        preset_offroad = QPushButton("越野型")
        preset_offroad.clicked.connect(lambda: self._load_preset('offroad'))
        preset_layout.addWidget(preset_comfort)
        preset_layout.addWidget(preset_sport)
        preset_layout.addWidget(preset_offroad)
        layout.addLayout(preset_layout)

        # 7. 分析按钮
        self.analyze_btn = QPushButton("开始分析")
        self.analyze_btn.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; font-size: 14px; padding: 10px; }")
        self.analyze_btn.clicked.connect(self._run_analysis)
        layout.addWidget(self.analyze_btn)

        layout.addStretch()

        return panel

    def _create_results_panel(self):
        """创建结果显示面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # 创建选项卡
        self.tabs = QTabWidget()

        # 1. 时域响应选项卡
        time_response_tab = QWidget()
        time_layout = QVBoxLayout(time_response_tab)
        self.time_figure = Figure(figsize=(10, 8))
        self.time_canvas = FigureCanvas(self.time_figure)
        time_layout.addWidget(self.time_canvas)
        self.tabs.addTab(time_response_tab, "时域响应")

        # 2. 频域响应选项卡
        freq_response_tab = QWidget()
        freq_layout = QVBoxLayout(freq_response_tab)
        self.freq_figure = Figure(figsize=(10, 8))
        self.freq_canvas = FigureCanvas(self.freq_figure)
        freq_layout.addWidget(self.freq_canvas)
        self.tabs.addTab(freq_response_tab, "频域响应")

        # 3. 性能指标选项卡
        metrics_tab = QWidget()
        metrics_layout = QVBoxLayout(metrics_tab)
        self.metrics_text = QTextEdit()
        self.metrics_text.setReadOnly(True)
        self.metrics_text.setStyleSheet("font-family: Consolas; font-size: 12px;")
        metrics_layout.addWidget(self.metrics_text)
        self.tabs.addTab(metrics_tab, "性能指标")

        layout.addWidget(self.tabs)

        return panel

    def _on_road_type_changed(self, road_type):
        """路面类型改变时更新参数界面"""
        # 清除现有参数
        while self.road_params_layout.count():
            child = self.road_params_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        if road_type == "正弦波路面":
            # 幅值
            amp_layout = QHBoxLayout()
            amp_layout.addWidget(QLabel("幅值 (m):"))
            self.road_amp = QDoubleSpinBox()
            self.road_amp.setRange(0.001, 0.5)
            self.road_amp.setValue(0.05)
            self.road_amp.setDecimals(3)
            amp_layout.addWidget(self.road_amp)
            self.road_params_layout.addLayout(amp_layout)

            # 波长
            wave_layout = QHBoxLayout()
            wave_layout.addWidget(QLabel("波长 (m):"))
            self.road_wavelength = QDoubleSpinBox()
            self.road_wavelength.setRange(1, 50)
            self.road_wavelength.setValue(5.0)
            self.road_wavelength.setDecimals(1)
            wave_layout.addWidget(self.road_wavelength)
            self.road_params_layout.addLayout(wave_layout)

        elif road_type == "随机路面 (ISO)":
            # 路面等级
            class_layout = QHBoxLayout()
            class_layout.addWidget(QLabel("路面等级:"))
            self.road_class = QComboBox()
            self.road_class.addItems(['A', 'B', 'C', 'D', 'E'])
            self.road_class.setCurrentText('C')
            class_layout.addWidget(self.road_class)
            self.road_params_layout.addLayout(class_layout)

        elif road_type == "单次冲击":
            # 高度
            height_layout = QHBoxLayout()
            height_layout.addWidget(QLabel("高度 (m):"))
            self.road_height = QDoubleSpinBox()
            self.road_height.setRange(0.01, 0.3)
            self.road_height.setValue(0.1)
            self.road_height.setDecimals(3)
            height_layout.addWidget(self.road_height)
            self.road_params_layout.addLayout(height_layout)

            # 宽度
            width_layout = QHBoxLayout()
            width_layout.addWidget(QLabel("宽度 (m):"))
            self.road_width = QDoubleSpinBox()
            self.road_width.setRange(0.1, 5)
            self.road_width.setValue(0.5)
            self.road_width.setDecimals(2)
            width_layout.addWidget(self.road_width)
            self.road_params_layout.addLayout(width_layout)

            # 位置
            pos_layout = QHBoxLayout()
            pos_layout.addWidget(QLabel("位置 (m):"))
            self.road_position = QDoubleSpinBox()
            self.road_position.setRange(0, 100)
            self.road_position.setValue(1.0)
            self.road_position.setDecimals(1)
            pos_layout.addWidget(self.road_position)
            self.road_params_layout.addLayout(pos_layout)

        elif road_type == "梯形路面":
            # 高度
            height_layout = QHBoxLayout()
            height_layout.addWidget(QLabel("高度 (m):"))
            self.road_step_height = QDoubleSpinBox()
            self.road_step_height.setRange(0.01, 0.5)
            self.road_step_height.setValue(0.15)
            self.road_step_height.setDecimals(3)
            height_layout.addWidget(self.road_step_height)
            self.road_params_layout.addLayout(height_layout)

            # 斜坡长度
            ramp_layout = QHBoxLayout()
            ramp_layout.addWidget(QLabel("斜坡长度 (m):"))
            self.road_ramp = QDoubleSpinBox()
            self.road_ramp.setRange(0, 2)
            self.road_ramp.setValue(0.2)
            self.road_ramp.setDecimals(2)
            ramp_layout.addWidget(self.road_ramp)
            self.road_params_layout.addLayout(ramp_layout)

            # 位置
            pos_layout = QHBoxLayout()
            pos_layout.addWidget(QLabel("位置 (m):"))
            self.road_step_position = QDoubleSpinBox()
            self.road_step_position.setRange(0, 100)
            self.road_step_position.setValue(1.0)
            self.road_step_position.setDecimals(1)
            pos_layout.addWidget(self.road_step_position)
            self.road_params_layout.addLayout(pos_layout)

    def _update_damping_ratio(self):
        """更新阻尼比显示"""
        c = self.c_spin.value()
        k2 = self.k2_spin.value()
        m2 = self.m2_spin.value()

        zeta = c / (2 * np.sqrt(k2 * m2))
        self.damping_ratio_label.setText(f"阻尼比: ζ = {zeta:.4f}")

    def _load_preset(self, preset_type):
        """加载预设参数"""
        presets = {
            'comfort': {  # 舒适型
                'm1': 40, 'm2': 320,
                'k1': 190000, 'k2': 16000,
                'c': 1000
            },
            'sport': {  # 运动型
                'm1': 40, 'm2': 320,
                'k1': 190000, 'k2': 25000,
                'c': 1500
            },
            'offroad': {  # 越野型
                'm1': 50, 'm2': 400,
                'k1': 200000, 'k2': 20000,
                'c': 1200
            }
        }

        params = presets.get(preset_type, {})
        self.m1_spin.setValue(params.get('m1', 40))
        self.m2_spin.setValue(params.get('m2', 320))
        self.k1_spin.setValue(params.get('k1', 190000))
        self.k2_spin.setValue(params.get('k2', 16000))
        self.c_spin.setValue(params.get('c', 1000))

    def _create_road_excitation(self):
        """根据界面参数创建路面激励对象"""
        road_type = self.road_type_combo.currentText()
        speed = self.speed_spin.value()

        if road_type == "无激励":
            return None
        elif road_type == "正弦波路面":
            return SineRoadExcitation(
                amplitude=self.road_amp.value(),
                wavelength=self.road_wavelength.value(),
                vehicle_speed=speed
            )
        elif road_type == "随机路面 (ISO)":
            return RandomRoadExcitation(
                road_class=self.road_class.currentText(),
                vehicle_speed=speed,
                duration=self.duration_spin.value(),
                n_samples=int(self.points_spin.value())
            )
        elif road_type == "单次冲击":
            return ImpulseRoadExcitation(
                height=self.road_height.value(),
                width=self.road_width.value(),
                position=self.road_position.value(),
                vehicle_speed=speed
            )
        elif road_type == "梯形路面":
            return StepRoadExcitation(
                height=self.road_step_height.value(),
                ramp_length=self.road_ramp.value(),
                position=self.road_step_position.value(),
                vehicle_speed=speed
            )

    def _run_analysis(self):
        """运行分析"""
        try:
            # 禁用分析按钮
            self.analyze_btn.setEnabled(False)
            self.analyze_btn.setText("分析中...")

            # 获取参数
            m1 = self.m1_spin.value()
            m2 = self.m2_spin.value()
            k1 = self.k1_spin.value()
            k2 = self.k2_spin.value()
            c = self.c_spin.value()

            # 创建路面激励
            road_excitation = self._create_road_excitation()

            # 创建求解器
            self.solver = SuspensionSolver(m1, m2, k1, k2, c, road_excitation)

            # 仿真参数
            time_span = (0, self.duration_spin.value())
            n_points = int(self.points_spin.value())

            # 创建后台线程
            self.analysis_thread = AnalysisThread(self.solver, time_span, n_points)
            self.analysis_thread.finished.connect(self._on_analysis_finished)
            self.analysis_thread.error.connect(self._on_analysis_error)
            self.analysis_thread.start()

        except Exception as e:
            self.analyze_btn.setEnabled(True)
            self.analyze_btn.setText("开始分析")
            QMessageBox.critical(self, "错误", f"分析失败:\n{str(e)}")

    def _on_analysis_finished(self, results):
        """分析完成"""
        self.results = results
        self.analyze_btn.setEnabled(True)
        self.analyze_btn.setText("开始分析")

        # 显示结果
        self._plot_time_response()
        self._plot_frequency_response()
        self._display_metrics()

        QMessageBox.information(self, "成功", "分析完成！")

    def _on_analysis_error(self, error_msg):
        """分析出错"""
        self.analyze_btn.setEnabled(True)
        self.analyze_btn.setText("开始分析")
        QMessageBox.critical(self, "错误", f"分析失败:\n{error_msg}")

    def _plot_time_response(self):
        """绘制时域响应"""
        if self.results is None:
            return

        self.time_figure.clear()

        t = self.results['time']
        x0 = self.results['x0']
        x1 = self.results['x1']
        x2 = self.results['x2']
        a2 = self.results['a2']

        # 创建4个子图
        ax1 = self.time_figure.add_subplot(4, 1, 1)
        ax1.plot(t, x0 * 1000, 'k-', linewidth=1.5, label='路面输入 x₀')
        ax1.set_ylabel('位移 (mm)')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right')
        ax1.set_title('悬架系统时域响应', fontsize=14, fontweight='bold')

        ax2 = self.time_figure.add_subplot(4, 1, 2)
        ax2.plot(t, x1 * 1000, 'b-', linewidth=1.5, label='轮胎位移 x₁')
        ax2.plot(t, x2 * 1000, 'r-', linewidth=1.5, label='车身位移 x₂')
        ax2.set_ylabel('位移 (mm)')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper right')

        ax3 = self.time_figure.add_subplot(4, 1, 3)
        suspension_travel = (x2 - x1) * 1000
        ax3.plot(t, suspension_travel, 'g-', linewidth=1.5, label='悬架行程 (x₂-x₁)')
        ax3.set_ylabel('行程 (mm)')
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='upper right')
        ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)

        ax4 = self.time_figure.add_subplot(4, 1, 4)
        ax4.plot(t, a2, 'm-', linewidth=1.5, label='车身加速度 a₂')
        ax4.set_xlabel('时间 (s)')
        ax4.set_ylabel('加速度 (m/s²)')
        ax4.grid(True, alpha=0.3)
        ax4.legend(loc='upper right')
        ax4.axhline(y=0, color='k', linestyle='--', alpha=0.5)

        self.time_figure.tight_layout()
        self.time_canvas.draw()

    def _plot_frequency_response(self):
        """绘制频域响应"""
        if self.solver is None:
            return

        self.freq_figure.clear()

        # 计算频率响应函数
        frequencies, H_x2, H_a2 = self.solver.frequency_response()

        # 创建2个子图
        ax1 = self.freq_figure.add_subplot(2, 1, 1)
        ax1.plot(frequencies, 20 * np.log10(H_x2 + 1e-10), 'b-', linewidth=2)
        ax1.set_ylabel('位移幅值 (dB)')
        ax1.set_title('频率响应函数', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([0, 50])

        # 标注固有频率
        for i, fn in enumerate(self.results['natural_frequencies']):
            ax1.axvline(x=fn, color='r', linestyle='--', alpha=0.5, label=f'f₍{i + 1}₎={fn:.2f} Hz' if i < 2 else '')
        ax1.legend()

        ax2 = self.freq_figure.add_subplot(2, 1, 2)
        ax2.plot(frequencies, 20 * np.log10(H_a2 + 1e-10), 'r-', linewidth=2)
        ax2.set_xlabel('频率 (Hz)')
        ax2.set_ylabel('加速度幅值 (dB)')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim([0, 50])

        # 标注固有频率
        for fn in self.results['natural_frequencies']:
            ax2.axvline(x=fn, color='r', linestyle='--', alpha=0.5)

        self.freq_figure.tight_layout()
        self.freq_canvas.draw()

    def _display_metrics(self):
        """显示性能指标"""
        if self.results is None:
            return

        params = self.results['parameters']
        comfort = self.results['comfort_metrics']
        fn = self.results['natural_frequencies']
        zeta = self.results['damping_ratios']

        report = f"""
╔════════════════════════════════════════════════════════════╗
║               悬架系统性能分析报告                         ║
╚════════════════════════════════════════════════════════════╝

【系统参数】
────────────────────────────────────────────────────────────
  非簧载质量 m₁:      {params['m1']:.1f} kg
  簧载质量 m₂:        {params['m2']:.1f} kg
  轮胎刚度 k₁:        {params['k1']:.0f} N/m
  悬架刚度 k₂:        {params['k2']:.0f} N/m
  阻尼系数 c:         {params['c']:.0f} N·s/m

【固有特性】
────────────────────────────────────────────────────────────
  第一阶固有频率:     {fn[0]:.3f} Hz
  第二阶固有频率:     {fn[1]:.3f} Hz
  模态阻尼比 ζ₁:      {zeta[0]:.4f}
  模态阻尼比 ζ₂:      {zeta[1]:.4f}

【舒适性指标】
────────────────────────────────────────────────────────────
  RMS加速度:          {comfort['rms_acceleration']:.4f} m/s²
  最大加速度:         {comfort['max_acceleration']:.4f} m/s²
  舒适性评级:         {comfort['comfort_rating']}

【操纵稳定性指标】
────────────────────────────────────────────────────────────
  最大悬架行程:       {comfort['max_suspension_travel'] * 1000:.2f} mm
  轮胎动载荷系数:     {comfort['dynamic_load_coefficient']:.3f}

【评价】
────────────────────────────────────────────────────────────
"""

        # 添加评价
        if comfort['rms_acceleration'] < 0.5:
            report += "  ✓ 舒适性优秀\n"
        elif comfort['rms_acceleration'] < 1.0:
            report += "  ✓ 舒适性良好\n"
        else:
            report += "  ✗ 舒适性有待改善\n"

        if comfort['dynamic_load_coefficient'] < 2.0:
            report += "  ✓ 路面附着良好\n"
        else:
            report += "  ✗ 可能存在车轮跳离路面风险\n"

        if comfort['max_suspension_travel'] < 0.08:
            report += "  ✓ 悬架行程充裕\n"
        else:
            report += "  ⚠ 悬架行程较大，注意限位\n"

        self.metrics_text.setText(report)


def main():
    """主函数"""
    app = QApplication(sys.argv)

    # 设置应用样式
    app.setStyle('Fusion')

    window = SuspensionAnalysisWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()