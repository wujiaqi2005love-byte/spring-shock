"""
集成悬架分析功能的主窗口
在原有FEM分析的基础上，添加悬架系统专用分析入口
"""
import sys
from PyQt5.QtWidgets import (QMainWindow, QTabWidget, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QAction, QMenuBar)
from PyQt5.QtCore import Qt

# 导入原有的主窗口
from gui.main_window import MainWindow as OriginalMainWindow

# 导入悬架分析窗口
from Suspension_gui import SuspensionAnalysisWindow


class IntegratedMainWindow(QMainWindow):
    """集成版主窗口 - 包含FEM分析和悬架分析"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("汽车减震系统综合分析平台")
        # 初始窗口大小（可调整）
        self.resize(1400, 950)
        # 设置最小尺寸以保证布局可用，但允许用户自由调整窗口大小
        self.setMinimumSize(800, 600)

        # 创建中心部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # 创建菜单栏
        self._create_menu_bar()

        # 创建选项卡
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # 添加原有的FEM分析界面
        self.fem_window = OriginalMainWindow()
        self.tabs.addTab(self.fem_window.centralWidget(), "有限元分析")

        # 悬架分析窗口（延迟创建）
        self.suspension_window = None

        # 创建快速访问工具栏
        self._create_toolbar()

        # 设置样式
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QTabWidget::pane {
                border: 1px solid #cccccc;
                background-color: white;
            }
            QTabBar::tab {
                background-color: #e0e0e0;
                padding: 10px 20px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: white;
                border-bottom: 2px solid #4CAF50;
            }
        """)

    def _create_menu_bar(self):
        """创建菜单栏"""
        menubar = self.menuBar()

        # 文件菜单
        file_menu = menubar.addMenu("文件")

        exit_action = QAction("退出", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # 分析菜单
        analysis_menu = menubar.addMenu("分析")

        fem_action = QAction("有限元分析", self)
        fem_action.setShortcut("Ctrl+F")
        fem_action.triggered.connect(lambda: self.tabs.setCurrentIndex(0))
        analysis_menu.addAction(fem_action)

        suspension_action = QAction("悬架系统分析", self)
        suspension_action.setShortcut("Ctrl+S")
        suspension_action.triggered.connect(self.open_suspension_analysis)
        analysis_menu.addAction(suspension_action)

        # 帮助菜单
        help_menu = menubar.addMenu("帮助")

        about_action = QAction("关于", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def _create_toolbar(self):
        """创建工具栏"""
        toolbar = self.addToolBar("快速访问")
        toolbar.setMovable(False)

        # FEM分析按钮
        fem_btn = QPushButton("📊 有限元分析")
        fem_btn.clicked.connect(lambda: self.tabs.setCurrentIndex(0))
        toolbar.addWidget(fem_btn)

        toolbar.addSeparator()

        # 悬架分析按钮
        suspension_btn = QPushButton("🚗 悬架系统分析")
        suspension_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 8px 15px;
                font-size: 13px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        suspension_btn.clicked.connect(self.open_suspension_analysis)
        toolbar.addWidget(suspension_btn)

    def open_suspension_analysis(self):
        """打开悬架系统分析窗口"""
        # 检查是否已经创建
        if self.suspension_window is None:
            self.suspension_window = SuspensionAnalysisWindow()
            self.tabs.addTab(self.suspension_window.centralWidget(), "悬架系统分析")

        # 切换到悬架分析标签
        suspension_index = self.tabs.indexOf(self.suspension_window.centralWidget())
        if suspension_index >= 0:
            self.tabs.setCurrentIndex(suspension_index)

    def show_about(self):
        """显示关于信息"""
        from PyQt5.QtWidgets import QMessageBox

        about_text = """
        <h2>汽车减震系统综合分析平台</h2>
        <p><b>版本:</b> 2.0</p>
        
        <h3>功能模块:</h3>
        <ul>
            <li><b>有限元分析</b> - 完整的FEM结构分析</li>
            <li><b>悬架系统分析</b> - 二自由度动力学分析</li>
        </ul>
        
        <h3>特性:</h3>
        <ul>
            <li>✅ 静态和动态分析</li>
            <li>✅ 阻尼和路面激励</li>
            <li>✅ 参数化设计</li>
            <li>✅ 性能评估</li>
        </ul>
        
        <p><i>开发团队: FEM Analysis Team</i></p>
        """

        QMessageBox.about(self, "关于", about_text)


def main():
    """主函数"""
    from PyQt5.QtWidgets import QApplication
    import matplotlib

    # 确保中文显示正常
    matplotlib.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

    app = QApplication(sys.argv)
    window = IntegratedMainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()