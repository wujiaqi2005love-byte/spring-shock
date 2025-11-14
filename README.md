## spring-shock

> 一个用于弹簧有限元/悬架仿真的小型 Python 项目（演示与学习用途）。

## 概要
仓库包含一个简单的弹簧/悬架 FEA 仿真程序，以及用于网格、材料、可视化和 GUI 的模块。

## 目录结构（简要）
- `complete-spring-fea-software/` - 主代码目录
  - `main.py` - 程序入口（示例运行脚本）
  - `analysis/` - 求解器相关代码
  - `gui/` - 界面相关代码
  - `material/`, `meshing/`, `model/`, `visualization/` - 功能模块

## 运行（本地）
建议使用虚拟环境运行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt   # 如果你有 requirements.txt
python complete-spring-fea-software\main.py
```

如果仓库中没有 `requirements.txt`，请根据需要安装常用科学计算包，例如 `numpy`、`scipy`、`matplotlib` 等。

## 贡献
欢迎提交 issue 或 pull request。若要贡献代码，请先创建分支并在本地运行测试/示例。

