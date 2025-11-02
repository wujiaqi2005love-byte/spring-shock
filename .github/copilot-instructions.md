## 目标
为 AI 编码助手提供立即上手本仓库的关键信息：总体结构、重要契约（API/数据形状）、运行/调试命令、以及常见陷阱与示例位置。

## 快速入口（先看这些文件）
- `complete-spring-fea-software/main.py`  — 程序入口（示例运行脚本）。
- `complete-spring-fea-software/meshing/mesher.py` — 网格生成，依赖 Gmsh（返回 mesh dict）。
- `complete-spring-fea-software/material/material.py` — Material API（需提供 get_elastic_matrix() 和属性 `rho`）。
- `complete-spring-fea-software/analysis/boundary.py` — 边界条件契约（apply_boundary_conditions, expand_displacement 等）。
- `complete-spring-fea-software/analysis/solver.py` — 核心求解器（FEMSolver），包含 assemble_*/solve/solve_dynamic 等方法。
- `complete-spring-fea-software/visualization/plotter.py` — 结果可视化（PyVista + Matplotlib）。

## 大局架构（短言）
- 数据流：网格（Mesher） -> 网格字典 mesh (见下) -> 边界条件（BoundaryConditions） -> 求解器（FEMSolver） -> 可视化（ResultsPlotter）。
- 意图：轻量教学/演示型 FEA 实现，代码在单机进程中同步调用（无服务边界或并发框架）。

## 关键契约（必须遵守）
- mesh 格式（由 `Mesher.generate_mesh` 返回）：
  - `mesh['nodes']` : numpy.ndarray, shape (n_nodes, 3)
  - `mesh['elements']` : ndarray/list, 每行是节点索引（注意：代码中使用 0-based 索引，Mesher 会把 gmsh 的 1-based 转为 0-based）
  - `mesh['type']` : `'triangle'` 或 `'tetrahedron'`

- Material 对象：必须实现 `get_elastic_matrix()` → 返回 6x6 numpy 矩阵（solver 假设按 3D 弹性矩阵组织）。并暴露 `rho`（密度）。（参见 `material/material.py`）

- BoundaryConditions 类（`analysis/boundary.py`）常用接口：
  - `bc.node_forces` : 长度为 3*n_nodes 的力向量
  - `bc.apply_boundary_conditions(K_csr, f)` -> `(reduced_K, reduced_f)`
  - `bc.expand_displacement(reduced_u)` -> full_u (3*n_nodes,)
  - `bc.get_stiffness_matrix_mask()` -> 布尔 mask
 这些函数在 `FEMSolver.solve` 与 `solve_dynamic` 中被直接调用。

- FEMSolver（`analysis/solver.py`）常用方法：
  - `assemble_stiffness_matrix()`, `assemble_mass_matrix()`, `assemble_damping_matrix()`
  - `solve()` -> 静态 K*u=F
  - `solve_dynamic(time_span, ...)` -> 动态历时解（M*a + C*v + K*u = F(t)），支持 `damping_config`：
      `{'type':'rayleigh'|'modal'|'proportional','alpha':..,'beta':..,'viscous_coeff':..}`

## 依赖 & 运行（开发者/AI 需要知道）
- 主要第三方：numpy, scipy, matplotlib, pyvista, gmsh。PyVista + gmsh 可能需要可视化/系统依赖。
- 在 Windows/PowerShell 下示例运行：
```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install numpy scipy matplotlib pyvista gmsh
python complete-spring-fea-software\main.py
```
- 仓库没有统一的 `requirements.txt`，因此在自动更改依赖或运行示例时请先确认实际环境。

## 常见约束与陷阱（从源码可观察到）
- 单元几何退化会抛异常（面积/体积接近 0），例如 `calculate_triangle_stiffness`、`calculate_tetrahedron_volume` 会报错；修改网格时请注意节点顺序和拓扑完整性。
- Mesher 假设 Gmsh 返回的 element type 编号：3=triangle, 4=tetrahedron，且已把索引 -1 转为 0-based。
- Plotter 假设 `results['displacement']` 可 reshape 为 (-1,3)；若输出的位移是标量或缺项，Plotter 会抛出异常。

## 编辑/实现新特性时的具体示例
- 若改进阻尼策略：编辑 `analysis/solver.py::assemble_damping_matrix`，保持 `damping_config` 字典键不变以兼容现有调用。
- 新网格输入（STEP/STL）应通过 `meshing/mesher.py::generate_mesh` 导入并返回与现有 `mesh` 字典一致的结构。
- 若要在不打开 GUI 的情况下自动化测试 `solve_dynamic`，构造一个小网格（triangle/tetra），提供 `BoundaryConditions`（调用 `auto_detect_fixed_and_load_faces`），并传入固定 `time_dependent_force` 函数以得到可确定的时间历程。

## 调试建议（对 AI 助手有用）
- 当遇到线性求解失败（spsolve 报错），检查 `bc.get_stiffness_matrix_mask()` 是否将足够的自由度保留（mask 长度与矩阵维度匹配）。
- 对于动力学问题，`odeint` 解算器可能对刚性系统不稳定；若数值问题，建议将矩阵转换为稀疏 CSR 并用更稳健的求解器或减小时间步。

## 编辑风格 / 约定
- 保持 numpy 原生数组与 scipy sparse 的分离（代码中大量使用 `lil_matrix` 并在求解前转为 `csr()`）。
- 遵循现有命名（`mesh`, `material`, `bc`, `FEMSolver`）以降低改动引入的整合成本。

如果需要我把这些点合并到 README 或补充 `requirements.txt`、添加示例脚本或小型单元测试，我可以继续编辑——请告诉我你优先希望改进的部分或补充的约定。
