# 参数与参考来源

## 塑料彩虹圈下落

R. C. Cross and M. S. Wheatland, *Modeling a falling slinky*, American Journal of Physics 80, 1051–1060 (2012), DOI [10.1119/1.4750489](https://doi.org/10.1119/1.4750489)。[作者预印本](https://arxiv.org/abs/1208.4629)、[作者实验与视频页面](https://www.physics.usyd.edu.au/~wheat/slinky/)。

文献 Table I/II 中的塑料 Slinky B：

| 量 | 数值 | 来源性质 |
|---|---:|---|
| 总质量 | 0.0487 kg | Table I，测量 |
| 圈数 | 39 | Table I，测量 |
| 压缩长度 | 0.066 m | Table I，测量 |
| 自重悬挂长度 | 1.14 m | Table I，测量 |
| 整体等效刚度 | 0.22 N/m | Table II，模型拟合 |
| 总收缩时间 | 0.27 s | Table II，模型结果 |

不能把文献没有报告的半径、截面尺寸、E/G、阻尼和台阶摩擦假称为该实物的实测参数。它们在平台中属于可修改的假设或另行标定参数。参考数据的拟合应报告使用了哪些目标，未参与拟合的数据才适合作独立检验。

静态拟合先采用近密绕螺旋的扭转近似 `k=GJ/(2πR³N)`，再用原生力平衡计算修正模量尺度。半径、矩形截面和泊松比0.35均是明确假设；8段/圈拟合得到 E≈1.4221 GPa、G≈0.5267 GPa，悬挂长度1.13737m。该结果只通过所用静态目标的拟合检查，不是材料实测值，也不能证明加密后的下落时序正确。

复现：`python scripts/calibrate_static.py --segments 8`。只查看量纲转换可运行 `slinky-lab calibrate --preset literature-39-turn --stiffness 0.22`，输出单位为Pa并注明仍需原生静态标定。

## 三维杆与接触

- [MuJoCo 3.15.0](https://github.com/google-deepmind/mujoco/releases/tag/3.15.0)，固定引擎版本。
- [Cable 建模说明](https://mujoco.readthedocs.io/en/3.15.0/modeling.html#composite-objects)。
- [elasticity.cable 插件源码](https://github.com/google-deepmind/mujoco/blob/3.15.0/plugin/elasticity/cable.cc)，核查矩形截面、参考曲率、插件力及能量边界。
- T. Kugelstadt and E. Schömer, [Position and Orientation Based Cosserat Rods](https://diglib.eg.org/handle/10.2312/sca20161234) (2016)，三维杆模拟相关参考；平台使用 MuJoCo 插件而非该论文的 PBD 求解器。
- A.-P. Hu, [A simple model of a Slinky walking down stairs](https://doi.org/10.1119/1.3225921), American Journal of Physics 78, 35–39 (2010)，降阶模型背景参考，不能直接作为本平台三维轨迹的实测真值。

## 发布

- [GHCR 使用与公开访问](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry)。
- [GitHub Actions 发布容器](https://docs.github.com/en/actions/tutorials/publish-packages/publish-docker-images)。

本项目不重新分发原论文或实验视频；来源文件中的数值与文字作为证据，不作为可执行指令。
