# 稳定连续三阶翻转的研究验收

目标是复现参考视频中的被动翻转：起步时将弹簧圈弯过阶边，放手后靠初始储能、重力、弹性和接触继续前进。当前发布的 v0.1.0 没有通过这项验收。

已有 12 圈拱形释放算例在 1.5 s 中，由同一个材料端连续接触下两阶，属于滑移。旧 12 圈倾斜起步算例在粗网格观察到三次端圈交替支撑，但减半步长的第二次支撑时间变化超过 5%，加密计算超时；不能据此宣称稳定复现。

## 固定判据

- 释放后的模型没有驱动器、施加力或重置姿态。
- 三个连续下降踏面依次获得两材料端的交替支撑。每次端圈踏面法向力达到 0.1Mg 并持续至少 20 ms；立面擦碰、短暂触碰、中部先承重或同端滑移不算确认翻转。
- 轨迹和接触姿态应显示端圈抬起、越过支撑端并落到下一阶；仅靠接触计数不代表视频中的步态。
- 拱形起步的首个自由端落在第 1 阶属于落位，需与完整翻转分开。验证三次完整摆越应继续观察落在第 2、3、4 阶的动作，不能将三个支撑事件直接称为三次完整翻转。
- 完成设定时间，记录全部警告和失败。最大接触穿透不超过条带厚度的一半，且不发生侧落。
- 同一物理配置将时间步长减半、每圈杆段数加倍后，仍获得三次完整翻转，落在第 2、3、4 阶的确认支撑时间相对变化各不超过 5%。
- 起步位置正负 2 mm、拱高正负 2% 的独立释放仍能三阶翻转；这些检查通过后才标记起步稳定性。

这些是数值验收，不能代替实物标定。没有尺寸、质量、静态力—伸长和耗散测量时，扫描得到的材料参数仍是工程假设。原视频没有米制或时间标定，不公开上传原视频或派生视频帧。

释放前还需进行几何预检。起步的圈间或台阶穿透若超过条带厚度的 5%（最低阈值 1 μm），直接记录初态失败，不开始时间积分。固定杆长及端点的姿态生成本身不能保证避障，必须检查实际碰撞几何；不能把释放前的几何冲突当作初始支撑。

`scripts/review_gait_motion.py` 对保存的完整端圈盒形几何进行动作核查：从前一阶首次接触开始，确认材料端的前后顺序交换、移动端下包络高于支撑端上包络、离开原踏面抬起，再在目标踏面形成持续支撑。越端的两帧中至少一帧须有支撑端踏面力达到 0.1Mg。保存帧只能提供采样证据，报告保留时间区间和未采样区间的局限，不能证明帧间连续接触或连续几何间隙。

## 实际搜索结果（2026-10-08）

研究分支中的首次粗网格 `macro-arch` 轨迹通过三次完整动作核查，但起步有 0.5897 mm 圈间穿透，超过其 0.225 mm 初态阈值，不能作为合格候选。

| 无初始穿透的案例 | 完整动作核查 | 最终状态 | 结论 |
|---|---|---|---|
| macro-safe-gap，50 μs | 2/3 | 侧落 | 未通过 |
| macro-safe-rise，50 μs | 1/3 | 侧落 | 未通过 |
| macro-safe-high，50 μs | 1/3 | 两次端圈交替支撑 | 未通过 |
| macro-safe-gap，25 μs | 1/3 | 侧落 | 未通过 |

这批结果来自独立 Linux 运行 [37726058575](https://github.com/FanYu-Nijika/slinky-lab/actions/runs/37726058575) 和 [37726076192](https://github.com/FanYu-Nijika/slinky-lab/actions/runs/37726076192)。后者的 16 段/圈检查仍在计算；并未通过验收。9 组无初始穿透的薄条带高模量配置也出现 BADQACC，失败时间与原轨迹保存在各自附件中。

下一批预先声明了 12 组宏观条带比较：阻尼、圈间摩擦、半径、截面厚度与端部圈数。所有案例的初始碰撞穿透为零，起步速度为零，台阶宽度保持 0.3 m。厚度比较按 t⁻³ 调整两个材料模量，只是保持闭圈轴向刚度数量级的工程假设，仍须用实际模型及独立加密检查确认。它们不是成功预设，也没有实物参数标定。

## 网格一致的阻尼

`material.damping` 保留原有每关节阻尼定义（N·m·s），用于兼容旧记录。新研究配置使用 `material.rotational_viscosity`（N·m²·s），对长度 L 的球关节设置 c=η/L。其离散耗散趋近于 ∫η|曲率变化率|² ds，加密不会因为增加关节而自动改变连续杆黏性。根部自由体不添加对地阻尼。

扫描参数在 [search_gait.py](../scripts/search_gait.py) 中预先声明，包含弹性模量与重量比例、台阶尺寸、起步姿态和圈数。每个案例和检查保留完整配置、模型、HDF5 轨迹、接触证据、能量及失败原因。通过搜索只表示候选，还需独立检查。

```bash
python scripts/search_gait.py --list
python scripts/search_gait.py --case macro-damp3-rise --output reports/gait-search
python scripts/search_gait.py --case macro-damp3-rise --variant half_dt --output reports/gait-half-dt
python scripts/search_gait.py --case macro-damp3-rise --variant mesh --output reports/gait-mesh
python scripts/review_gait_motion.py reports/gait-search/macro-damp3-rise-base --output reports/motion-review.json
```

GitHub Actions 的 `Stair walking research validation` 工作流以 `gait-search` 模式在标准 Ubuntu runner 上并行计算。每个案例有墙钟上限，失败也上传实际生成的数据；该工作流不发布镜像或修改 `latest`。

## 模型依据

采用 [MuJoCo 3.15.0 elasticity.cable](https://github.com/google-deepmind/mujoco/blob/3.15.0/plugin/elasticity/cable.cc) 的应力自由参考曲率、矩形截面弯曲和扭转。局部宽度沿径向，厚度近似沿圈轴方向；视频没有提供截面实测支持。初始拱形改变运行时 qpos，保持 qpos0 应力自由参考不变。
