# 稳定连续三阶翻转的研究验收

目标是复现参考视频中的被动翻转：起步时将弹簧圈弯过阶边，放手后靠初始储能、重力、弹性和接触继续前进。当前发布的 v0.1.0 没有通过这项验收。

已有 12 圈拱形释放算例在 1.5 s 中，由同一个材料端连续接触下两阶，属于滑移。旧 12 圈倾斜起步算例在粗网格观察到三次端圈交替支撑，但减半步长的第二次支撑时间变化超过 5%，加密计算超时；不能据此宣称稳定复现。

## 固定判据

- 释放后的模型没有驱动器、施加力或重置姿态。
- 三个连续下降踏面依次获得两材料端的交替支撑。每次端圈踏面法向力达到 0.1Mg 并持续至少 20 ms；立面擦碰、短暂触碰、中部先承重或同端滑移不算确认翻转。
- 轨迹和接触姿态应显示端圈抬起、越过支撑端并落到下一阶；仅靠接触计数不代表视频中的步态。
- 完成设定时间，记录全部警告和失败。最大接触穿透不超过条带厚度的一半，且不发生侧落。
- 同一物理配置将时间步长减半、每圈杆段数加倍后，仍获得三阶翻转，前三次确认支撑时间相对变化各不超过 5%。
- 起步位置正负 2 mm、拱高正负 2% 的独立释放仍能三阶翻转；这些检查通过后才标记起步稳定性。

这些是数值验收，不能代替实物标定。没有尺寸、质量、静态力—伸长和耗散测量时，扫描得到的材料参数仍是工程假设。原视频没有米制或时间标定，不公开上传原视频或派生视频帧。

释放前还需进行几何预检。起步的圈间或台阶穿透若超过条带厚度的 5%（最低阈值 1 μm），直接记录初态失败，不开始时间积分。固定杆长及端点的姿态生成本身不能保证避障，必须检查实际碰撞几何；不能把释放前的几何冲突当作初始支撑。

## 网格一致的阻尼

`material.damping` 保留原有每关节阻尼定义（N·m·s），用于兼容旧记录。新研究配置使用 `material.rotational_viscosity`（N·m²·s），对长度 L 的球关节设置 c=η/L。其离散耗散趋近于 ∫η|曲率变化率|² ds，加密不会因为增加关节而自动改变连续杆黏性。根部自由体不添加对地阻尼。

扫描参数在 [search_gait.py](../scripts/search_gait.py) 中预先声明，包含弹性模量与重量比例、台阶尺寸、起步姿态和圈数。每个案例和检查保留完整配置、模型、HDF5 轨迹、接触证据、能量及失败原因。通过搜索只表示候选，还需独立检查。

```bash
python scripts/search_gait.py --list
python scripts/search_gait.py --case n12-g4-h06 --output reports/gait-search
python scripts/search_gait.py --case n12-g4-h06 --variant half_dt --output reports/gait-half-dt
python scripts/search_gait.py --case n12-g4-h06 --variant mesh --output reports/gait-mesh
```

GitHub Actions 的 `Stair walking research validation` 工作流以 `gait-search` 模式在标准 Ubuntu runner 上并行计算。每个案例有墙钟上限，失败也上传实际生成的数据；该工作流不发布镜像或修改 `latest`。

## 模型依据

采用 [MuJoCo 3.15.0 elasticity.cable](https://github.com/google-deepmind/mujoco/blob/3.15.0/plugin/elasticity/cable.cc) 的应力自由参考曲率、矩形截面弯曲和扭转。局部宽度沿径向，厚度近似沿圈轴方向；视频没有提供截面实测支持。初始拱形改变运行时 qpos，保持 qpos0 应力自由参考不变。
