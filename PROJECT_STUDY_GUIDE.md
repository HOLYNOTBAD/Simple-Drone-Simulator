# Simple-Drone-Simulator 学习导图

这份文档的目标不是重复源码注释，而是给你一条可以“快速建立全局模型，再按模块精读”的路径。项目本质上是一个面向无人机拦截/跟踪实验的仿真框架，核心由 5 层组成：

1. 配置与脚本入口
2. 状态/指令协议层
3. 控制层
4. 动力学与传感器模型层
5. 调度、可视化、日志层

如果只抓主线，可以把整个系统理解成下面这条链：

```text
YAML配置
  -> scripts/*.py 组装对象
  -> Scheduler 决定不同频率的执行时机
  -> Camera 生成视觉测量
  -> PerfectObserver 组装 Observation
  -> 高层控制器(IBVS / IBVS-SO3 / Position Demo)
  -> BasicController 低层级联 + 力矩/推力分配
  -> Motors 电机/旋翼模型
  -> RigidBody6DoF 刚体动力学
  -> Simulator / Logger / Metrics 做显示和记录
```

---

## 1. 项目目录怎么读

### 1.1 最重要的目录

- `scripts/`: 入口脚本，定义一次完整仿真如何搭起来。
- `configs/`: 仿真参数、控制器参数、初值、可视化参数。
- `models/`: 刚体、电机、目标、状态协议。
- `control/`: 高层控制器和 PX4 风格低层控制链。
- `sensors/`: 当前主要是针孔相机。
- `observe/`: 把真实状态和传感器输出拼成控制器真正看到的 `Observation`。
- `sim/`: 多频率调度和实时可视化。
- `utils/`: 四元数/旋转矩阵、日志、指标等公共工具。
- `visualization/`: 回放与监控图。

### 1.2 推荐先读哪些文件

第一遍建议按这个顺序：

1. `scripts/ibvs_ctrl_sim.py`
2. `models/state.py`
3. `control/ibvs_controller.py`
4. `control/basic_control/basic_controller.py`
5. `models/motors.py`
6. `models/rigid_body.py`
7. `sensors/camera.py`
8. `observe/perfect.py`

这样你会先看到“系统怎么跑起来”，再看“模块之间传什么”，最后看“每个模块怎么算”。

---

## 2. 核心架构：谁负责什么

## 2.1 运行时对象关系

典型 IBVS 仿真入口在 `scripts/ibvs_ctrl_sim.py` 或 `scripts/ibvs_so3_ctrl_sim.py`，它们做的事情基本一致：

1. 从 YAML 读参数。
2. 构造 `MultiRateScheduler`。
3. 构造 `RigidBody6DoF`、`Motors`、`TargetPointMass`。
4. 构造 `PinholeCamera`。
5. 构造 `PerfectObserver`。
6. 构造高层控制器，如 `IBVSController` 或 `IBVSSO3Controller`。
7. 构造 `BasicController`，把高层命令继续变成电机电流命令。
8. 主循环里按不同频率执行相机、控制、物理积分、可视化、日志。

所以这个项目不是“控制器直接驱动物理模型”，而是分成两层：

- 高层控制器：给出 `ControlCommand`，也就是“总推力 + 角速度命令”。
- 低层 BasicController：把高层命令继续变成 `ForceSetpoint` 和 `MotorCommand`。

这是一个非常关键的理解点。

## 2.2 当前代码里的两套接口状态

项目里其实同时存在“当前接口”和“目标接口”。

### 当前真正跑通的接口

高层控制器输出：

```python
ControlCommand(t, thrust, omega_cmd_b)
```

然后通过：

```python
BasicController.step_from_command(...)
```

进入低层控制。

### 设计目标接口

README 和 `basic_control` 的结构说明，目标是让高层控制器直接输出某种 setpoint：

```text
PositionSetpoint
  -> VelocitySetpoint
  -> AttThrustSetpoint
  -> RateThrustSetpoint
  -> ForceSetpoint
  -> MotorCommand
```

也就是说，项目正在从“高层直接给 rate+thrust”逐步演进到“高层给 setpoint，低层统一跟踪”。

你读代码时一定要把这两件事区分开：

- `basic_control/` 体现的是目标架构。
- `ibvs_controller.py` 和 `ibvs_so3_controller.py` 仍然使用旧式 `ControlCommand` 输出。

---

## 3. 协议层：整个项目最重要的接口文件

协议层定义在 `models/state.py` 和 `control/basic_control/setpoints.py`。

### 3.1 为什么它最重要

这个项目模块之间基本不直接传裸 numpy 数组，而是传 dataclass。这样做有两个好处：

1. 每层输入输出语义清楚。
2. `__post_init__` 会强制检查维度和数值合法性，接口错了会尽早炸出来。

### 3.2 主要数据结构

#### UAVState

表示无人机真值状态：

```text
t      时间
p_e    世界系位置
v_e    世界系速度
q_eb   四元数，body -> world
w_b    机体系角速度
```

#### TargetState

目标点质量模型状态：

```text
t, p_e, v_e
```

#### CameraMeasurement

相机量测：

```text
p_norm   归一化像平面坐标 [x/z, y/z]
uv_px    像素坐标
range_m  距离
valid    是否在视场内
```

#### Observation

控制器真正看到的量：

```text
t
p_norm
q_eb
w_b
v_e
p_r
v_r
has_target
```

这里的相对量定义非常重要：

$$
p_r = p_{uav} - p_{tgt}, \quad v_r = v_{uav} - v_{tgt}
$$

所以“目标相对无人机”的向量其实是：

$$
p_{tgt/uav} = -p_r, \quad v_{tgt/uav} = -v_r
$$

#### ControlCommand

当前高层控制器输出：

$$
u = (f, \omega_{cmd}^b)
$$

其中：

- `thrust`: 总推力标量
- `omega_cmd_b`: 机体系角速度命令

#### ForceSetpoint

低层率控制器输出的“机体总推力 + 机体系力矩目标”：

$$
u_f = (T, \tau_b)
$$

#### MotorCommand

电机模型真正执行的命令：

```text
motor_current_cmd: 4个电机电流命令
```

#### Setpoints

`setpoints.py` 定义了 4 类控制目标：

- `PositionSetpoint`
- `VelocitySetpoint`
- `AttThrustSetpoint`
- `RateThrustSetpoint`

这 4 个类是“未来统一接口”的核心。

---

## 4. 坐标系与方向约定

这个项目最值得先吃透的不是控制律，而是坐标系。

### 4.1 世界系

世界系 `{e}` 使用 NED：

- x: North
- y: East
- z: Down

因此重力在世界系中写成：

$$
g_e = [0, 0, g]^T
$$

### 4.2 机体系

机体系 `{b}` 使用 FRD：

- x: Forward
- y: Right
- z: Down

这是右手系。

### 4.3 相机系

相机系 `{c}` 使用：

- x: right
- y: down
- z: forward

零安装角时，代码里使用的 body 到 camera 映射是：

$$
z_c = x_b, \quad x_c = y_b, \quad y_c = z_b
$$

对应 `camera.py` 中的映射矩阵：

$$
R_{c\leftarrow b,0} =
\begin{bmatrix}
0 & 1 & 0 \\
0 & 0 & 1 \\
1 & 0 & 0
\end{bmatrix}
$$

### 4.4 四元数约定

四元数采用标量在前：

$$
q = [w, x, y, z]
$$

`q_eb` 表示 body 到 world 的旋转，也就是：

$$
v_e = R_e^b(q_{eb}) v_b
$$

在代码里函数名是 `quat_to_R(q_eb)`，返回的正是这个 $R_e^b$。

---

## 5. 控制层总览

控制层分成两块：

1. 高层任务控制器
2. 低层级联控制器

### 5.1 高层任务控制器有哪些

- `control/ibvs_controller.py`: 经典 IBVS 控制器
- `control/ibvs_so3_controller.py`: SO(3) 几何形式的 IBVS/拦截控制器

### 5.2 低层控制器有哪些

都在 `control/basic_control/`：

- `position_controller.py`: 位置到速度
- `velocity_controller.py`: 速度到姿态+推力
- `attitude_controller.py`: 姿态到角速度+推力
- `rate_controller.py`: 角速度到力矩+推力
- `basic_controller.py`: 把上面几层串起来，再做电机分配

一句话概括：

$$
位置 \to 速度 \to 姿态 \to 角速度 \to 力矩/推力 \to 电机电流
$$

---

## 6. BasicController：整个项目的低层骨架

这是你最应该精读的控制文件之一。

## 6.1 BasicController 的职责

`BasicController` 不自己发明控制规律，它只做两件事：

1. 根据 setpoint 类型，决定要走哪条级联链路。
2. 最终把目标变成 `MotorCommand`。

它支持四种模式：

- `POSITION`
- `VELOCITY`
- `ATT_THRUST`
- `RATE_THRUST`

## 6.2 级联链路

### 位置环

在 `position_controller.py` 中：

$$
v_{sp} = K_p^p (p_{sp} - p)
$$

然后再做范数饱和：

$$
v_{sp} \leftarrow \mathrm{clamp\_norm}(v_{sp}, v_{max})
$$

这是一个非常简单的 P 位置环。

### 速度环

在 `velocity_controller.py` 中：

$$
a_{sp} = K_p^v (v_{sp} - v)
$$

再构造期望推力向量：

$$
f_e^{des} = m (a_{sp} - [0,0,g]^T)
$$

总推力大小：

$$
T_{sp} = \|f_e^{des}\|
$$

由于四旋翼推力沿机体 $-z_b$ 方向施加，所以期望机体 $z$ 轴方向取为：

$$
z_{b,des}^e = -\frac{f_e^{des}}{\|f_e^{des}\|}
$$

再结合期望偏航，构造 $x_b, y_b, z_b$ 三轴，得到期望姿态矩阵 $R_{sp}$，最后转为四元数。

这个步骤本质上就是：

“我想要一个姿态，使得机体的推力方向正好能产生所需加速度。”

### 姿态环

在 `attitude_controller.py` 中，姿态误差写成：

$$
e_R = \mathrm{vex}(R_{sp}^T R - R^T R_{sp})
$$

然后给一个比例型角速度命令：

$$
\omega_{sp} = -K_R e_R
$$

再按分量限幅。

这是一种简洁的几何姿态误差写法。

### 角速度环

在 `rate_controller.py` 中：

$$
e_\omega = \omega_{sp} - \omega
$$

积分项：

$$
I_\omega \leftarrow I_\omega + e_\omega \Delta t
$$

积分项做 anti-windup 限幅后，输出目标力矩：

$$
\tau_{sp} = K_p^\omega e_\omega + K_i^\omega I_\omega
$$

总推力则直接传递并限幅：

$$
T_{sp} \leftarrow \mathrm{clip}(T_{sp}, T_{min}, T_{max})
$$

### 控制分配器

`basic_controller.py` 里 `_ControlAllocator` 完成：

$$
[T, \tau_x, \tau_y, \tau_z]^T \to \{f_i\}_{i=1}^4 \to \{\omega_i\}_{i=1}^4 \to \{i_i\}_{i=1}^4
$$

先定义控制输入：

$$
u =
\begin{bmatrix}
T \\
\tau_x \\
\tau_y \\
\tau_z
\end{bmatrix}
$$

对每个旋翼，分配矩阵采用：

$$
A =
\begin{bmatrix}
1 & 1 & 1 & 1 \\
-y_1 & -y_2 & -y_3 & -y_4 \\
x_1 & x_2 & x_3 & x_4 \\
d_1\kappa & d_2\kappa & d_3\kappa & d_4\kappa
\end{bmatrix},
\quad
\kappa = \frac{k_m}{k_\eta}
$$

其中：

- $(x_i, y_i)$ 是旋翼在机体系的平面位置
- $d_i \in \{+1, -1\}$ 表示旋转方向

然后用伪逆求每个旋翼推力：

$$
f_{rot} = A^+ u
$$

再由推力反推转速：

$$
\omega_i = \sqrt{\frac{f_i}{k_\eta}}
$$

再由转速反推电流命令：

$$
i_i = \frac{\omega_i}{k_{current}}
$$

这部分把“控制理论输出”真正接到了“电机可执行量”。

---

## 7. 经典 IBVS 控制器

`control/ibvs_controller.py` 是一个更“直接”的图像控制器，它没有输出姿态 setpoint，而是直接输出角速度和总推力。

## 7.1 输入是什么

它主要依赖 `Observation` 中的：

- `p_norm`: 目标在归一化像平面中的位置
- `q_eb`: 当前姿态
- `p_r, v_r`: 相对位置和相对速度

## 7.2 图像误差定义

代码中把归一化坐标重新乘回焦距：

$$
e_x = f \cdot p_{norm,x}, \quad e_y = f \cdot p_{norm,y}
$$

这里的 $f$ 是配置里的 `foc`。

## 7.3 角速度控制律

在相机系中：

$$
\omega_c =
\begin{bmatrix}
-k_2 e_y - k_3(\theta - \theta_d) \\
k_5 e_x \\
k_6(\phi - \phi_d)
\end{bmatrix}
$$

其中：

- $\theta$ 是 pitch
- $\phi$ 是 roll
- $\theta_d = \min(\theta_{th}, \arctan2(-e_y, f))$

然后通过固定矩阵把相机角速度映射到机体系：

$$
\omega_b = R_b^c \omega_c
$$

最后做范数限幅。

## 7.4 推力控制律

代码实现的推力是：

$$
T = \frac{m}{\cos\theta} \left(k_4(c_{vy} - k_1 e_y) + g\right)
$$

其中 $c_{vy}$ 是目标相对速度在相机系 y 方向分量。

所以这个控制器可以理解为：

- 横滚/俯仰/偏航主要由图像误差和姿态误差决定
- 推力主要由垂向图像误差和目标相对运动决定

## 7.5 它的特点

- 好处：结构简单，能直接从图像误差产生命令。
- 代价：高层与低层耦合更紧，因为它直接输出 `thrust + body rate`。

---

## 8. IBVS-SO3 控制器

`control/ibvs_so3_controller.py` 是更“几何控制”风格的实现。文件里的注释已经标了论文公式编号，适合对照论文读。

## 8.1 基本思想

它不是直接从图像误差拼一个经验角速度，而是：

1. 从图像得到目标视线方向。
2. 在 SO(3) 上构造期望朝向。
3. 利用相对位置/速度设计期望加速度。
4. 由期望加速度得到期望推力方向和大小。
5. 最后再输出角速度命令和推力命令。

## 8.2 视线方向

若图像归一化坐标为：

$$
p_{norm} = [x_{img}, y_{img}]^T
$$

则机体系下目标方向近似为：

$$
dir_b = \frac{[1, x_{img}, y_{img}]^T}{\|[1, x_{img}, y_{img}]^T\|}
$$

再转到世界系：

$$
n_t = R_e^b dir_b
$$

期望视线方向 `ntd_e` 则取图像中心方向，也就是机体前向轴在世界系中的方向：

$$
n_{td} = R_e^b [1,0,0]^T
$$

## 8.3 视场约束误差

代码里定义：

$$
z_1 = 1 - n_{td}^T n_t
$$

并使用 barrier 系数：

$$
\alpha(z_1) = \frac{z_1}{k_b^2 - z_1^2}
$$

这意味着当视线偏差接近边界时，控制量会明显增大，用来避免目标跑出视场。

## 8.4 第一部分角速度项

$$
\omega_{1,b} = \alpha(z_1) R_b^e (n_{td} \times n_t)
$$

这部分主要负责“把目标重新拉回图像中心方向附近”。

## 8.5 相对运动误差和期望加速度

定义：

$$
z_2 = v_r + k_1 p_r
$$

然后构造期望加速度：

$$
a_d = -k_1 v_r - k_2 z_2 - k_p p_r + \alpha(z_1) \frac{m}{r}(-I + n_t n_t^T)n_{td}
$$

其中：

$$
r = \|p_r\|
$$

你可以把它理解成两部分：

- 前三项：经典的相对位置/速度收敛项
- 最后一项：视线约束相关的修正项

## 8.6 由期望加速度生成期望姿态

先定义期望净受力方向：

$$
n_{fd} = \frac{a_d - g_e}{\|a_d - g_e\|}
$$

当前真实推力方向：

$$
n_f = R_e^b [0,0,-1]^T
$$

然后根据 $n_f$ 和 $n_{fd}$ 的夹角，用 Rodrigues 公式构造一个倾转旋转 $R_{tilt}$，再得到期望姿态：

$$
R_d = R_{tilt} R_e^b
$$

## 8.7 第二部分角速度项

姿态误差项：

$$
E = R_d^T R - R^T R_d
$$

$$
\omega_{2,b} = -\mathrm{vex}(E)
$$

最终角速度命令：

$$
\omega_{cmd} = \mathrm{sat}(\omega_1 + \omega_2, \omega_{max})
$$

## 8.8 推力命令

沿当前推力方向投影：

$$
T = n_f^T m(a_d - g_e)
$$

再限幅到 `[0, thrust_max]`。

## 8.9 它比经典 IBVS 多了什么

- 显式使用 SO(3) 姿态几何误差。
- 显式处理 LOS 约束和 barrier 项。
- 通过期望加速度先构造期望姿态，再生成角速度命令。

所以它更像“视觉引导 + 几何姿态控制”的组合。

## 8.10 这个文件里两个很值得注意的工程处理

### 目标丢失重捕获

当 `obs.has_target=False` 或 `p_norm=None` 时，它不会直接失效，而是：

- 如果仍有相对几何信息，就按目标相对方向做重捕获控制。
- 如果目标在机体后方，就优先自旋搜寻。

### FOV guard

当目标逼近图像边缘时，控制器会临时切换到“优先回中”的保护逻辑。

这两个处理非常工程化，不是纯论文推导，但对仿真稳定性很重要。

---

## 9. 动力学模型

动力学的主体在 `models/rigid_body.py`。

## 9.1 状态定义

刚体状态就是 `UAVState`：

$$
x = (p_e, v_e, q_{eb}, \omega_b)
$$

## 9.2 角运动

代码实现：

$$
I \dot{\omega} + \omega \times (I\omega) = \tau
$$

即：

$$
\dot{\omega} = I^{-1}(\tau - \omega \times I\omega)
$$

然后用显式一步积分：

$$
\omega_{k+1} = \omega_k + \dot{\omega} \Delta t
$$

再做角速度饱和。

## 9.3 平动

先把速度转到机体系，构造简单二次阻力：

$$
f_{drag,b} =
\begin{bmatrix}
-c_{Dx} v_x |v_x| \\
-c_{Dy} v_y |v_y| \\
-c_{Dz} v_z |v_z|
\end{bmatrix}
$$

总机体系受力：

$$
f_{total,b} = f_b + f_{drag,b}
$$

世界系加速度：

$$
a_e = g_e + \frac{R_e^b f_{total,b}}{m}
$$

然后使用 semi-implicit Euler：

$$
v_{k+1} = v_k + a_e \Delta t
$$

$$
p_{k+1} = p_k + v_{k+1} \Delta t
$$

## 9.4 姿态积分

四元数更新在 `utils/math3d.py` 的 `integrate_quat_body_rate` 中，使用指数映射：

$$
q_{k+1} = q_k \otimes \delta q(\omega_b \Delta t)
$$

这是比直接欧拉积分快一些且更稳的方式。

---

## 10. 电机与旋翼模型

在 `models/motors.py` 中。

## 10.1 输入输出

输入：

```text
4个电机电流命令 i_cmd
```

输出：

```text
omega      旋翼转速
thrusts    各旋翼推力
force_b    机体系总力
torque_b   机体系总力矩
```

## 10.2 一阶电机动态

电流命令先变成期望转速：

$$
\omega_i^{des} = k_{current} i_i
$$

再经过一阶惯性：

$$
\dot{\omega}_i = \frac{\omega_i^{des} - \omega_i}{\tau_m}
$$

## 10.3 推力模型

每个旋翼推力：

$$
f_i = k_\eta \omega_i^2
$$

因为推力沿 $-z_b$ 方向，所以总力：

$$
f_b = [0, 0, -\sum_i f_i]^T
$$

## 10.4 力矩模型

由旋翼安装位置产生的力臂力矩：

$$
\tau_{arm} = \sum_i r_i \times [0,0,-f_i]^T
$$

偏航反扭矩：

$$
\tau_z^{react} = \sum_i d_i k_m \omega_i^2
$$

最后：

$$
\tau_b = \tau_{arm} + [0,0,\tau_z^{react}]^T
$$

这使得电机模型成为“控制器”和“刚体动力学”之间的真实执行环节。

---

## 11. 目标模型

在 `models/target.py` 中，`TargetPointMass` 非常简单：

- 如果没有设加速度，就是匀速目标。
- 如果设置 `accel_e`，就是常加速度目标。

公式就是：

$$
v_{k+1} = v_k + a \Delta t
$$

$$
p_{k+1} = p_k + v_{k+1} \Delta t
$$

它的作用主要是给拦截问题提供一个可控、可复现实验对象。

---

## 12. 相机模型与观测层

## 12.1 相机模型

在 `sensors/camera.py` 中，相机做了三件事：

1. 世界系目标相对位置转到机体系
2. 机体系再转到相机系
3. 做针孔投影和 FOV 判断

### 投影公式

相机系下目标点为：

$$
p_c = [x, y, z]^T
$$

归一化像平面坐标：

$$
p_{norm} = [x/z, y/z]^T
$$

像素坐标：

$$
u = f_x x/z + c_x
$$

$$
v = f_y y/z + c_y
$$

当 $z \le 0$ 时，目标在相机后方，直接视为无效。

### 一个很重要的工程细节

当目标虽然超出图像边界，但仍在相机前方时，代码会：

- `valid=False`
- 但保留 `p_norm`

这意味着控制器还能根据“目标大致在什么方向”来回拉目标，而不是彻底失明。

## 12.2 PerfectObserver

在 `observe/perfect.py` 中，当前观测器是“理想观测器”：

- 自身姿态、角速度、速度全都直接取真值。
- 目标图像坐标也直接取相机输出。
- 不加噪声、不加延迟。

它的作用是把“传感器层”和“控制器层”之间隔开。未来若要做更真实的系统，只需要在这里换成有延迟/噪声/估计器的版本。

---

## 13. 调度器：为什么有多频率

`sim/scheduler.py` 里的 `MultiRateScheduler` 很简单，但它是这个仿真框架工程味最强的地方之一。

它固定物理积分步长：

$$
\Delta t = 1 / physics\_hz
$$

再根据频率比计算：

- 每多少步运行一次控制器
- 每多少步运行一次相机
- 每多少步刷新一次可视化

所以项目天然支持：

- 物理 1000 Hz
- 控制 200 Hz
- 相机 30 Hz
- 可视化 24 Hz

这种分离对视觉控制尤其重要，因为现实里相机一定慢于 IMU/内环。

---

## 14. 配置文件怎么映射到代码

## 14.1 `configs/hummingbird.yaml`

这是机体本体参数文件，主要会进入 `RigidBodyParams`：

- 质量、重力、惯量
- 旋翼位置与转向
- 推力系数 `k_eta`
- 反扭矩系数 `k_m`
- 电机时间常数 `tau_m`
- 最大最小转速、电流

你可以把它当作“无人机硬件模型”。

## 14.2 `configs/ibvs_ctrl.yaml`

决定经典 IBVS 仿真：

- 初始无人机和目标状态
- 相机参数
- 经典 IBVS 控制器增益
- 调度频率
- 终止条件

## 14.3 `configs/pos_ctrl.yaml`

主要用于验证 `basic_control` 这条低层级联链能不能独立工作。

它本质上是一个 PX4-like 位置跟踪 demo。

## 14.4 一个需要特别注意的现状

`scripts/ibvs_so3_ctrl_sim.py` 里会按 `IBVSSO3ControllerParams` 的字段过滤 YAML 参数；但当前 `configs/ibvs_so3_ctrl.yaml` 中的 `controller` 配置字段像：

- `k_yaw`
- `k_pitch`
- `k_roll`
- `v_fwd_des`
- `k_vfwd`
- `thrust_bias`

并不在 `IBVSSO3ControllerParams` 里。

结果就是：这些字段当前会被静默忽略。

这不是你理解架构的障碍，但你调参数时必须知道这件事，否则会误以为控制器没反应。

---

## 15. 三个最关键的主循环分别在做什么

## 15.1 `scripts/ibvs_ctrl_sim.py`

主线：

```text
Camera -> PerfectObserver -> IBVSController -> BasicController.step_from_command
       -> Motors -> RigidBody6DoF
```

这个脚本最适合拿来理解“高层视觉控制 + 低层执行链”的完整闭环。

## 15.2 `scripts/ibvs_so3_ctrl_sim.py`

主线和上面一样，但高层换成了 `IBVSSO3Controller`。

它最适合拿来对照论文看公式落地。

## 15.3 `scripts/pos_ctrl_sim.py`

这个脚本没有真正使用视觉闭环，而是把轨迹点直接构造成 `PositionSetpoint`，走：

```text
PositionSetpoint -> BasicController.step -> Motors -> RigidBody6DoF
```

所以它最适合理解 `basic_control/` 这套级联本身。

---

## 16. 如果你只想快速学懂控制器和模型，应该怎么读

### 第一阶段：建立系统全貌

读：

1. `scripts/ibvs_ctrl_sim.py`
2. `models/state.py`
3. `sim/scheduler.py`

目标：

- 明白主循环顺序
- 明白每层传什么对象
- 明白多频率怎么实现

### 第二阶段：吃透低层执行链

读：

1. `control/basic_control/basic_controller.py`
2. `control/basic_control/rate_controller.py`
3. `models/motors.py`
4. `models/rigid_body.py`

目标：

- 明白 `ControlCommand` 怎么变成电机命令
- 明白电机命令怎么变成总力/总力矩
- 明白总力/总力矩怎么推动状态更新

### 第三阶段：吃透视觉控制

读：

1. `sensors/camera.py`
2. `observe/perfect.py`
3. `control/ibvs_controller.py`
4. `control/ibvs_so3_controller.py`

目标：

- 明白图像点是怎么来的
- 明白 `Observation` 的相对量定义
- 明白高层视觉控制律到底在控制什么

### 第四阶段：回到配置调参

读：

1. `configs/hummingbird.yaml`
2. `configs/ibvs_ctrl.yaml`
3. `configs/ibvs_so3_ctrl.yaml`
4. `configs/pos_ctrl.yaml`

目标：

- 把每个参数和代码里的公式对应起来
- 知道哪些参数真的生效，哪些当前没有接进去

---

## 17. 你可以把这个项目记成一张脑图

```text
状态协议层
  UAVState / TargetState / Observation / ControlCommand / ForceSetpoint / MotorCommand

高层任务控制
  IBVSController
  IBVSSO3Controller

低层执行控制
  PositionController
  VelocityController
  AttitudeController
  RateController
  ControlAllocator

物理执行层
  Motors
  RigidBody6DoF
  TargetPointMass

感知层
  PinholeCamera
  PerfectObserver

系统层
  MultiRateScheduler
  Simulator
  Logger / Metrics / Monitor
```

---

## 18. 这个项目当前最重要的 5 个结论

1. 这是一个“高层任务控制 + 低层 PX4-like 执行 + 刚体动力学 + 相机观测”的模块化仿真框架。
2. 真正稳定的协议层是 `models/state.py` 和 `setpoints.py`，这是全项目的接口中心。
3. 当前 IBVS 控制器仍输出 `ControlCommand`，而不是 setpoint；`basic_control` 体现的是未来统一架构。
4. 动力学主线是：`MotorCommand -> Motors -> force/torque -> RigidBody6DoF -> UAVState`。
5. 视觉主线是：`TargetState -> PinholeCamera -> CameraMeasurement -> PerfectObserver -> Observation -> Controller`。

---

## 19. 最后给你的实用建议

如果你现在就想最快上手，不要一开始试图把所有公式都记住。先抓住下面三条：

1. `Observation` 里到底有哪些量，方向和符号是什么。
2. `BasicController` 怎么把高层命令一路变成电机电流。
3. `RigidBody6DoF` 和 `Motors` 怎么把命令变成真实运动。

只要这三条吃透，再回头看 IBVS 和 SO(3) 公式，会容易非常多。