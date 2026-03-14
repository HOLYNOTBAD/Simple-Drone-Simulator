```markdown
# UAV 仿真器相机模型优化建议

本文总结了当前 `PinholeCamera` 相机模型在无人机仿真器中的优化方向。总体原则是：

> 对于控制仿真（IBVS / 目标跟踪 / RL），**针孔投影模型本身已经足够准确**，真正需要优化的是 **测量模型的真实性（噪声、延迟、丢帧等）**。

---

# 1 当前模型结构

当前相机模型实现流程：

```

Target / UAV state (world frame)
↓
relative position (world)
↓
body frame
↓
camera frame
↓
pinhole projection
↓
normalized image plane
↓
pixel coordinate

```

核心数学模型：

```

p_rel_e = p_target - p_uav

p_rel_b = R(q)^T p_rel_e

p_rel_c = R_cb p_rel_b

x_n = X_c / Z_c
y_n = Y_c / Z_c

u = fx * x_n + cx
v = fy * y_n + cy

```

其中：

```

R_cb : body → camera rotation
fx fy : focal length
cx cy : principal point

```

这是标准 **pinhole camera model**。

---

# 2 针孔模型本身是否需要优化

结论：

```

不需要替换针孔模型

```

原因：

- 数学严格
- 计算极快
- 可微（适合控制与RL）
- 是机器人仿真中最常见的模型

因此：

```

projection model ≈ 已经最优

```

仿真误差主要来自：

```

measurement noise
detection failure
sensor latency

```

而不是投影模型本身。

---

# 3 针孔模型可以改进的细节

虽然核心模型正确，但有三个建议优化。

---

## 3.1 使用 FOV 推导内参（推荐）

当前写死：

```

fx = 320
fy = 320
cx = 320
cy = 240

```

建议改为 **由视场角计算**：

```

fx = width  / (2 * tan(FOVx / 2))
fy = height / (2 * tan(FOVy / 2))

```

示例：

```

FOVx = 90°
width = 640

fx ≈ 320

```

优势：

- 参数更符合真实相机
- 修改 FOV 更方便
- 更符合视觉系统配置方式

---

## 3.2 提高 Z 方向数值稳定性

当前代码：

```

if z <= 1e-6:

```

建议：

```

if z <= 0:
target behind camera

```

并在计算归一化时：

```

z = max(z, 1e-6)

```

作用：

```

避免 x/z 数值爆炸

```

特别是在 RL 训练中很重要。

---

## 3.3 返回 bearing vector（推荐）

很多视觉控制算法使用 **方向向量**：

```

bearing = p_rel_c / ||p_rel_c||

```

可在测量中增加：

```

bearing_c

```

优势：

- 对深度不敏感
- IBVS / RL 控制更稳定

---

# 4 强烈推荐增加的测量模型

这些比修改针孔模型更重要。

---

# 4.1 像素噪声

真实视觉系统一定存在像素误差。

简单实现：

```

u += N(0, σ_px)
v += N(0, σ_px)

```

建议：

```

σ_px ≈ 0.5 – 2 px

```

更真实模型：

```

σ = σ0 + k * range

```

即：

```

远距离目标噪声更大

```

---

# 4.2 目标检测丢失 (dropout)

真实视觉系统不会每帧检测到目标。

实现：

```

if random() > p_detect:
measurement invalid

```

典型参数：

```

p_detect ≈ 0.9 – 0.98

```

也可以与距离相关：

```

p_detect = exp(-range / r0)

```

作用：

```

提升 RL / 控制算法鲁棒性

```

---

# 4.3 相机测量延迟

真实系统包含：

```

exposure
image processing
communication

```

通常延迟：

```

20 ms – 80 ms

```

仿真器可实现：

```

measurement buffer

```

示例：

```

queue.append(measurement)

return measurement(t - delay)

```

作用：

```

更接近真实无人机控制链路

```

---

# 5 相机外参优化

当前外参仅支持：

```

mount_pitch_deg

```

建议扩展为完整外参：

```

R_cb
t_cb

```

完整模型：

```

p_c = R_cb p_b + t_cb

```

优势：

```

支持机头相机
支持机腹相机
支持偏移安装

```

---

# 6 可选视觉真实性增强

如果追求更高保真视觉仿真，可考虑以下模型。

---

## 6.1 畸变模型

真实相机存在：

```

radial distortion

```

公式：

```

r² = x² + y²

x_d = x (1 + k1 r² + k2 r⁴)
y_d = y (1 + k1 r² + k2 r⁴)

```

典型参数：

```

k1 ≈ -0.1 ~ -0.3

```

但控制仿真通常不需要。

---

## 6.2 广角相机模型

如果模拟 drone racing camera：

```

FOV ≈ 120° – 170°

```

pinhole 会产生较大误差。

可使用：

```

fisheye model
equidistant model

```

但大多数控制研究仍使用 pinhole。

---

## 6.3 Rolling Shutter

真实相机：

```

逐行曝光

```

高速运动时产生：

```

rolling distortion

```

但通常只在图像级仿真中使用。

---

# 7 仿真性能优化

如果仿真规模较大（RL训练），建议优化计算效率。

---

## 7.1 预计算外参矩阵

当前每次调用：

```

rot_y()

```

建议初始化时计算：

```

R_mount
R_cb

```

测量时直接：

```

p_rel_c = R_cb p_rel_b

```

减少矩阵运算。

---

## 7.2 支持批量计算

若存在多个目标：

```

P_rel_c = R_cb @ P_rel_b.T

```

可减少 Python 循环开销。

---

# 8 推荐优化优先级

对于无人机视觉控制仿真器，推荐实现顺序：

### 第一优先级（最重要）

```

pixel noise
detection dropout
measurement delay

```

---

### 第二优先级

```

完整相机外参 (R_cb, t_cb)

```

---

### 第三优先级

```

FOV-based intrinsics
bearing vector

```

---

### 第四优先级（可选）

```

distortion model
fisheye camera
rolling shutter

```

---

# 9 总结

当前针孔模型：

```

工程质量：高
数学正确性：高
适用于控制仿真

```

核心结论：

```

针孔投影模型已经足够好

```

仿真真实性主要取决于：

```

measurement noise
detection failure
sensor latency

```

在加入这些机制后，相机模型即可达到 **论文级无人机仿真器传感器水平**。
```
