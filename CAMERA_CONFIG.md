# 相机参数配置说明

本文档说明本项目 `camera:` 配置段当前支持的参数、默认值、作用范围和推荐用法。

相机配置由 [sensors/camera.py](sensors/camera.py) 中的 `build_camera_from_config` 统一解析，当前已在以下脚本中接入：

- [scripts/ibvs_ctrl_sim.py](scripts/ibvs_ctrl_sim.py)
- [scripts/ibvs_so3_ctrl_sim.py](scripts/ibvs_so3_ctrl_sim.py)
- [scripts/pos_ctrl_sim.py](scripts/pos_ctrl_sim.py)

## 1. 配置目标

当前相机模型支持以下能力：

- 针孔投影
- 由焦距或视场角推导内参
- 默认机体系到相机系的轴映射
- 完整外参旋转 `R_cb`
- 相机安装偏移 `t_cb_b`
- 像素噪声
- 目标丢检
- 测量延迟

## 2. 坐标约定

项目中的相关坐标系约定如下：

- 世界系 `{e}`: NED
- 机体系 `{b}`: FRD
- 相机系 `{c}`:
  - `x_c`: 右
  - `y_c`: 下
  - `z_c`: 前方光轴

默认情况下，相机采用项目已有的 FRD 到相机系映射：

- `z_c = x_b`
- `x_c = y_b`
- `y_c = z_b`

## 3. 参数总表

`camera:` 段支持以下字段。

| 参数名 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `width` | int | `640` | 图像宽度，单位 px |
| `height` | int | `480` | 图像高度，单位 px |
| `fx` | float or null | `320.0` | x 方向焦距，单位 px |
| `fy` | float or null | `320.0` | y 方向焦距，单位 px |
| `cx` | float or null | 图像中心 | 主点 x 坐标，单位 px |
| `cy` | float or null | 图像中心 | 主点 y 坐标，单位 px |
| `fov_x_deg` | float or null | `null` | 水平视场角，单位 deg |
| `fov_y_deg` | float or null | `null` | 垂直视场角，单位 deg |
| `use_default_frd_to_camera` | bool | `true` | 是否使用默认 FRD 到相机系映射 |
| `mount_pitch_deg` | float | `20.0` | 相机俯仰安装角，正值表示相机朝下俯视 |
| `R_cb` | 3x3 array or null | `null` | 完整 body -> camera 旋转矩阵 |
| `t_cb_b` | 3 array | `[0, 0, 0]` | 相机原点在机体系下的位置偏移，单位 m |
| `noise_std_px` | float | `0.0` | 固定像素噪声标准差 |
| `noise_std_range_scale_px` | float | `0.0` | 随距离增长的像素噪声系数 |
| `detection_prob` | float | `1.0` | 基础检测概率 |
| `detection_range_decay_m` | float | `0.0` | 检测概率的距离衰减长度 |
| `delay_s` | float | `0.0` | 测量延迟，单位 s |
| `seed` | int or null | `null` | 相机独立随机种子 |

## 4. 参数解析规则

### 4.1 内参优先级

内参解析规则如下：

1. 如果显式给了 `fx` 或 `fy`，优先使用显式值。
2. 如果对应焦距为 `null`，但给了 `fov_x_deg` 或 `fov_y_deg`，则按视场角推导焦距。
3. 如果 `fx` 和 `fy` 都未提供，则默认退化为半幅宽的典型针孔值。
4. 如果 `cx` 或 `cy` 为 `null`，则自动取图像中心。

FOV 推导公式为：

$$
f_x = \frac{width}{2 \tan(FOV_x / 2)}, \quad
f_y = \frac{height}{2 \tan(FOV_y / 2)}
$$

说明：

- `fov_x_deg` 和 `fov_y_deg` 只有在对应 `fx` 或 `fy` 为 `null` 时才会生效。
- 如果只给 `fov_x_deg`，则会先求出 `fx`，再令 `fy = fx`。

### 4.2 外参优先级

外参解析规则如下：

1. 若提供 `R_cb`，则直接使用该旋转矩阵。
2. 若 `R_cb` 为 `null`，则使用默认轴映射和 `mount_pitch_deg` 组合得到旋转。
3. `t_cb_b` 始终表示相机在机体系中的安装偏移。

测量时的几何关系为：

$$
p_c = R_{cb}(p_b - t_{cb}^b)
$$

其中：

- `p_b`: 目标相对位置在机体系下的表示
- `t_cb_b`: 相机位置在机体系下的表示

### 4.3 随机项规则

如果 `seed` 为 `null`：

- 相机噪声和丢检复用全局 `numpy.random` 状态
- 脚本里设置的全局随机种子仍然会影响相机结果

如果 `seed` 有值：

- 相机使用独立随机源
- 相机噪声和主程序其他随机项解耦

## 5. 测量模型说明

每次相机测量会输出 `CameraMeasurement`，其中最常用字段如下：

| 字段名 | 含义 |
| --- | --- |
| `p_cam` | 目标在相机系中的相对位置 |
| `bearing_c` | 相机系下的单位视线方向 |
| `p_norm` | 归一化像平面坐标 |
| `uv_px` | 像素坐标 |
| `range_m` | 目标视线距离 |
| `valid` | 当前是否在视场内且检测成功 |

补充说明：

- 如果目标在相机后方，即 `z <= 0`，测量直接记为无效。
- 如果目标出了图像边界，`valid = false`，但仍尽量保留 `p_norm` 和 `uv_px`，方便控制器朝目标方向回拉。
- 如果发生丢检，当前实现返回 `valid = false`，并清空 `p_norm` 与 `uv_px`。
- 如果配置了 `delay_s > 0`，则 `measure(...)` 在延迟未满足时可能返回空。

## 6. 噪声与丢检模型

### 6.1 像素噪声

像素噪声标准差为：

$$
\sigma_{px} = noise\_std\_px + noise\_std\_range\_scale\_px \cdot range
$$

然后分别对 `u` 和 `v` 注入高斯噪声。

建议取值：

- 轻微噪声: `noise_std_px: 0.5`
- 中等噪声: `noise_std_px: 1.0`
- 距离相关噪声: `noise_std_range_scale_px: 0.01`

### 6.2 丢检概率

基础检测概率由 `detection_prob` 控制。

如果 `detection_range_decay_m > 0`，则最终检测概率为：

$$
p = detection\_prob \cdot e^{-range / detection\_range\_decay\_m}
$$

建议：

- 理想情况: `detection_prob: 1.0`
- 较真实仿真: `detection_prob: 0.95`
- 距离衰减明显时再配置 `detection_range_decay_m`

## 7. 延迟模型

`delay_s` 用于模拟曝光、处理、通信等链路延迟。

行为如下：

- 每次调用 `measure(...)` 时，先生成当前时刻测量
- 测量进入内部队列
- 只有当某条历史测量满足 `t_meas <= t_now - delay_s` 时，该历史测量才会被输出

因此：

- 刚开始仿真时，前几次相机调用可能返回空
- 控制与观测链路必须允许 `cam is None`

建议取值：

- 无延迟: `0.0`
- 轻度延迟: `0.02`
- 更贴近真实链路: `0.05` 到 `0.08`

## 8. 推荐配置示例

### 8.1 保持原有行为

```yaml
camera:
  fx: 320.0
  fy: 320.0
  cx: 320.0
  cy: 240.0
  width: 640
  height: 480
  use_default_frd_to_camera: true
  mount_pitch_deg: 0.0
  R_cb: null
  t_cb_b: [0.0, 0.0, 0.0]
  noise_std_px: 0.0
  noise_std_range_scale_px: 0.0
  detection_prob: 1.0
  detection_range_decay_m: 0.0
  delay_s: 0.0
```

### 8.2 使用 FOV 推导内参

```yaml
camera:
  fx: null
  fy: null
  fov_x_deg: 90.0
  fov_y_deg: 73.74
  cx: null
  cy: null
  width: 640
  height: 480
  mount_pitch_deg: 10.0
  noise_std_px: 0.5
  detection_prob: 0.98
  delay_s: 0.02
```

### 8.3 指定完整外参

```yaml
camera:
  fx: 320.0
  fy: 320.0
  cx: 320.0
  cy: 240.0
  width: 640
  height: 480
  R_cb:
    - [0.0, 1.0, 0.0]
    - [0.0, 0.0, 1.0]
    - [1.0, 0.0, 0.0]
  t_cb_b: [0.1, 0.0, 0.02]
  noise_std_px: 1.0
  noise_std_range_scale_px: 0.01
  detection_prob: 0.95
  detection_range_decay_m: 120.0
  delay_s: 0.05
  seed: 42
```

## 9. 常见注意事项

### 9.1 FOV 和 fx/fy 不要同时写死成冲突值

如果你同时给了：

- `fx: 300`
- `fov_x_deg: 90`

那么当前实现会优先使用 `fx: 300`，不会再按 `fov_x_deg` 重算。

### 9.2 延迟开启后初始空测量是正常现象

如果你看到仿真刚开始时：

- `cam is None`
- `has_target = false`

这不一定是错误，也可能只是延迟队列还没吐出首帧。

### 9.3 丢检与出框都可能导致 `valid = false`

两者区别在于：

- 出框时通常还能保留 `p_norm`
- 丢检时当前实现会清空 `p_norm`

### 9.4 使用独立相机种子会改变重复性来源

如果要让整条仿真链统一受主程序 `seed` 控制，保持：

```yaml
seed: null
```

如果要让相机噪声在不同试验中独立可复现，再单独指定相机 `seed`。

## 10. 当前配置入口位置

项目内已有相机配置入口文件包括：

- [configs/ibvs_ctrl.yaml](configs/ibvs_ctrl.yaml)
- [configs/ibvs_so3_ctrl.yaml](configs/ibvs_so3_ctrl.yaml)
- [configs/pos_ctrl.yaml](configs/pos_ctrl.yaml)
- [configs/multi_pos_ctrl.yaml](configs/multi_pos_ctrl.yaml)

如果后续新增脚本，建议也统一通过 [sensors/camera.py](sensors/camera.py) 中的 `build_camera_from_config` 创建相机，避免重复解析逻辑。