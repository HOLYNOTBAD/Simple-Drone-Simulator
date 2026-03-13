from __future__ import annotations

import argparse

import numpy as np

try:
    import yaml
except ImportError as e:
    raise SystemExit("Please install PyYAML: pip install pyyaml") from e

from control.basic_control.basic_controller import BasicController
from control.basic_control.setpoints import RateThrustSetpoint
from models.motors import Motors
from models.rigid_body import RigidBody6DoF, RigidBodyParams
from models.state import TargetState, UAVState
from observe.perfect import PerfectObserver
from sim.scheduler import MultiRateScheduler, RateConfig
from sim.simulator import Simulator, TerminationConfig
from utils.config import resolve_script_config
from visualization.monitor import Monitor, MonitorConfig


def _load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _omega_sp(t_now: float, out: np.ndarray) -> np.ndarray:
    c = np.cos(2.0 * np.pi * t_now)
    out[0] = c
    out[1] = c
    out[2] = 1.0
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None)
    args = ap.parse_args()

    config_path = resolve_script_config(__file__, args.config)
    cfg = _load_cfg(config_path)
    seed = int(cfg.get("seed", 0))
    np.random.seed(seed)

    rates_cfg = cfg["rates"]
    rates = RateConfig(
        physics_hz=float(rates_cfg["physics_hz"]),
        control_hz=float(rates_cfg["control_hz"]),
        camera_hz=float(rates_cfg["camera_hz"]),
        visualization_hz=float(rates_cfg.get("visualization_hz", 10.0)),
    )
    sch = MultiRateScheduler(rates)

    viz_cfg = cfg.get("visualization", {})
    mon_cfg = cfg.get("monitor", {})
    sim_viz = Simulator(
        scheduler=sch,
        enable=bool(viz_cfg.get("enable", True)),
        enable_realtime_animation=bool(viz_cfg.get("enable_realtime_animation", True)),
        enable_offline_animation=bool(viz_cfg.get("enable_offline_animation", False)),
        save_cache=bool(viz_cfg.get("save_cache", False)),
        realtime=bool(viz_cfg.get("realtime", True)),
        enable_fov=False,
        trail_len=int(viz_cfg.get("trail_len", 1000)),
        auto_axis=bool(viz_cfg.get("auto_axis", False)),
        ned_axes=bool(viz_cfg.get("ned_axes", True)),
        map_center=tuple(viz_cfg.get("map_center", [0.0, 0.0, 0.0])),
        map_size=tuple(viz_cfg.get("map_size", [20.0, 20.0, 12.0])),
        colors=viz_cfg.get("colors", {}),
        uav_visual_scale=float(viz_cfg.get("uav_visual_scale", 1.0)),
        target_marker_size=float(viz_cfg.get("target_marker_size", 6.0)),
        fov_target_diameter_m=float(viz_cfg.get("fov_target_diameter_m", 1.0)),
    )

    u0 = cfg["uav0"]
    uav = UAVState(
        t=float(u0["t"]),
        p_e=np.array(u0["p_e"], dtype=float),
        v_e=np.array(u0["v_e"], dtype=float),
        q_eb=np.array(u0["q_eb"], dtype=float),
        w_b=np.array(u0["w_b"], dtype=float),
    )
    tgt = TargetState(t=uav.t, p_e=np.zeros(3, dtype=float), v_e=np.zeros(3, dtype=float))

    rb_params = RigidBodyParams.from_yaml(cfg["rigid_body"]["params_yaml"])
    uav_model = RigidBody6DoF(rb_params)
    motors = Motors(rb_params)
    basic_ctrl = BasicController(rb_params)
    observer = PerfectObserver()

    thrust_sp = rb_params.mass * rb_params.g

    basic_ctrl.reset()
    motors.reset()

    t_final = 5.0
    term = TerminationConfig(t_final=t_final, hit_radius=0.0)
    sim_viz.set_t_final(term.t_final)

    monitor = Monitor(
        scheduler=sch,
        cfg=MonitorConfig(
            enable=bool(mon_cfg.get("enable", True)),
            realtime=bool(mon_cfg.get("realtime", True)),
            max_points=mon_cfg.get("max_points", None),
            step_stride=int(mon_cfg.get("step_stride", 1)),
            title=str(mon_cfg.get("title", "Rate Controller Monitor")),
            x_min=float(mon_cfg.get("x_min", 0.0)),
            x_max=float(mon_cfg.get("x_max", t_final)),
        ),
    )

    steps = int(np.ceil(term.t_final / sch.dt))
    omega_sp_b = np.zeros(3, dtype=float)
    rate_sp = RateThrustSetpoint(omega_sp_b=omega_sp_b, thrust_sp=thrust_sp)
    basic_ctrl.update_setpoint(rate_sp)

    for k in range(steps):
        _omega_sp(uav.t, omega_sp_b)

        obs = observer.make_observation(t_now=uav.t, uav=uav, cam=None, tgt=None)
        motor_cmd = basic_ctrl.step(uav_state=uav, obs=obs, t_now=uav.t, dt=sch.dt)
        motor_out = motors.step(motor_cmd.motor_current_cmd, sch.dt)
        actual_thrust = float(-motor_out.force_b[2])

        uav = uav_model.step(uav, force_b=motor_out.force_b, torque_b=motor_out.torque_b, dt=sch.dt)
        sim_viz.update(step=k, uav=uav, tgt=tgt, cam=None, has_target=False)

        monitor.push(name="cmd", color="tab:red", data=omega_sp_b[0], t=uav.t, group="omega_x", step=k)
        monitor.push(name="actual", color="tab:blue", data=uav.w_b[0], t=uav.t, group="omega_x", step=k)
        monitor.push(name="cmd", color="tab:red", data=omega_sp_b[1], t=uav.t, group="omega_y", step=k)
        monitor.push(name="actual", color="tab:blue", data=uav.w_b[1], t=uav.t, group="omega_y", step=k)
        monitor.push(name="cmd", color="tab:red", data=omega_sp_b[2], t=uav.t, group="omega_z", step=k)
        monitor.push(name="actual", color="tab:blue", data=uav.w_b[2], t=uav.t, group="omega_z", step=k)
        monitor.push(name="cmd", color="tab:red", data=thrust_sp, t=uav.t, group="thrust", step=k)
        monitor.push(name="actual", color="tab:blue", data=actual_thrust, t=uav.t, group="thrust", step=k)
        monitor.update(step=k)

    if sim_viz.enable:
        sim_viz.close(block=True)
        monitor.close(block=False)
    else:
        monitor.close(block=True)


if __name__ == "__main__":
    main()
