# scripts/ibvs_ctrl_sim.py
from __future__ import annotations

from pathlib import Path
from dataclasses import fields
import argparse
import time
import numpy as np

try:
    import yaml
except ImportError as e:
    raise SystemExit("Please install PyYAML: pip install pyyaml") from e

from models.state import ControlCommand, UAVState, TargetState
from models.rigid_body import RigidBodyParams, RigidBody6DoF
from models.motors import Motors
from models.target import TargetParams, TargetPointMass
from sensors.camera import build_camera_from_config
from sim.scheduler import RateConfig, MultiRateScheduler
from sim.simulator import TerminationConfig, Simulator

from observe.perfect import PerfectObserver
from control.basic_control.basic_controller import BasicController
from control.ibvs_controller import IBVSController, IBVSControllerParams
from utils.config import resolve_script_config
from utils.log import NPZLogger
from utils.metrics import Metrics, MetricsConfig
from visualization.monitor import Monitor, MonitorConfig


def _load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():

    
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None)
    args = ap.parse_args()

    config_path = resolve_script_config(__file__, args.config)
    cfg = _load_cfg(config_path)
    camera = build_camera_from_config(cfg.get("camera"))

    logger_cfg = cfg.get("logging", {})

    # Seed
    seed = int(logger_cfg.get("seed", 0))
    np.random.seed(seed)

    # Scheduler
    r = cfg["rates"]
    rates = RateConfig(
        physics_hz=float(r["physics_hz"]),
        control_hz=float(r["control_hz"]),
        camera_hz=float(r["camera_hz"]),
        visualization_hz=float(r.get("visualization_hz", 10.0)),
    )
    sch = MultiRateScheduler(rates)
    viz_cfg = cfg.get("visualization", {})
    mon_cfg = cfg.get("monitor", {})
    sim_viz = Simulator(
        scheduler=sch,
        enable=bool(viz_cfg.get("enable", True)),
        t_final=float(cfg.get("termination", {}).get("t_final", 20.0)),
        enable_realtime_animation=bool(viz_cfg.get("enable_realtime_animation", True)),
        enable_offline_animation=bool(viz_cfg.get("enable_offline_animation", False)),
        save_cache=bool(viz_cfg.get("save_cache", False)),
        realtime=bool(viz_cfg.get("realtime", True)),
        enable_fov=bool(viz_cfg.get("enable_fov", True)),
        cam_width=int(camera.K.width),
        cam_height=int(camera.K.height),
        cam_fx=float(camera.K.fx),
        cam_fy=float(camera.K.fy),
        cam_cx=float(camera.K.cx),
        cam_cy=float(camera.K.cy),
        trail_len=int(viz_cfg.get("trail_len", 1000)),
        auto_axis=bool(viz_cfg.get("auto_axis", False)),
        ned_axes=bool(viz_cfg.get("ned_axes", True)),
        map_center=tuple(viz_cfg.get("map_center", [0.0, 0.0, 0.0])),
        map_size=tuple(viz_cfg.get("map_size", [160.0, 120.0, 80.0])),
        colors=viz_cfg.get("colors", {}),
        uav_visual_scale=float(viz_cfg.get("uav_visual_scale", 1.0)),
        target_marker_size=float(viz_cfg.get("target_marker_size", 6.0)),
        fov_target_marker_size=float(viz_cfg.get("fov_target_marker_size", 7.0)),
        fov_target_marker_size_min=float(viz_cfg.get("fov_target_marker_size_min", 4.0)),
        fov_target_marker_size_max=float(viz_cfg.get("fov_target_marker_size_max", 18.0)),
        fov_target_diameter_m=float(viz_cfg.get("fov_target_diameter_m", 1.0)),
    )

    # Initial states
    u0 = cfg["uav0"]
    t0 = cfg["tgt0"]
    uav0 = UAVState(
        t=float(u0["t"]),
        p_e=np.array(u0["p_e"], dtype=float),
        v_e=np.array(u0["v_e"], dtype=float),
        q_eb=np.array(u0["q_eb"], dtype=float),
        w_b=np.array(u0["w_b"], dtype=float),
    )
    tgt0 = TargetState(
        t=float(t0["t"]),
        p_e=np.array(t0["p_e"], dtype=float),
        v_e=np.array(t0["v_e"], dtype=float),
    )

    # Models
    rb = cfg["rigid_body"]
    rb_params = RigidBodyParams.from_yaml(rb["params_yaml"])
    uav_model = RigidBody6DoF(rb_params)
    motors = Motors(rb_params)
    basic_ctrl = BasicController(rb_params)

    tg = cfg.get("target", {})
    accel = tg.get("accel_e", None)
    tgt_model = TargetPointMass(TargetParams(accel_e=None if accel is None else np.array(accel, dtype=float)))

    # Observer 
    observer = PerfectObserver()

    # Controller
    cc = cfg["controller"]
    ctrl_allowed = {f.name for f in fields(IBVSControllerParams)}
    ctrl_kwargs = {k: cc[k] for k in cc if k in ctrl_allowed}
    ctrl_params = IBVSControllerParams(**ctrl_kwargs)
    controller = IBVSController(ctrl_params)
    basic_ctrl.reset()
    motors.reset()

    # Termination / metrics / logger
    term = cfg["termination"]
    term_cfg = TerminationConfig(
        t_final=float(term["t_final"]),
        hit_radius=float(term["hit_radius"]),
    )
    sim_viz.set_t_final(term_cfg.t_final)
    monitor = Monitor(
        scheduler=sch,
        cfg=MonitorConfig(
            enable=bool(mon_cfg.get("enable", True)),
            realtime=bool(mon_cfg.get("realtime", True)),
            max_points=mon_cfg.get("max_points", None),
            step_stride=int(mon_cfg.get("step_stride", 1)),
            title=str(mon_cfg.get("title", "IBVS Controller Monitor")),
            x_min=float(mon_cfg.get("x_min", 0.0)),
            x_max=float(mon_cfg.get("x_max", term_cfg.t_final)),
        ),
    )
    controller.set_monitor(monitor)

    metrics = Metrics(MetricsConfig(hit_radius=term_cfg.hit_radius))
    logging_enabled = bool(logger_cfg.get("enable", True))
    logger = NPZLogger(run_dir=str(logger_cfg.get("run_dir", "runs"))) if logging_enabled else None
    filename = logger_cfg.get("filename", None)

    # --- Run loop (explicit here, using scheduler and modules) ---
    uav = uav0
    tgt = tgt0
    last_cmd = ControlCommand(
        t=uav.t,
        thrust=float(rb_params.mass * rb_params.g),
        omega_cmd_b=np.zeros(3, dtype=float),
    )
    last_i_cmd = np.zeros(rb_params.num_rotors, dtype=float)
    last_omega = np.zeros(rb_params.num_rotors, dtype=float)
    last_cam = camera.measure(uav, tgt, t_meas=uav.t)
    initial_obs = observer.make_observation(t_now=uav.t, uav=uav, cam=last_cam, tgt=tgt)
    nan2 = (np.nan, np.nan)
    termination_reason = None

    sim_viz.update(step=0, uav=uav, tgt=tgt, cam=last_cam, has_target=initial_obs.has_target)

    steps = int(np.ceil(term_cfg.t_final / sch.dt))
    for k in range(steps):
        t_now = uav.t

        # camera
        if sch.should_camera(k):
            last_cam = camera.measure(uav, tgt, t_meas=t_now)

        # observation (perfect self-state)
        obs = observer.make_observation(t_now=t_now, uav=uav, cam=last_cam, tgt=tgt)

        # control
        if sch.should_control(k):
            cmd = controller.compute(obs)
            last_cmd = cmd
        else:
            cmd = last_cmd

        # low-level control + motors + physics
        force_sp, motor_cmd = basic_ctrl.step_from_command(uav, cmd, sch.dt)
        motor_out = motors.step(motor_cmd.motor_current_cmd, sch.dt)

        uav = uav_model.step(uav, force_b=motor_out.force_b, torque_b=motor_out.torque_b, dt=sch.dt)
        tgt = tgt_model.step(tgt, sch.dt)
        last_i_cmd = motor_out.i_cmd
        last_omega = motor_out.omega
        sim_viz.update(step=k, uav=uav, tgt=tgt, cam=last_cam, has_target=obs.has_target)
        actual_thrust = float(-motor_out.force_b[2])
        monitor.push(name="cmd_thrust", color="tab:red", data=force_sp.thrust_sp, t=t_now, group="command", step=k)
        monitor.push(name="actual_thrust", color="tab:blue", data=actual_thrust, t=t_now, group="command", step=k)
        monitor.push(name="cmd_x", color="tab:red", data=cmd.omega_cmd_b[0], t=t_now, group="omega_cmd", step=k)
        monitor.push(name="cmd_y", color="tab:red", data=cmd.omega_cmd_b[1], t=t_now, group="omega_cmd", step=k)
        monitor.push(name="cmd_z", color="tab:red", data=cmd.omega_cmd_b[2], t=t_now, group="omega_cmd", step=k)
        monitor.push(name="actual_x", color="tab:blue", data=uav.w_b[0], t=t_now, group="omega_cmd", step=k)
        monitor.push(name="actual_y", color="tab:cyan", data=uav.w_b[1], t=t_now, group="omega_cmd", step=k)
        monitor.push(name="actual_z", color="tab:green", data=uav.w_b[2], t=t_now, group="omega_cmd", step=k)
        monitor.update(step=k)

        # metrics + logging
        dist = metrics.update(t=t_now, uav_p=uav.p_e, tgt_p=tgt.p_e)

        if logger is not None:
            logger.push("t", t_now)
            logger.push("uav_p", uav.p_e)
            logger.push("uav_v", uav.v_e)
            logger.push("uav_q", uav.q_eb)
            logger.push("uav_w", uav.w_b)
            logger.push("tgt_p", tgt.p_e)
            logger.push("tgt_v", tgt.v_e)
            logger.push("dist", dist)

            if last_cam is None:
                logger.push("cam_valid", False)
                logger.push("cam_p_norm", nan2)
                logger.push("cam_uv", nan2)
            else:
                logger.push("cam_valid", bool(last_cam.valid))
                if last_cam.p_norm is None:
                    logger.push("cam_p_norm", nan2)
                else:
                    logger.push("cam_p_norm", last_cam.p_norm)
                if last_cam.uv_px is None:
                    logger.push("cam_uv", nan2)
                else:
                    logger.push("cam_uv", last_cam.uv_px)

            logger.push("cmd_thrust", cmd.thrust)
            logger.push("cmd_omega", cmd.omega_cmd_b)
            logger.push("force_sp_thrust", force_sp.thrust_sp)
            logger.push("force_sp_tau", force_sp.tau_sp_b)
            logger.push("motor_cmd", motor_cmd.motor_current_cmd)
            logger.push("motor_i_cmd", last_i_cmd)
            logger.push("motor_omega", last_omega)

        if metrics.hit:
            termination_reason = "hit"
            break

    summary = metrics.summary()
    meta = {
        "config": Path(config_path).as_posix(),
        "seed": seed,
        "summary": summary,
        "rates": r,
    }
    save_path = None
    if logger is not None:
        save_path = logger.save(meta=meta, filename=None if filename in (None, "null") else str(filename))

    print("\n=== Simulation Summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")
    if save_path is not None:
        print(f"saved: {save_path}")
    else:
        print("logging disabled: no file saved")
    if sim_viz.enable:
        sim_viz.close(block=True, termination_reason=termination_reason)
        monitor.close(block=False)
    else:
        monitor.close(block=True)


if __name__ == "__main__":
    main()
