# scripts/vpn_acc_ctrl_sim.py
from __future__ import annotations

from pathlib import Path
from dataclasses import fields
import argparse
import numpy as np

try:
    import yaml
except ImportError as e:
    raise SystemExit("Please install PyYAML: pip install pyyaml") from e

from models.state import ControlCommand, UAVState, TargetState
from models.rigid_body import RigidBodyParams, RigidBody6DoF
from models.motors import Motors
from models.target import TargetParams, TargetPointMass
from sensors.camera import CameraIntrinsics, CameraExtrinsics, PinholeCamera
from sim.scheduler import RateConfig, MultiRateScheduler
from sim.simulator import TerminationConfig, Simulator

from observe.perfect import PerfectObserver
from observe.obj_tracker import ObjTracker, ObjTrackerParams
from control.basic_control.basic_controller import BasicController
from control.vpn_acc_controller import VpnAccController, VpnAccControllerParams
from utils.config import resolve_script_config
from utils.log import NPZLogger
from utils.metrics import Metrics, MetricsConfig
from visualization.monitor import Monitor, MonitorConfig


def _load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None)
    args = ap.parse_args()

    config_path = resolve_script_config(__file__, args.config)
    cfg = _load_cfg(config_path)

    logger_cfg = cfg.get("logging", {})
    seed = int(logger_cfg.get("seed", 0))
    np.random.seed(seed)

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
        cam_width=int(cfg["camera"].get("width", 640)),
        cam_height=int(cfg["camera"].get("height", 480)),
        cam_fx=float(cfg["camera"].get("fx", 320.0)),
        cam_fy=float(cfg["camera"].get("fy", 320.0)),
        cam_cx=float(cfg["camera"].get("cx", 320.0)),
        cam_cy=float(cfg["camera"].get("cy", 240.0)),
        trail_len=int(viz_cfg.get("trail_len", 1000)),
        auto_axis=bool(viz_cfg.get("auto_axis", False)),
        ned_axes=bool(viz_cfg.get("ned_axes", True)),
        map_center=tuple(viz_cfg.get("map_center", [0.0, 0.0, 0.0])),
        map_size=tuple(viz_cfg.get("map_size", [160.0, 120.0, 80.0])),
        colors=viz_cfg.get("colors", {}),
        uav_visual_scale=float(viz_cfg.get("uav_visual_scale", 1.0)),
        target_marker_size=float(viz_cfg.get("target_marker_size", 6.0)),
        fov_target_diameter_m=float(viz_cfg.get("fov_target_diameter_m", 1.0)),
    )

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

    rb = cfg["rigid_body"]
    rb_params = RigidBodyParams.from_yaml(rb["params_yaml"])
    uav_model = RigidBody6DoF(rb_params)
    motors = Motors(rb_params)
    basic_ctrl = BasicController(rb_params)

    tg = cfg.get("target", {})
    accel = tg.get("accel_e", None)
    target_diameter_m = float(tg.get("diameter_m", viz_cfg.get("fov_target_diameter_m", 1.0)))
    tgt_model = TargetPointMass(TargetParams(accel_e=None if accel is None else np.array(accel, dtype=float)))

    cam = cfg["camera"]
    K = CameraIntrinsics(
        fx=float(cam["fx"]),
        fy=float(cam["fy"]),
        cx=float(cam["cx"]),
        cy=float(cam["cy"]),
        width=int(cam["width"]),
        height=int(cam["height"]),
    )
    camera = PinholeCamera(
        K,
        CameraExtrinsics(
            mount_pitch_deg=float(cam.get("mount_pitch_deg", 20.0)),
        ),
    )
    observer = PerfectObserver()
    tracker = ObjTracker(
        ObjTrackerParams(
            fx=K.fx,
            fy=K.fy,
            target_width_m=target_diameter_m,
            target_height_m=target_diameter_m,
        )
    )

    cc = cfg["controller"]
    ctrl_allowed = {f.name for f in fields(VpnAccControllerParams)}
    ctrl_kwargs = {k: cc[k] for k in cc if k in ctrl_allowed}
    ctrl_params = VpnAccControllerParams(**ctrl_kwargs)
    controller = VpnAccController(ctrl_params)
    basic_ctrl.reset()
    motors.reset()

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
            title=str(mon_cfg.get("title", "VPN Acceleration Controller Monitor")),
            x_min=float(mon_cfg.get("x_min", 0.0)),
            x_max=float(mon_cfg.get("x_max", term_cfg.t_final)),
        ),
    )
    controller.set_monitor(monitor)

    metrics = Metrics(MetricsConfig(hit_radius=term_cfg.hit_radius))
    logging_enabled = bool(logger_cfg.get("enable", True))
    logger = NPZLogger(run_dir=str(logger_cfg.get("run_dir", "runs"))) if logging_enabled else None
    filename = logger_cfg.get("filename", None)

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
    last_bbox = tracker.track(last_cam) if last_cam is not None else None
    initial_obs = observer.make_observation(t_now=uav.t, uav=uav, cam=last_cam, tgt=tgt)
    termination_reason = None
    nan2 = (np.nan, np.nan)

    sim_viz.update(step=0, uav=uav, tgt=tgt, cam=last_cam, bbox=last_bbox, has_target=initial_obs.has_target)

    steps = int(np.ceil(term_cfg.t_final / sch.dt))
    for k in range(steps):
        t_now = uav.t

        if sch.should_camera(k):
            last_cam = camera.measure(uav, tgt, t_meas=t_now)
            last_bbox = tracker.track(last_cam) if last_cam is not None else None

        obs = observer.make_observation(t_now=t_now, uav=uav, cam=last_cam, tgt=tgt)

        if sch.should_control(k):
            bbox_width_px = None if last_bbox is None else float(last_bbox.bw)
            cmd = controller.compute_with_bbox(obs, bbox_width_px=bbox_width_px)
            last_cmd = cmd
        else:
            cmd = last_cmd
            bbox_width_px = None if last_bbox is None else float(last_bbox.bw)

        force_sp, motor_cmd = basic_ctrl.step_from_command(uav, cmd, sch.dt)
        motor_out = motors.step(motor_cmd.motor_current_cmd, sch.dt)

        uav = uav_model.step(uav, force_b=motor_out.force_b, torque_b=motor_out.torque_b, dt=sch.dt)
        tgt = tgt_model.step(tgt, sch.dt)
        last_i_cmd = motor_out.i_cmd
        last_omega = motor_out.omega
        sim_viz.update(step=k, uav=uav, tgt=tgt, cam=last_cam, bbox=last_bbox, has_target=obs.has_target)

        actual_thrust = float(-motor_out.force_b[2])
        if bbox_width_px is not None:
            monitor.push(name="bbox_width_px", color="tab:brown", data=bbox_width_px, t=t_now, group="vpn_bbox", step=k)
        if last_bbox is not None:
            monitor.push(name="bbox_height_px", color="tab:pink", data=last_bbox.bh, t=t_now, group="vpn_bbox", step=k)
        monitor.push(name="cmd_thrust", color="tab:red", data=force_sp.thrust_sp, t=t_now, group="command", step=k)
        monitor.push(name="actual_thrust", color="tab:blue", data=actual_thrust, t=t_now, group="command", step=k)
        monitor.push(name="cmd_x", color="tab:red", data=cmd.omega_cmd_b[0], t=t_now, group="omega_cmd", step=k)
        monitor.push(name="cmd_y", color="tab:orange", data=cmd.omega_cmd_b[1], t=t_now, group="omega_cmd", step=k)
        monitor.push(name="cmd_z", color="tab:green", data=cmd.omega_cmd_b[2], t=t_now, group="omega_cmd", step=k)
        monitor.push(name="actual_x", color="tab:blue", data=uav.w_b[0], t=t_now, group="omega_cmd", step=k)
        monitor.push(name="actual_y", color="tab:cyan", data=uav.w_b[1], t=t_now, group="omega_cmd", step=k)
        monitor.push(name="actual_z", color="tab:purple", data=uav.w_b[2], t=t_now, group="omega_cmd", step=k)
        monitor.update(step=k)

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
            logger.push("bbox_width_px", np.nan if bbox_width_px is None else bbox_width_px)
            logger.push("bbox_height_px", np.nan if last_bbox is None else last_bbox.bh)
            logger.push("bbox_center_uv", nan2 if last_bbox is None else np.array([last_bbox.u, last_bbox.v], dtype=float))
            if last_cam is None:
                logger.push("cam_valid", False)
                logger.push("cam_p_norm", nan2)
                logger.push("cam_uv", nan2)
            else:
                logger.push("cam_valid", bool(last_cam.valid))
                logger.push("cam_p_norm", nan2 if last_cam.p_norm is None else last_cam.p_norm)
                logger.push("cam_uv", nan2 if last_cam.uv_px is None else last_cam.uv_px)
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
    meta = {"config": Path(config_path).as_posix(), "seed": seed, "summary": summary, "rates": r}
    save_path = None
    if logger is not None:
        save_path = logger.save(meta=meta, filename=None if filename in (None, "null") else str(filename))

    print("=== VPN Acceleration Run Summary ===")
    for key, value in summary.items():
        print(f"{key}: {value}")
    if save_path is not None:
        print(f"saved: {save_path}")
    else:
        print("logging disabled: no file saved")
    sim_viz.close(block=True, termination_reason=termination_reason)


if __name__ == "__main__":
    main()
