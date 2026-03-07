from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np

try:
    import yaml
except ImportError as e:
    raise SystemExit("Please install PyYAML: pip install pyyaml") from e

from control.basic_control.attitude_controller import AttitudeController, AttitudeControllerParams
from control.basic_control.basic_controller import BasicController
from control.basic_control.position_controller import PositionController, PositionControllerParams
from control.basic_control.setpoints import PositionSetpoint
from control.basic_control.velocity_controller import VelocityController, VelocityControllerParams
from models.motors import Motors
from models.rigid_body import RigidBody6DoF, RigidBodyParams
from models.state import Observation, TargetState, UAVState
from sensors.camera import CameraExtrinsics, CameraIntrinsics, PinholeCamera
from sim.scheduler import MultiRateScheduler, RateConfig
from sim.simulator import Simulator, TerminationConfig
from utils.log import NPZLogger
from visualization.monitor import Monitor, MonitorConfig


def _load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_observation(uav: UAVState, p_sp_e: np.ndarray) -> Observation:
    return Observation(
        t=uav.t,
        p_norm=None,
        q_eb=uav.q_eb,
        w_b=uav.w_b,
        v_e=uav.v_e,
        p_r=uav.p_e - p_sp_e,
        v_r=uav.v_e,
        has_target=False,
    )


def _figure_eight_sp(t: float, center_e: np.ndarray) -> np.ndarray:
    amp_x = 50.0
    amp_y = 35.0
    omega = 2.0 * np.pi / 36.0
    return np.array(
        [
            center_e[0] + amp_x * np.sin(omega * t),
            center_e[1] + amp_y * np.sin(2.0 * omega * t),
            center_e[2],
        ],
        dtype=float,
    )


def _figure_eight_yaw_sp(t: float) -> float:
    amp_x = 50.0
    amp_y = 35.0
    omega = 2.0 * np.pi / 36.0
    vx = amp_x * omega * np.cos(omega * t)
    vy = 2.0 * amp_y * omega * np.cos(2.0 * omega * t)
    return float(np.arctan2(vy, vx))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/pos_ctrl.yaml")
    ap.add_argument("--p-sp", type=float, nargs=3, default=None, help="position setpoint in NED")
    ap.add_argument("--yaw-sp", type=float, default=None, help="yaw setpoint placeholder")
    ap.add_argument("--t-final", type=float, default=None, help="override simulation horizon")
    ap.add_argument("--log-name", type=str, default="pos_ctrl_sim.npz")
    args = ap.parse_args()

    cfg = _load_cfg(args.config)
    logger_cfg = cfg.get("logging", {})

    seed = int(logger_cfg.get("seed", 0))
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
        realtime=bool(viz_cfg.get("realtime", True)),
        enable_fov=bool(viz_cfg.get("enable_fov", True)),
        cam_width=int(cfg.get("camera", {}).get("width", 640)),
        cam_height=int(cfg.get("camera", {}).get("height", 480)),
        cam_fx=float(cfg.get("camera", {}).get("fx", 320.0)),
        cam_fy=float(cfg.get("camera", {}).get("fy", 320.0)),
        cam_cx=float(cfg.get("camera", {}).get("cx", 320.0)),
        cam_cy=float(cfg.get("camera", {}).get("cy", 240.0)),
        trail_len=int(viz_cfg.get("trail_len", 1000)),
        auto_axis=bool(viz_cfg.get("auto_axis", False)),
        ned_axes=bool(viz_cfg.get("ned_axes", True)),
        map_center=tuple(viz_cfg.get("map_center", [0.0, 0.0, 0.0])),
        map_size=tuple(viz_cfg.get("map_size", [160.0, 120.0, 80.0])),
        colors=viz_cfg.get("colors", {}),
        uav_visual_scale=float(viz_cfg.get("uav_visual_scale", 1.0)),
        target_marker_size=float(viz_cfg.get("target_marker_size", 6.0)),
        fov_target_marker_size=float(viz_cfg.get("fov_target_marker_size", 7.0)),
    )

    u0 = cfg["uav0"]
    uav = UAVState(
        t=float(u0["t"]),
        p_e=np.array(u0["p_e"], dtype=float),
        v_e=np.array(u0["v_e"], dtype=float),
        q_eb=np.array(u0["q_eb"], dtype=float),
        w_b=np.array(u0["w_b"], dtype=float),
    )

    pos_sp_cfg = cfg.get("position_setpoint", {})
    p_sp_center_e = np.array(
        args.p_sp if args.p_sp is not None else pos_sp_cfg.get("p_sp_e", [0.0, 0.0, 0.0]),
        dtype=float,
    )
    yaw_sp_cfg = float(args.yaw_sp if args.yaw_sp is not None else pos_sp_cfg.get("yaw_sp", 0.0))
    use_traj_yaw = args.yaw_sp is None
    p_sp_e = _figure_eight_sp(uav.t, p_sp_center_e)
    yaw_sp = _figure_eight_yaw_sp(uav.t) if use_traj_yaw else yaw_sp_cfg
    tgt = TargetState(t=uav.t, p_e=p_sp_e.copy(), v_e=np.zeros(3, dtype=float))

    rb_params = RigidBodyParams.from_yaml(cfg["rigid_body"]["params_yaml"])
    uav_model = RigidBody6DoF(rb_params)
    motors = Motors(rb_params)

    cam_cfg = cfg.get("camera", {})
    camera = PinholeCamera(
        CameraIntrinsics(
            fx=float(cam_cfg.get("fx", 320.0)),
            fy=float(cam_cfg.get("fy", 320.0)),
            cx=float(cam_cfg.get("cx", 320.0)),
            cy=float(cam_cfg.get("cy", 240.0)),
            width=int(cam_cfg.get("width", 640)),
            height=int(cam_cfg.get("height", 480)),
        ),
        CameraExtrinsics(
            mount_pitch_deg=float(cam_cfg.get("mount_pitch_deg", 20.0)),
        ),
    )

    pos_cfg = cfg.get("position_controller", {})
    vel_cfg = cfg.get("velocity_controller", {})
    att_cfg = cfg.get("attitude_controller", {})

    pos_ctrl = PositionController(
        PositionControllerParams(
            kp=np.array(pos_cfg.get("kp", [1.0, 1.0, 1.0]), dtype=float),
            v_max=float(pos_cfg.get("v_max", 8.0)),
        )
    )
    vel_ctrl = VelocityController(
        VelocityControllerParams(
            mass=rb_params.mass,
            g=rb_params.g,
            kp=np.array(vel_cfg.get("kp", [1.0, 1.0, 1.0]), dtype=float),
        )
    )
    att_ctrl = AttitudeController(
        AttitudeControllerParams(
            k_r=np.array(att_cfg.get("k_r", [4.0, 4.0, 2.0]), dtype=float),
            omega_max=float(att_cfg.get("omega_max", 6.0)),
        )
    )
    basic_ctrl = BasicController(
        rb_params,
        position_controller=pos_ctrl,
        velocity_controller=vel_ctrl,
        attitude_controller=att_ctrl,
    )

    basic_ctrl.update_setpoint(PositionSetpoint(p_sp_e=p_sp_e, yaw_sp=yaw_sp))
    basic_ctrl.reset()
    motors.reset()

    term_cfg = cfg.get("termination", {})
    t_final = float(args.t_final if args.t_final is not None else term_cfg.get("t_final", 20.0))
    hit_radius = float(term_cfg.get("hit_radius", 10))
    term = TerminationConfig(t_final=t_final, hit_radius=hit_radius)
    monitor = Monitor(
        scheduler=sch,
        cfg=MonitorConfig(
            enable=bool(mon_cfg.get("enable", True)),
            realtime=bool(mon_cfg.get("realtime", True)),
            max_points=mon_cfg.get("max_points", None),
            step_stride=int(mon_cfg.get("step_stride", 1)),
            title=str(mon_cfg.get("title", "Position Controller Monitor")),
            x_min=float(mon_cfg.get("x_min", 0.0)),
            x_max=float(mon_cfg.get("x_max", t_final)),
        ),
    )

    logging_enabled = bool(logger_cfg.get("enable", True))
    logger = NPZLogger(run_dir=str(logger_cfg.get("run_dir", "runs"))) if logging_enabled else None

    steps = int(np.ceil(term.t_final / sch.dt))
    err_norm = float(np.linalg.norm(uav.p_e - p_sp_e))
    last_cam = None

    ############### Main simulation loop ##############

    for k in range(steps):
        p_sp_e = _figure_eight_sp(uav.t, p_sp_center_e)
        yaw_sp = _figure_eight_yaw_sp(uav.t) if use_traj_yaw else yaw_sp_cfg
        tgt = TargetState(t=uav.t, p_e=p_sp_e.copy(), v_e=np.zeros(3, dtype=float))
        if sch.should_camera(k):
            last_cam = camera.measure(uav, tgt, t_meas=uav.t)
        basic_ctrl.update_setpoint(PositionSetpoint(p_sp_e=p_sp_e, yaw_sp=yaw_sp))
        obs = _build_observation(uav, p_sp_e)
        motor_cmd = basic_ctrl.step(uav_state=uav, obs=obs, t_now=uav.t, dt=sch.dt)
        motor_out = motors.step(motor_cmd.motor_current_cmd, sch.dt)

        uav = uav_model.step(uav, force_b=motor_out.force_b, torque_b=motor_out.torque_b, dt=sch.dt)

        tgt = TargetState(t=uav.t, p_e=p_sp_e.copy(), v_e=np.zeros(3, dtype=float))
        sim_viz.update(step=k, uav=uav, tgt=tgt, cam=last_cam, has_target=(last_cam.valid if last_cam is not None else False))
        monitor.push(name="x_actual", color="tab:blue", data=uav.p_e[0], t=uav.t, group="pos_x", step=k)
        monitor.push(name="x_ref", color="black", data=p_sp_e[0], t=uav.t, group="pos_x", step=k)
        monitor.push(name="y_actual", color="tab:orange", data=uav.p_e[1], t=uav.t, group="pos_y", step=k)
        monitor.push(name="y_ref", color="black", data=p_sp_e[1], t=uav.t, group="pos_y", step=k)
        monitor.push(name="z_actual", color="tab:green", data=uav.p_e[2], t=uav.t, group="pos_z", step=k)
        monitor.push(name="z_ref", color="black", data=p_sp_e[2], t=uav.t, group="pos_z", step=k)
        monitor.push(name="yaw_ref", color="tab:red", data=yaw_sp, t=uav.t, group="yaw", step=k)
        monitor.update(step=k)

        err = uav.p_e - p_sp_e
        err_norm = float(np.linalg.norm(err))

        # Log data at each step
        if logger is not None:
            logger.push("t", uav.t)
            logger.push("uav_p", uav.p_e)
            logger.push("uav_v", uav.v_e)
            logger.push("uav_q", uav.q_eb)
            logger.push("uav_w", uav.w_b)
            logger.push("p_sp", p_sp_e)
            logger.push("yaw_sp", yaw_sp)
            logger.push("pos_err", err)
            logger.push("pos_err_norm", err_norm)
            logger.push("motor_cmd", motor_cmd.motor_current_cmd)
            logger.push("motor_omega", motor_out.omega)


    ############## Simulation ended, prepare summary and save logs if enabled ##############
    summary = {
        "p_sp_e": p_sp_e.tolist(),
        "final_p_e": uav.p_e.tolist(),
        "final_v_e": uav.v_e.tolist(),
        "final_err_norm": err_norm,
        "reached": bool(err_norm <= term.hit_radius),
        "t_final": float(uav.t),
    }

    save_path = None
    if logger is not None:
        save_path = logger.save(
            meta={
                "config": Path(args.config).as_posix(),
                "seed": seed,
                "summary": summary,
            },
            filename=args.log_name,
        )

    print("=== Position Control Summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")
    if save_path is not None:
        print(f"saved: {save_path}")
    else:
        print("logging disabled: no file saved")

    sim_viz.close(block=True)


if __name__ == "__main__":
    main()
