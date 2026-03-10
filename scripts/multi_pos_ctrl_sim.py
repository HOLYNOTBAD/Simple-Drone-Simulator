from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

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
from sim.scheduler import MultiRateScheduler, RateConfig
from sim.simulator import Simulator, TerminationConfig


@dataclass(slots=True)
class TrajectoryPoint:
    t_s: float
    p_e: np.ndarray


@dataclass(slots=True)
class MultiUAVAgent:
    uav_id: str
    uav: UAVState
    tgt: TargetState
    uav_model: RigidBody6DoF
    motors: Motors
    controller: BasicController
    pos_sp: PositionSetpoint
    obs: Observation
    obs_p_r: np.ndarray
    traj: list[TrajectoryPoint]
    next_wp_idx: int
    err_norm: float = 0.0


def _load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _normalize(name: str) -> str:
    return "".join(ch for ch in name.strip().lower() if ch.isalnum())


def _pick_field(fieldnames: list[str], candidates: tuple[str, ...]) -> str:
    norm_map = {_normalize(name): name for name in fieldnames}
    for cand in candidates:
        key = _normalize(cand)
        if key in norm_map:
            return norm_map[key]
    raise ValueError(f"CSV missing required column, expected one of: {candidates}")


def _load_csv_trajectories(path: str) -> dict[str, list[TrajectoryPoint]]:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV file must contain a header row")

        id_col = _pick_field(reader.fieldnames, ("id", "uav_id", "drone_id", "aircraft_id"))
        x_col = _pick_field(reader.fieldnames, ("x", "pos_x", "p_x", "px"))
        y_col = _pick_field(reader.fieldnames, ("y", "pos_y", "p_y", "py"))
        z_col = _pick_field(reader.fieldnames, ("z", "pos_z", "p_z", "pz"))
        t_col = _pick_field(reader.fieldnames, ("timestamp_ms", "time_ms", "t_ms", "timestamp", "time"))

        traj_map: dict[str, list[TrajectoryPoint]] = {}
        for row_idx, row in enumerate(reader, start=2):
            try:
                uav_id = str(row[id_col]).strip()
                if not uav_id:
                    raise ValueError("empty ID")
                t_ms = float(row[t_col])
                p_e = np.array([float(row[x_col]), float(row[y_col]), float(row[z_col])], dtype=float)
            except Exception as e:
                raise ValueError(f"Invalid CSV row {row_idx}: {row}") from e

            traj_map.setdefault(uav_id, []).append(TrajectoryPoint(t_s=1e-3 * t_ms, p_e=p_e))

    if not traj_map:
        raise ValueError("CSV file contains no trajectory rows")

    for uav_id, traj in traj_map.items():
        traj.sort(key=lambda pt: pt.t_s)
        first_t = traj[0].t_s
        if first_t > 1e-9:
            raise ValueError(
                f"UAV '{uav_id}' first timestamp is {first_t:.6f}s; expected to start near 0s so the initial state is defined"
            )
    return traj_map


def _build_controller(cfg: dict, rb_params: RigidBodyParams) -> BasicController:
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
    return BasicController(
        rb_params,
        position_controller=pos_ctrl,
        velocity_controller=vel_ctrl,
        attitude_controller=att_ctrl,
    )


def _build_agents(cfg: dict, traj_map: dict[str, list[TrajectoryPoint]]) -> list[MultiUAVAgent]:
    base_u0 = cfg["uav0"]
    rb_params = RigidBodyParams.from_yaml(cfg["rigid_body"]["params_yaml"])
    yaw_sp = float(cfg.get("position_setpoint", {}).get("yaw_sp", 0.0))
    zero3 = np.zeros(3, dtype=float)

    agents: list[MultiUAVAgent] = []
    for uav_id in sorted(traj_map.keys()):
        traj = traj_map[uav_id]
        p0 = traj[0].p_e.copy()
        uav = UAVState(
            t=float(base_u0.get("t", 0.0)),
            p_e=p0,
            v_e=np.array(base_u0["v_e"], dtype=float),
            q_eb=np.array(base_u0["q_eb"], dtype=float),
            w_b=np.array(base_u0["w_b"], dtype=float),
        )
        pos_sp = PositionSetpoint(p_sp_e=p0.copy(), yaw_sp=yaw_sp)
        tgt = TargetState(t=uav.t, p_e=pos_sp.p_sp_e.copy(), v_e=zero3.copy())
        obs_p_r = np.zeros(3, dtype=float)
        np.subtract(uav.p_e, pos_sp.p_sp_e, out=obs_p_r)
        obs = Observation(
            t=uav.t,
            p_norm=None,
            q_eb=uav.q_eb.copy(),
            w_b=uav.w_b.copy(),
            v_e=uav.v_e.copy(),
            p_r=obs_p_r,
            v_r=uav.v_e.copy(),
            has_target=False,
        )

        controller = _build_controller(cfg, rb_params)
        controller.update_setpoint(pos_sp)
        controller.reset()

        motors = Motors(rb_params)
        motors.reset()

        agent = MultiUAVAgent(
            uav_id=uav_id,
            uav=uav,
            tgt=tgt,
            uav_model=RigidBody6DoF(rb_params),
            motors=motors,
            controller=controller,
            pos_sp=pos_sp,
            obs=obs,
            obs_p_r=obs_p_r,
            traj=traj,
            next_wp_idx=1,
            err_norm=0.0,
        )
        agents.append(agent)
    return agents


def _update_setpoint_from_csv(agent: MultiUAVAgent, current_t: float) -> None:
    while agent.next_wp_idx < len(agent.traj) and agent.traj[agent.next_wp_idx].t_s <= current_t + 1e-12:
        wp = agent.traj[agent.next_wp_idx]
        np.copyto(agent.pos_sp.p_sp_e, wp.p_e)
        agent.next_wp_idx += 1
    agent.tgt.t = current_t
    np.copyto(agent.tgt.p_e, agent.pos_sp.p_sp_e)
    agent.tgt.v_e.fill(0.0)


def _build_visualizer(cfg: dict, sch: MultiRateScheduler) -> Simulator:
    viz_cfg = cfg.get("visualization", {})
    return Simulator(
        scheduler=sch,
        enable=bool(viz_cfg.get("enable", True)),
        realtime=bool(viz_cfg.get("realtime", True)),
        enable_fov=False,
        cam_width=640,
        cam_height=480,
        cam_fx=320.0,
        cam_fy=320.0,
        cam_cx=320.0,
        cam_cy=240.0,
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


def _resolve_t_final(args_t_final: float | None, cfg: dict, traj_map: dict[str, list[TrajectoryPoint]]) -> float:
    if args_t_final is not None:
        return float(args_t_final)
    cfg_t_final = float(cfg.get("termination", {}).get("t_final", 20.0))
    csv_t_final = max(traj[-1].t_s for traj in traj_map.values())
    return max(cfg_t_final, csv_t_final)


def _current_waypoint_index(agent: MultiUAVAgent) -> int:
    return max(0, agent.next_wp_idx - 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/multi_pos_ctrl.yaml")
    ap.add_argument("--csv", type=str, required=True, help="trajectory csv with columns: ID, X, Y, Z, timestamp_ms")
    ap.add_argument("--yaw-sp", type=float, default=None, help="override yaw setpoint for all UAVs")
    ap.add_argument("--t-final", type=float, default=None, help="override simulation horizon")
    args = ap.parse_args()

    cfg = _load_cfg(args.config)
    seed = int(cfg.get("seed", cfg.get("logging", {}).get("seed", 0)))
    np.random.seed(seed)

    rates_cfg = cfg["rates"]
    rates = RateConfig(
        physics_hz=float(rates_cfg["physics_hz"]),
        control_hz=float(rates_cfg["control_hz"]),
        camera_hz=float(rates_cfg["camera_hz"]),
        visualization_hz=float(rates_cfg.get("visualization_hz", 10.0)),
    )
    sch = MultiRateScheduler(rates)

    traj_map = _load_csv_trajectories(args.csv)
    agents = _build_agents(cfg, traj_map)
    if args.yaw_sp is not None:
        for agent in agents:
            agent.pos_sp.yaw_sp = float(args.yaw_sp)

    sim_viz = _build_visualizer(cfg, sch)

    t_final = _resolve_t_final(args.t_final, cfg, traj_map)
    hit_radius = float(cfg.get("termination", {}).get("hit_radius", 10.0))
    term = TerminationConfig(t_final=t_final, hit_radius=hit_radius)
    steps = int(np.ceil(term.t_final / sch.dt))

    ############### Main simulation loop ##############

    for k in range(steps):
        for agent in agents:
            _update_setpoint_from_csv(agent, agent.uav.t)
            agent.obs.t = agent.uav.t
            agent.obs.q_eb = agent.uav.q_eb
            agent.obs.w_b = agent.uav.w_b
            agent.obs.v_e = agent.uav.v_e
            agent.obs.v_r = agent.uav.v_e
            np.subtract(agent.uav.p_e, agent.pos_sp.p_sp_e, out=agent.obs_p_r)
            motor_cmd = agent.controller.step(uav_state=agent.uav, obs=agent.obs, t_now=agent.uav.t, dt=sch.dt)
            motor_out = agent.motors.step(motor_cmd.motor_current_cmd, sch.dt)
            agent.uav = agent.uav_model.step(
                agent.uav,
                force_b=motor_out.force_b,
                torque_b=motor_out.torque_b,
                dt=sch.dt,
            )
            agent.tgt.t = agent.uav.t
            np.copyto(agent.tgt.p_e, agent.pos_sp.p_sp_e)
            np.subtract(agent.uav.p_e, agent.pos_sp.p_sp_e, out=agent.obs_p_r)
            agent.err_norm = float(np.linalg.norm(agent.obs_p_r))

        sim_viz.update_multi(
            step=k,
            uavs={agent.uav_id: agent.uav for agent in agents},
            tgts={agent.uav_id: agent.tgt for agent in agents},
            cam=None,
            has_target=False,
        )

    ############## Simulation ended, prepare summary ##############
    print("=== Multi Position Control Summary ===")
    print(f"uav_count: {len(agents)}")
    print(f"t_final: {float(max(agent.uav.t for agent in agents))}")
    for agent in agents:
        current_wp_idx = _current_waypoint_index(agent)
        summary = {
            "final_p_e": agent.uav.p_e.tolist(),
            "final_v_e": agent.uav.v_e.tolist(),
            "final_setpoint_e": agent.pos_sp.p_sp_e.tolist(),
            "final_err_norm": agent.err_norm,
            "reached": bool(agent.err_norm <= term.hit_radius),
            "waypoint_count": len(agent.traj),
            "current_waypoint_index": current_wp_idx,
            "last_waypoint_time_s": float(agent.traj[-1].t_s),
        }
        print(f"--- UAV {agent.uav_id} ---")
        for key, value in summary.items():
            print(f"{key}: {value}")

    max_csv_t_final = max(traj[-1].t_s for traj in traj_map.values())
    if term.t_final + 1e-12 < max_csv_t_final:
        print("[warning] simulation ended before the last CSV waypoint time was reached")
        print(f"[warning] simulation_t_final={term.t_final:.3f}s, last_csv_waypoint_t={max_csv_t_final:.3f}s")

    sim_viz.close(block=True)


if __name__ == "__main__":
    main()
