# observe/perfect.py
from __future__ import annotations

from dataclasses import dataclass

from models.state import UAVState, TargetState, CameraMeasurement, Observation


@dataclass(slots=True)
class PerfectObserverConfig:
    """
    L1 perfect observer:
    - Uses perfect self-state (q_eb, w_b, v_e)
    - Uses camera measurement p_norm if available
    - No noise, no delay (delay will be handled in L2 by a delay queue)
    """
    pass


class PerfectObserver:
    def __init__(self, cfg: PerfectObserverConfig | None = None):
        self.cfg = cfg if cfg is not None else PerfectObserverConfig()

    def make_observation(
        self,
        t_now: float,
        uav: UAVState,
        cam: CameraMeasurement | None,
        tgt: TargetState | None = None,
    ) -> Observation:
        # Paper-style relative states: interceptor relative to target.
        # p_r = p_uav - p_tgt, v_r = v_uav - v_tgt
        p_r = None if tgt is None else (uav.p_e - tgt.p_e)
        v_r = None if tgt is None else (uav.v_e - tgt.v_e)

        if cam is None or cam.p_norm is None:
            return Observation(
                t=t_now,
                p_norm=None,
                q_eb=uav.q_eb,
                w_b=uav.w_b,
                v_e=uav.v_e,
                p_r=p_r,
                v_r=v_r,
                bearing_c=None,
                range_m=None,
                has_target=False,
            )

        return Observation(
            t=t_now,
            p_norm=cam.p_norm,
            q_eb=uav.q_eb,
            w_b=uav.w_b,
            v_e=uav.v_e,
            p_r=p_r,
            v_r=v_r,
            bearing_c=cam.bearing_c,
            range_m=cam.range_m,
            has_target=True,  # includes out-of-frame bearing when p_norm is available
        )
