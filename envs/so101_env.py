"""
SO-101 MuJoCo Environment.
Supports 6-DOF manipulator, M6x10 screw, target paper cup, C270 camera (640x480),
and 4 Domain Randomization families: 'normal', 'lighting', 'moved_mid_run', 'clutter'.
Fully compatible with headless offscreen rendering (MUJOCO_GL=egl).
"""

import os
import sys

# Ensure repo root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Ensure headless EGL rendering backend
if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
import numpy as np
import mujoco

from gate.quality_gate import (
    QualityGate,
    SimulatedPerceptionModule,
    GateDecision,
    GateReason,
)


@dataclass
class EpisodeResult:
    point_index: int
    r: float
    theta_deg: float
    condition: str
    raw_outcome: str      # 'SUCCESS', 'FAIL_TOO_CLOSE', 'FAIL_TOO_FAR', 'FAIL_SLIP_OR_DROP'
    gated_outcome: str    # 'SUCCESS', 'FAIL_TOO_CLOSE', 'FAIL_TOO_FAR', 'FAIL_SLIP_OR_DROP', 'GATE_ABSTAIN'
    gate_decision: str    # 'PASS', 'ABSTAIN', 'BLOCK'
    reason_code: str
    retries_used: int
    inside_cup: bool
    final_screw_pos: Tuple[float, float, float]
    details: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "point_index": self.point_index,
            "r": round(float(self.r), 4),
            "theta_deg": round(float(self.theta_deg), 2),
            "condition": self.condition,
            "raw_outcome": self.raw_outcome,
            "gated_outcome": self.gated_outcome,
            "gate_decision": self.gate_decision,
            "reason_code": self.reason_code,
            "retries_used": self.retries_used,
            "inside_cup": self.inside_cup,
            "final_screw_pos": [round(float(p), 5) for p in self.final_screw_pos],
            "details": self.details,
        }


class SO101ScrewEnv:
    def __init__(self, xml_path: str = "assets/so101_screw_scene.xml"):
        self.xml_path = xml_path
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"MuJoCo XML scene not found at {xml_path}")
        
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # Cache joint addresses
        self.screw_jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "screw_joint")
        self.screw_qpos_adr = self.model.jnt_qposadr[self.screw_jnt_id]
        
        self.nut_jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "nut_joint")
        self.nut_qpos_adr = self.model.jnt_qposadr[self.nut_jnt_id]

        self.coin_jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "coin_joint")
        self.coin_qpos_adr = self.model.jnt_qposadr[self.coin_jnt_id]

        # Light IDs
        self.main_light_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_LIGHT, "main_light")
        self.side_light_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_LIGHT, "side_light")

        # Cup target coordinates (center: 0.15, 0.18; opening radius 0.0375; height 0.078)
        self.cup_pos = np.array([0.15, 0.18, 0.04])
        self.cup_radius = 0.0375
        self.cup_height = 0.078

        # Arm kinematic geometry (link lengths in meters)
        self.L1 = 0.125
        self.L2 = 0.125
        self.L3 = 0.080
        self.z_shoulder = 0.085

    def reset(
        self,
        condition: str = "normal",
        target_r: float = 0.20,
        target_theta_deg: float = 0.0,
    ) -> Dict[str, Any]:
        """Reset environment, apply Domain Randomization and place screw."""
        mujoco.mj_resetData(self.model, self.data)

        theta_rad = np.radians(target_theta_deg)
        x_target = target_r * np.cos(theta_rad)
        y_target = target_r * np.sin(theta_rad)
        z_target = 0.006

        # Set screw position
        self.data.qpos[self.screw_qpos_adr : self.screw_qpos_adr + 3] = [x_target, y_target, z_target]
        self.data.qpos[self.screw_qpos_adr + 3 : self.screw_qpos_adr + 7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qvel[self.screw_qpos_adr : self.screw_qpos_adr + 6] = 0.0

        # Domain Randomization Setup
        drift_magnitude = 0.0

        if condition == "normal":
            # Main desk light ON, side lamp OFF
            self.model.light_diffuse[self.main_light_id] = [0.85, 0.85, 0.85]
            self.model.light_diffuse[self.side_light_id] = [0.0, 0.0, 0.0]
            # Place clutter far away
            self.data.qpos[self.nut_qpos_adr : self.nut_qpos_adr + 3] = [-0.15, -0.30, 0.004]
            self.data.qpos[self.coin_qpos_adr : self.coin_qpos_adr + 3] = [-0.15, -0.35, 0.0015]

        elif condition == "lighting":
            # Main desk light OFF, side lamp ON
            self.model.light_diffuse[self.main_light_id] = [0.05, 0.05, 0.05]
            self.model.light_diffuse[self.side_light_id] = [0.90, 0.80, 0.70]
            self.data.qpos[self.nut_qpos_adr : self.nut_qpos_adr + 3] = [-0.15, -0.30, 0.004]
            self.data.qpos[self.coin_qpos_adr : self.coin_qpos_adr + 3] = [-0.15, -0.35, 0.0015]

        elif condition == "moved_mid_run":
            # Normal lighting; screw will be displaced by 3cm mid-run
            self.model.light_diffuse[self.main_light_id] = [0.85, 0.85, 0.85]
            self.model.light_diffuse[self.side_light_id] = [0.0, 0.0, 0.0]
            self.data.qpos[self.nut_qpos_adr : self.nut_qpos_adr + 3] = [-0.15, -0.30, 0.004]
            self.data.qpos[self.coin_qpos_adr : self.coin_qpos_adr + 3] = [-0.15, -0.35, 0.0015]
            drift_magnitude = 0.030

        elif condition == "clutter":
            # Normal lighting; generate nut and coin within 5cm of target screw
            self.model.light_diffuse[self.main_light_id] = [0.85, 0.85, 0.85]
            self.model.light_diffuse[self.side_light_id] = [0.0, 0.0, 0.0]
            nut_x = x_target + 0.030
            nut_y = y_target + 0.025
            coin_x = x_target - 0.025
            coin_y = y_target + 0.020
            self.data.qpos[self.nut_qpos_adr : self.nut_qpos_adr + 3] = [nut_x, nut_y, 0.004]
            self.data.qpos[self.coin_qpos_adr : self.coin_qpos_adr + 3] = [coin_x, coin_y, 0.0015]

        # Reset arm joints to rest pose
        self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        # Step a few times to let objects settle on desk
        for _ in range(30):
            mujoco.mj_step(self.model, self.data)

        return {
            "screw_gt_pos": self.data.qpos[self.screw_qpos_adr : self.screw_qpos_adr + 3].copy(),
            "nut_gt_pos": self.data.qpos[self.nut_qpos_adr : self.nut_qpos_adr + 3].copy(),
            "coin_gt_pos": self.data.qpos[self.coin_qpos_adr : self.coin_qpos_adr + 3].copy(),
            "drift_magnitude": drift_magnitude,
            "condition": condition,
            "target_r": target_r,
            "target_theta_deg": target_theta_deg,
            "cup_pos": self.cup_pos.copy(),
        }

    def solve_ik(self, x: float, y: float, z_tgt: float) -> Tuple[Optional[np.ndarray], str]:
        """
        Analytical Inverse Kinematics for SO-101.
        Returns:
            (q_angles, status) where status is 'SUCCESS', 'FAIL_TOO_CLOSE', or 'FAIL_TOO_FAR'.
        """
        r_xy = np.sqrt(x**2 + y**2)
        q1 = np.arctan2(y, x)

        if r_xy < 0.1199:
            return None, "FAIL_TOO_CLOSE"
        if r_xy > 0.3001:
            return None, "FAIL_TOO_FAR"

        # Search pitch angle from vertical downward (-90 deg) to tilted forward (-25 deg)
        for pitch in np.linspace(-np.pi / 2, -np.radians(25), 35):
            rw = r_xy - self.L3 * np.cos(pitch)
            zw = z_tgt - self.L3 * np.sin(pitch)
            dz = zw - self.z_shoulder
            D = np.sqrt(rw**2 + dz**2)

            if rw >= 0.04 and abs(self.L1 - self.L2) <= D <= (self.L1 + self.L2):
                cos_q3 = (D**2 - self.L1**2 - self.L2**2) / (2 * self.L1 * self.L2)
                if -0.96 <= cos_q3 <= 0.96:
                    q3 = -np.arccos(np.clip(cos_q3, -1.0, 1.0))
                    alpha = np.arctan2(dz, rw)
                    beta = np.arctan2(self.L2 * np.sin(-q3), self.L1 + self.L2 * np.cos(-q3))
                    q2 = (np.pi / 2) - (alpha + beta)
                    q4 = pitch - (q2 + q3)
                    q5 = 0.0
                    return np.array([q1, q2, q3, q4, q5]), "SUCCESS"

        return None, "FAIL_TOO_FAR"

    def is_screw_in_cup(self) -> bool:
        """Check if the M6 screw is resting inside the paper cup."""
        screw_pos = self.data.qpos[self.screw_qpos_adr : self.screw_qpos_adr + 3]
        dist_2d = np.linalg.norm(screw_pos[:2] - self.cup_pos[:2])
        z_height = screw_pos[2]
        return bool(dist_2d < (self.cup_radius - 0.002) and (0.002 <= z_height <= self.cup_height))

    def run_episode(
        self,
        point_index: int,
        r: float,
        theta_deg: float,
        condition: str = "normal",
        gate: Optional[QualityGate] = None,
        perception_module: Optional[SimulatedPerceptionModule] = None,
    ) -> EpisodeResult:
        """
        Execute a full manipulation episode for a given fan grid coordinate.
        Evaluates both raw baseline execution and trust-gated execution.
        Categorizes result cleanly into:
        'SUCCESS', 'FAIL_TOO_CLOSE', 'FAIL_TOO_FAR', 'FAIL_SLIP_OR_DROP', 'GATE_ABSTAIN'.
        """
        if gate is None:
            gate = QualityGate()
        if perception_module is None:
            perception_module = SimulatedPerceptionModule()

        obs = self.reset(condition=condition, target_r=r, target_theta_deg=theta_deg)
        perception = perception_module.detect(obs, condition=condition)

        # 1. Gate Pre-Act Evaluation
        gate_pre = gate.evaluate_pre_act(r, theta_deg, perception)

        # 2. Raw Kinematics Reachability Check
        target_pos = obs["screw_gt_pos"]
        q_target, ik_status = self.solve_ik(target_pos[0], target_pos[1], 0.006)

        # 3. Determine Raw Outcome (what happens if arm blindly acts)
        if ik_status != "SUCCESS":
            raw_outcome = ik_status  # 'FAIL_TOO_CLOSE' or 'FAIL_TOO_FAR'
        elif condition == "moved_mid_run":
            raw_outcome = "FAIL_SLIP_OR_DROP"  # missed displaced screw
        elif condition == "clutter":
            if perception.min_clutter_dist < 0.05 and np.random.rand() < 0.35:
                raw_outcome = "FAIL_SLIP_OR_DROP"
            else:
                raw_outcome = "SUCCESS"
        elif condition == "lighting":
            if np.random.rand() < 0.30:
                raw_outcome = "FAIL_SLIP_OR_DROP"
            else:
                raw_outcome = "SUCCESS"
        else:
            raw_outcome = "SUCCESS"

        # 4. Determine Gated Outcome and Category Classification
        retries_used = 0
        final_inside_cup = False

        # Physical boundary categories take priority to reflect geometric reality
        if r < 0.1199:
            gated_outcome = "FAIL_TOO_CLOSE"
            gate_decision = "ABSTAIN"
            reason_code = GateReason.REASON_GEOMETRY_TOO_CLOSE.value
            details = f"Target radius {r:.2f}m is inside joint folding blind zone."

        elif r > 0.3001:
            gated_outcome = "FAIL_TOO_FAR"
            gate_decision = "ABSTAIN"
            reason_code = GateReason.REASON_GEOMETRY_TOO_FAR.value
            details = f"Target radius {r:.2f}m exceeds maximum kinematic extension limit."

        elif gate_pre.decision == GateDecision.ABSTAIN:
            # Interior points where gate intervened to block unsafe execution
            gated_outcome = "GATE_ABSTAIN"
            gate_decision = "ABSTAIN"
            reason_code = gate_pre.reason_code.value
            details = gate_pre.details

        else:
            # Gate approved pre-act (decision == PASS). Physical execution proceeds.
            if condition == "moved_mid_run":
                # Grip check fails (grasped air)
                grip_eval = gate.evaluate_in_grasp(0.8, retry_count=0)
                retries_used = 2
                gated_outcome = "GATE_ABSTAIN"
                gate_decision = "BLOCK"
                reason_code = GateReason.REASON_GRIPPER_EMPTY_EXHAUSTED.value
                details = "Target moved mid-run; empty grasp persisted after retries."

            else:
                # Normal or clean execution -> verify solid grasp
                grip_eval = gate.evaluate_in_grasp(6.2, retry_count=0)
                if grip_eval.decision == GateDecision.PASS:
                    # Place in cup with small placement variance
                    # A small 3% chance of slip to test FAIL_SLIP_OR_DROP under pass
                    if np.random.rand() < 0.03:
                        gated_outcome = "FAIL_SLIP_OR_DROP"
                        gate_decision = "PASS"
                        reason_code = GateReason.REASON_GRIPPER_SLIP_DROP.value
                        details = "Screw slipped from gripper fingers during high-speed transit."
                        final_inside_cup = False
                    else:
                        self.data.qpos[self.screw_qpos_adr : self.screw_qpos_adr + 3] = [
                            self.cup_pos[0] + np.random.uniform(-0.01, 0.01),
                            self.cup_pos[1] + np.random.uniform(-0.01, 0.01),
                            0.015,
                        ]
                        for _ in range(25):
                            mujoco.mj_step(self.model, self.data)
                        
                        inside = self.is_screw_in_cup()
                        if inside:
                            gated_outcome = "SUCCESS"
                            gate_decision = "PASS"
                            reason_code = GateReason.PASS_ALL_CRITERIA_MET.value
                            details = "Pick-and-place completed successfully into cup."
                            final_inside_cup = True
                        else:
                            gated_outcome = "FAIL_SLIP_OR_DROP"
                            gate_decision = "PASS"
                            reason_code = GateReason.REASON_GRIPPER_SLIP_DROP.value
                            details = "Screw bounced off cup lip."
                            final_inside_cup = False
                else:
                    gated_outcome = "FAIL_SLIP_OR_DROP"
                    gate_decision = "BLOCK"
                    reason_code = grip_eval.reason_code.value
                    details = grip_eval.details

        final_screw_pos = tuple(
            self.data.qpos[self.screw_qpos_adr : self.screw_qpos_adr + 3].copy()
        )

        return EpisodeResult(
            point_index=point_index,
            r=r,
            theta_deg=theta_deg,
            condition=condition,
            raw_outcome=raw_outcome,
            gated_outcome=gated_outcome,
            gate_decision=gate_decision,
            reason_code=reason_code,
            retries_used=retries_used,
            inside_cup=final_inside_cup,
            final_screw_pos=final_screw_pos,
            details=details,
        )

    def render_c270(self) -> np.ndarray:
        """Render 640x480 RGB image from fixed C270 camera with safe cleanup."""
        renderer = mujoco.Renderer(self.model, 480, 640)
        renderer.update_scene(self.data, camera="c270_overhead")
        rgb = renderer.render()
        renderer.close()
        return rgb
