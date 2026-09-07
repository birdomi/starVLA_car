"""Gravity-compensation feedforward torque (sol_tauff equivalent) via MuJoCo.

xr_teleoperate's IK returns sol_tauff = pin.rnea(model, data, q, 0, 0) — the
static torque holding the arms against gravity — and sends it as the motor tau
feedforward. When replaying dataset actions or model predictions WITHOUT it,
the arms droop by gravity_torque/kp (2-4 cm at the end-effector).

This module computes the same torques from the bundled H1-2 MJCF, so the deploy
loop needs only `pip install mujoco` (no pinocchio):

    from gravity_comp import GravityComp
    gc = GravityComp()                       # loads sim/assets/scene.xml
    tau_l, tau_r = gc.tauff_from_action(a42) # 7 torques per arm, xr_teleoperate order
    arm_ctrl.ctrl_dual_arm(q_target, np.concatenate([tau_l, tau_r]))

Torque order per arm: [shoulder_pitch, shoulder_roll, shoulder_yaw,
elbow_pitch, elbow_roll, wrist_pitch, wrist_yaw] — same as the action vector
and H1_2_ArmController's arm joint order (verify against your controller's
motor index mapping before first use).
"""

from pathlib import Path

import mujoco
import numpy as np

ARM_JOINT_NAMES = [
    "{side}_shoulder_pitch_joint",
    "{side}_shoulder_roll_joint",
    "{side}_shoulder_yaw_joint",
    "{side}_elbow_pitch_joint",
    "{side}_elbow_roll_joint",
    "{side}_wrist_pitch_joint",
    "{side}_wrist_yaw_joint",
]
# action(42) = head(2) + left_arm(7) + left_hand(13) + right_arm(7) + right_hand(13)
LARM, RARM, HEAD = slice(2, 9), slice(22, 29), slice(0, 2)


class GravityComp:
    def __init__(self, scene_xml: str | None = None):
        if scene_xml is None:
            scene_xml = str(Path(__file__).resolve().parent / "assets" / "scene.xml")
        self.model = mujoco.MjModel.from_xml_path(scene_xml)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetData(self.model, self.data)
        self._joints = {
            side: [self.model.joint(n.format(side=side)) for n in ARM_JOINT_NAMES]
            for side in ("left", "right")
        }
        self._head = [self.model.joint(n) for n in ("head_yaw_joint", "head_pitch_joint")]

    def tauff(self, q_left7, q_right7, head2=None):
        """Static gravity torque for each arm joint at the given configuration."""
        for side, q in (("left", q_left7), ("right", q_right7)):
            for j, v in zip(self._joints[side], q):
                self.data.qpos[j.qposadr[0]] = v
        if head2 is not None:  # [yaw, pitch] with pitch positive-up (dataset convention)
            self.data.qpos[self._head[0].qposadr[0]] = head2[0]
            self.data.qpos[self._head[1].qposadr[0]] = -head2[1]
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)
        out = []
        for side in ("left", "right"):
            out.append(np.array([self.data.qfrc_bias[j.dofadr[0]] for j in self._joints[side]]))
        return out[0], out[1]

    def tauff_from_action(self, action42):
        """Convenience: slice a 42-dim action vector and return (tau_left7, tau_right7)."""
        a = np.asarray(action42, dtype=np.float64)
        return self.tauff(a[LARM], a[RARM], head2=a[HEAD])


if __name__ == "__main__":
    gc = GravityComp()
    a = np.zeros(42)
    tl, tr = gc.tauff_from_action(a)
    print("tau_left :", tl.round(2))
    print("tau_right:", tr.round(2))
