"""Fourier GR1 (arms + waist) forward / inverse kinematics in torch and the 29-D joint action <-> wrist pose adapter.

Kinematic tree parsed from assets/gr1_robot.xml (unmodified copy of robosuite 1.5.1 models/assets/robots/gr1/robot.xml,
robosuite commit a071383d, MIT): base -> torso_waist_yaw / pitch / roll -> {left,right}_arm_mount -> shoulder pitch /
roll / yaw -> elbow pitch -> wrist yaw / roll / pitch -> {left,right}_eef (the hand mount frame = wrist pose). All
joints are hinges at their body origin (checked at parse time). Poses are in the robot BASE frame (body "base").

RoboCasa GR1 29-D action / state layout (data_config FourierGr1ArmsWaistDataConfig key order):
    left_arm 0:7 | right_arm 7:14 | left_hand 14:20 | right_hand 20:26 | waist 26:29
    arm joints in MJCF order: shoulder_pitch, shoulder_roll, shoulder_yaw, elbow_pitch, wrist_yaw, wrist_roll, wrist_pitch
    waist: torso_waist_yaw, torso_waist_pitch, torso_waist_roll
Arm / waist entries are ABSOLUTE joint position targets (rad); hand entries are 6-D finger commands ([-3, 3]) that have
no kinematics here (they are corrected additively). The 58-D state is sin/cos per key block: [sin(d), cos(d)] for
left_arm, right_arm, left_hand, right_hand, waist (StateActionSinCosTransform) -> angles = atan2.
"""

import os
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from starVLA.model.modules.geo_residual.se3 import axis_angle_to_matrix, quat_wxyz_to_matrix, rotation_residual

DEFAULT_XML = os.path.join(os.path.dirname(__file__), "assets", "gr1_robot.xml")
SIDES = ("left", "right")
ARM = {"left": slice(0, 7), "right": slice(7, 14)}
HAND = {"left": slice(14, 20), "right": slice(20, 26)}
WAIST = slice(26, 29)
STATE_BLOCKS = (("left_arm", 7), ("right_arm", 7), ("left_hand", 6), ("right_hand", 6), ("waist", 3))
WAIST_JOINTS = ["torso_waist_yaw", "torso_waist_pitch", "torso_waist_roll"]
ARM_JOINTS = {s: [f"{s[0]}_{j}" for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow_pitch",
                                          "wrist_yaw", "wrist_roll", "wrist_pitch")] for s in SIDES}
EEF_BODY = {"left": "left_eef", "right": "right_eef"}


def state_sincos_to_raw(state58: torch.Tensor) -> torch.Tensor:
    """[..., 58] sin/cos state -> [..., 29] angles in the action key order."""
    out, i = [], 0
    for _, d in STATE_BLOCKS:
        out.append(torch.atan2(state58[..., i:i + d], state58[..., i + d:i + 2 * d]))
        i += 2 * d
    return torch.cat(out, dim=-1)


def _floats(s: Optional[str], n: int, default) -> List[float]:
    if s is None:
        return list(default)
    v = [float(x) for x in s.split()]
    if len(v) != n:
        raise ValueError(f"expected {n} floats, got {s!r}")
    return v


def _parse_chain(xml_path: str, tip: str) -> List[Tuple[List[float], List[float], List[Tuple[str, List[float], Tuple[float, float]]]]]:
    """Bodies from the child of 'base' down to `tip`: [(pos, quat_wxyz, [(joint, axis, range), ...]), ...]."""
    root = ET.parse(xml_path).getroot()
    base = root.find(".//body[@name='base']")
    if base is None:
        raise ValueError(f"{xml_path}: no body 'base'")

    def walk(body, path):
        if body.get("name") == tip:
            return path
        for ch in body.findall("body"):
            r = walk(ch, path + [ch])
            if r is not None:
                return r
        return None

    path = walk(base, [])
    if path is None:
        raise ValueError(f"{xml_path}: body {tip!r} not under 'base'")
    chain = []
    for b in path:
        joints = []
        for j in b.findall("joint"):
            if j.get("type", "hinge") != "hinge":
                raise ValueError(f"joint {j.get('name')}: only hinge joints are supported")
            if any(abs(v) > 0 for v in _floats(j.get("pos"), 3, (0, 0, 0))):
                raise ValueError(f"joint {j.get('name')}: non-zero joint pos is not supported")
            rng = _floats(j.get("range"), 2, (-3.2, 3.2))
            joints.append((j.get("name"), _floats(j.get("axis"), 3, (0, 0, 1)), (rng[0], rng[1])))
        chain.append((_floats(b.get("pos"), 3, (0, 0, 0)), _floats(b.get("quat"), 4, (1, 0, 0, 0)), joints))
    return chain


class GR1Kinematics(nn.Module):
    """Batched FK of both wrists (as a function of waist + arm joints) and damped-least-squares IK of an arm."""

    def __init__(self, xml_path: str = DEFAULT_XML):
        super().__init__()
        self.sides = SIDES
        for side in SIDES:
            chain = _parse_chain(xml_path, EEF_BODY[side])
            names = [j[0] for _, _, js in chain for j in js]
            if names != WAIST_JOINTS + ARM_JOINTS[side]:
                raise ValueError(f"{side} chain joints {names} != expected {WAIST_JOINTS + ARM_JOINTS[side]}")
            # one entry per body: fixed transform (pos, R) then 0 or 1 joint axis
            pos = torch.tensor([c[0] for c in chain], dtype=torch.float32)
            rot = quat_wxyz_to_matrix(torch.tensor([c[1] for c in chain], dtype=torch.float32))
            axis = torch.tensor([c[2][0][1] if c[2] else (0.0, 0.0, 0.0) for c in chain], dtype=torch.float32)
            jidx = torch.tensor([names.index(c[2][0][0]) if c[2] else -1 for c in chain], dtype=torch.long)
            if any(len(c[2]) > 1 for c in chain):
                raise ValueError("at most one joint per body is supported")
            lim = torch.tensor([j[2] for _, _, js in chain for j in js], dtype=torch.float32)  # [10, 2]
            self.register_buffer(f"{side}_pos", pos, persistent=False)
            self.register_buffer(f"{side}_rot", rot, persistent=False)
            self.register_buffer(f"{side}_axis", axis, persistent=False)
            self.register_buffer(f"{side}_jidx", jidx, persistent=False)
            self.register_buffer(f"{side}_arm_lo", lim[3:, 0].clone(), persistent=False)
            self.register_buffer(f"{side}_arm_hi", lim[3:, 1].clone(), persistent=False)

    def fk(self, side: str, waist: torch.Tensor, arm: torch.Tensor, return_joints: bool = False):
        """waist [..., 3], arm [..., 7] (rad) -> wrist p [..., 3], R [..., 3, 3] in the base frame
        (+ joint origins [..., 10, 3] and world axes [..., 10, 3] for the 3 waist + 7 arm joints)."""
        q = torch.cat([waist, arm], dim=-1)
        dt = q.dtype
        pos, rot = getattr(self, f"{side}_pos").to(dt), getattr(self, f"{side}_rot").to(dt)
        axis, jidx = getattr(self, f"{side}_axis").to(dt), getattr(self, f"{side}_jidx")
        p = torch.zeros(q.shape[:-1] + (3,), dtype=dt, device=q.device)
        R = torch.eye(3, dtype=dt, device=q.device).expand(q.shape[:-1] + (3, 3))
        jp, jz = [], []
        for b in range(pos.shape[0]):
            p = p + (R @ pos[b].view(3, 1)).squeeze(-1)
            R = R @ rot[b]
            if int(jidx[b]) >= 0:
                a = axis[b]
                jp.append(p)
                jz.append((R @ a.view(3, 1)).squeeze(-1))
                R = R @ axis_angle_to_matrix(a * q[..., int(jidx[b])].unsqueeze(-1))
        if return_joints:
            return p, R, torch.stack(jp, dim=-2), torch.stack(jz, dim=-2)
        return p, R

    def joints_to_wrists(self, a29: torch.Tensor) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """[..., 29] raw joint vector (action or state order) -> {side: (p [..., 3], R [..., 3, 3])}."""
        return {s: self.fk(s, a29[..., WAIST], a29[..., ARM[s]]) for s in SIDES}

    def ik(self, side: str, waist: torch.Tensor, q0: torch.Tensor, p_tgt: torch.Tensor, R_tgt: torch.Tensor,
           iters: int = 8, damping: float = 0.02, rot_weight: float = 0.2, max_step: float = 0.1) -> torch.Tensor:
        """Arm joints [..., 7] reaching (p_tgt, R_tgt) with the waist fixed, from q0 (the System 2 joints): damped least
        squares on the 6-D error [dp; rot_weight * dw] (rot_weight m/rad), each step clipped to max_step rad per joint
        and to the MJCF joint ranges; returns the best iterate (q0 included), so the error never exceeds q0's."""
        lo, hi = getattr(self, f"{side}_arm_lo").to(q0.dtype), getattr(self, f"{side}_arm_hi").to(q0.dtype)
        q = q0.clone()
        eye = torch.eye(6, dtype=q.dtype, device=q.device)
        best_q, best_err = q0.clone(), None
        for it in range(iters + 1):
            p, R, jp, jz = self.fk(side, waist, q, return_joints=True)
            e = torch.cat([p_tgt - p, rot_weight * rotation_residual(R_tgt, R)], dim=-1)
            err = e.norm(dim=-1)
            if best_err is None:
                best_err = err
            else:
                better = err < best_err
                best_q = torch.where(better.unsqueeze(-1), q, best_q)
                best_err = torch.where(better, err, best_err)
            if it == iters:
                break
            jp, jz = jp[..., 3:, :], jz[..., 3:, :]  # arm joints only
            Jp = torch.cross(jz, p.unsqueeze(-2) - jp, dim=-1)  # [..., 7, 3]
            J = torch.cat([Jp, rot_weight * jz], dim=-1).transpose(-1, -2)  # [..., 6, 7]
            dq = J.transpose(-1, -2) @ torch.linalg.solve(J @ J.transpose(-1, -2) + damping ** 2 * eye, e.unsqueeze(-1))
            q = torch.maximum(torch.minimum(q + dq.squeeze(-1).clamp(-max_step, max_step), hi), lo)
        return best_q
