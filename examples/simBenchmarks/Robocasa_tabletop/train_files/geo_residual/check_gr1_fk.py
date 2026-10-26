# (starVLA env) torch GR1Kinematics vs MuJoCo poses from dump_mujoco_fk.py; IK round trip; se3 exp/log round trip.
#   PYTHONPATH=$PWD python check_gr1_fk.py <mj_fk.npz>
import sys, numpy as np, torch
from starVLA.model.modules.geo_residual.gr1_kinematics import GR1Kinematics
from starVLA.model.modules.geo_residual.se3 import axis_angle_to_matrix, matrix_to_axis_angle, rotation_residual
d = np.load(sys.argv[1]); q = torch.tensor(d["q29"], dtype=torch.float64)
kin = GR1Kinematics().double()
w = kin.joints_to_wrists(q)
for i, s in enumerate(("left", "right")):
    p, R = w[s]
    dp = (p - torch.tensor(d["p"][:, i])).norm(dim=-1); dr = rotation_residual(R, torch.tensor(d["R"][:, i])).norm(dim=-1)
    print(f"FK {s}: pos err max {dp.max().item() * 1e3:.4f} mm, rot err max {dr.max().item():.2e} rad (n={len(dp)})")
torch.manual_seed(0)
for s in ("left", "right"):
    sl = slice(0, 7) if s == "left" else slice(7, 14)
    waist, q0 = q[:, 26:29], q[:, sl]
    p0, R0 = kin.fk(s, waist, q0)
    dp = torch.randn_like(p0) * 0.02; dw = torch.randn_like(p0) * 0.05  # 2 cm / ~3 deg targets
    pt, Rt = p0 + dp, axis_angle_to_matrix(dw) @ R0
    qi = kin.ik(s, waist, q0, pt, Rt, iters=8)
    p1, R1 = kin.fk(s, waist, qi)
    ep = (p1 - pt).norm(dim=-1) * 1e3; er = rotation_residual(Rt, R1).norm(dim=-1)
    print(f"IK {s} (8 it, 2 cm / 0.05 rad targets): pos err median {ep.median():.3f} max {ep.max():.3f} mm | rot err median "
          f"{er.median():.2e} max {er.max():.2e} rad | |dq| median {(qi - q0).norm(dim=-1).median():.3f} rad")
v = torch.randn(1000, 3, dtype=torch.float64) * 1.5  # angles beyond pi map to the equivalent rotation -> compare matrices
Rv = axis_angle_to_matrix(v)
print("exp(log(R)) round trip max err", (axis_angle_to_matrix(matrix_to_axis_angle(Rv)) - Rv).abs().max().item(),
      "| log(exp(v)) for |v| < 3.1", (lambda u: (matrix_to_axis_angle(axis_angle_to_matrix(u)) - u).abs().max().item())(v[v.norm(dim=-1) < 3.1]),
      "| tiny", (matrix_to_axis_angle(axis_angle_to_matrix(v * 1e-7)) - v * 1e-7).abs().max().item())
