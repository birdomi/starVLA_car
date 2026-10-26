"""SO(3) / SE(3) helpers in torch (batched over leading dims, float32/64).

Conventions: quaternions are MuJoCo order (w, x, y, z); rotation vectors (axis-angle) are the so(3) log of a rotation
matrix; residuals are applied in the robot BASE frame: p' = p + dp, R' = exp(dw) @ R.
"""

import torch


def quat_wxyz_to_matrix(q: torch.Tensor) -> torch.Tensor:
    q = q / q.norm(dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dim=-1).reshape(q.shape[:-1] + (3, 3))


def skew(v: torch.Tensor) -> torch.Tensor:
    x, y, z = v.unbind(-1)
    o = torch.zeros_like(x)
    return torch.stack([o, -z, y, z, o, -x, -y, x, o], dim=-1).reshape(v.shape[:-1] + (3, 3))


def axis_angle_to_matrix(v: torch.Tensor) -> torch.Tensor:
    """Rodrigues exp map; Taylor terms below 1e-4 rad (exact to fp32 there)."""
    th2 = (v * v).sum(-1, keepdim=True)[..., None]  # [..., 1, 1]
    th = th2.clamp_min(1e-24).sqrt()
    small = th2 < 1e-8
    a = torch.where(small, 1 - th2 / 6, torch.sin(th) / th)
    b = torch.where(small, 0.5 - th2 / 24, (1 - torch.cos(th)) / th2.clamp_min(1e-24))
    K = skew(v)
    eye = torch.eye(3, dtype=v.dtype, device=v.device).expand(K.shape)
    return eye + a * K + b * (K @ K)


def matrix_to_axis_angle(R: torch.Tensor) -> torch.Tensor:
    """so(3) log map, stable near 0 and near pi."""
    tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos = ((tr - 1) / 2).clamp(-1.0, 1.0)
    th = torch.acos(cos)
    vee = torch.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]], dim=-1)
    sin = torch.sin(th)
    small = th < 1e-4
    scale = torch.where(small, 0.5 + th * th / 12, th / (2 * sin.clamp_min(1e-12)))
    out = vee * scale[..., None]
    # near pi: axis from the largest diagonal of (R + I) / 2, sign from vee
    near_pi = th > 3.14159265 - 1e-3
    if bool(near_pi.any()):
        B = (R + torch.eye(3, dtype=R.dtype, device=R.device)) / 2
        diag = torch.diagonal(B, dim1=-2, dim2=-1).clamp_min(0)
        i = diag.argmax(-1, keepdim=True)
        col = torch.gather(B, -1, i[..., None, :].expand(B.shape[:-1] + (1,))).squeeze(-1)
        axis = col / col.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        sgn = torch.sign((axis * vee).sum(-1, keepdim=True))
        sgn = torch.where(sgn == 0, torch.ones_like(sgn), sgn)
        out = torch.where(near_pi[..., None], axis * sgn * th[..., None], out)
    return out


def matrix_to_rot6d(R: torch.Tensor) -> torch.Tensor:
    """First two columns, [..., 6] (Zhou et al. 2019 continuous rotation representation)."""
    return torch.cat([R[..., :, 0], R[..., :, 1]], dim=-1)


def rotation_residual(R_target: torch.Tensor, R_src: torch.Tensor) -> torch.Tensor:
    """dw with exp(dw) @ R_src = R_target (base frame)."""
    return matrix_to_axis_angle(R_target @ R_src.transpose(-1, -2))


def compose_residual(p: torch.Tensor, R: torch.Tensor, dp: torch.Tensor, dw: torch.Tensor):
    return p + dp, axis_angle_to_matrix(dw) @ R
