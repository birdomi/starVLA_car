# (robocasa env) Build a real RoboCasa GR1 eval env, print the per-part joint order robosuite uses for robot0_left /
# robot0_right / robot0_torso, set random arm + waist joints, mj forward, and dump the eef poses relative to the robot base
# -> npz for check_gr1_fk.py (starVLA env).  python dump_mujoco_fk.py <out.npz>
import sys, numpy as np, gymnasium as gym
import robocasa, robosuite  # noqa: F401
from robocasa.utils.gym_utils import GrootRoboCasaEnv  # noqa: F401  (registers gr1_unified/*)
env = gym.make("gr1_unified/PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_Env", enable_render=False)
env.reset(seed=0)
u = env.unwrapped
rs = getattr(u, "env", u)
while not hasattr(rs, "robots") and hasattr(rs, "env"):
    rs = rs.env
robot = rs.robots[0]; sim = rs.sim
print("parts:", {k: [sim.model.joint_id2name(i) for i in v] for k, v in robot._ref_joints_indexes_dict.items()})
parts = {k: [sim.model.joint_id2name(i) for i in v] for k, v in robot._ref_joints_indexes_dict.items()}
left, right, torso = parts["left"], parts["right"], parts["torso"]
rng = np.random.default_rng(0)
def rnd(names):
    out = []
    for n in names:
        lo, hi = sim.model.jnt_range[sim.model.joint_name2id(n)]
        out.append(rng.uniform(lo, hi) * 0.8 if hi > lo else 0.0)
    return np.array(out)
base = "robot0_base"
Q, P, R = [], [], []
for _ in range(64):
    q = {"left": rnd(left), "right": rnd(right), "torso": rnd(torso)}
    for k, names in (("left", left), ("right", right), ("torso", torso)):
        for n, v in zip(names, q[k]):
            sim.data.qpos[sim.model.get_joint_qpos_addr(n)] = v
    sim.forward()
    bid = sim.model.body_name2id(base); pb, Rb = sim.data.xpos[bid].copy(), sim.data.xmat[bid].reshape(3, 3).copy()
    ps, Rs = [], []
    for eef in ("robot0_left_eef", "robot0_right_eef"):
        i = sim.model.body_name2id(eef)
        ps.append(Rb.T @ (sim.data.xpos[i] - pb)); Rs.append(Rb.T @ sim.data.xmat[i].reshape(3, 3))
    Q.append(np.concatenate([q["left"], q["right"], np.zeros(12), q["torso"]])); P.append(ps); R.append(Rs)
np.savez(sys.argv[1], q29=np.array(Q), p=np.array(P), R=np.array(R), left=left, right=right, torso=torso)
print("saved", np.array(Q).shape)
