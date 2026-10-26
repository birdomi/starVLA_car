# Gaussian action primitives

An explicit, trainable Gaussian bank in **action + relative time**. Observation
features deform each primitive's center, scale, slope and mass. The resulting
field represents a set of possible actions. Inference selects a dense path
through that set, rather than taking the mean of different action modes.

This borrows the primitive-field idea from Gaussian splatting. It is an additive
Gaussian **density** field / conditional GMM, not a graphics renderer: no camera
projection, visibility ordering, or alpha compositing. The first version uses a
fixed bank size; it has no adaptive split/prune or rollout-based success reward.

```text
shared Gaussian bank ── primitive encoder + identity queries ──┐
image + language ── VLM features ─────────────────────────────┤ cross-attention
robot state ── state encoder ────────────────────────────────┘
              ↓ observation-conditioned offsets
bank + offsets → Gaussian action-time field
              ↓ evaluate each relative time
mixture density → continuous mode-path selection → [B,H,D] actions
```

Each primitive factors as:

```text
t | k       ~ Normal(time_center[k], time_width[k]^2)
a | t,k,obs ~ Normal(u[k] + slope[k] * (t - time_center[k]), diag(std[k]^2))
p(a|t,obs)  = sum_k p(k|t,obs) * p(a|t,k,obs)
```

This is a valid joint action-time Gaussian with action-time cross-covariance.
The bank has `K * (3 + 3*D)` raw parameters; time is normalized to `[0,1]`
within a fixed-duration future chunk, not episode progress or diffusion time.

Training minimizes conditional demo NLL per valid action dimension. Positive
scale floors prevent variance collapse; density math runs in FP32. There is no
action-mean reconstruction loss. Successful demonstrations provide imitation
targets; task success itself is not optimized without environment feedback.

Inference retains up to eight component-center candidates per time step, scores
each against the entire mixture, and uses dynamic programming with an action
change penalty to choose one path. This is an approximate mode path, not a global
continuous MAP solution or a guarantee of dynamically feasible robot motion.

## Use in starVLA

Framework: `QwenGR00T_Regressor`. Set `framework.action_model.head_type: gaussian`.
The existing framework handles VLM features, state, checkpointing, and action
normalization. Existing regression configurations keep their original head.

Ready configuration:
`examples/realRobots/EgoDex/train_files/starvla_qwengr00t_gaussian_hf15_rel_part1_unfreeze_50k.yaml`

Defaults: 32 shared primitives, 48 action dimensions, 30 future slots,
one path, AdamW. `num_hypotheses` must be 1; WTA and multi-window mode are not
implemented for this head. Its loss is NLL regardless of inherited regressor
`loss_type` defaults. Action units and relative/absolute semantics come from
the dataset, unchanged by the head.

```bash
torchrun --nproc_per_node=4 --master_port=29631 \
  starVLA/training/train_starvla.py \
  --config_yaml examples/realRobots/EgoDex/train_files/starvla_qwengr00t_gaussian_hf15_rel_part1_unfreeze_50k.yaml
```

The head accepts an optional `action_mask` (`True` = valid). The current framework
does not pass one, so normal training supervises every returned slot, including
the dataloader's endpoint padding. The direct fit below uses complete chunks.

## Check representation fit

```bash
python examples/modelExtensions/GaussianActions/fit_actions.py \
  --egodex-root /mnt/DATA2/yyg/egodex/egodex_lerobot_part1_epsplit \
  --max-chunks 16 --num-gaussians 8 32 --steps 500 \
  --out playground/gaussian_action_probe

python -m unittest discover -s tests -p test_gaussian_action_head.py -v
```

Alternatively pass `--actions normalized_chunks.npy` with finite `[B,H,D]`
targets. The fit tool initializes and optimizes a separate bank for each demo
chunk. It writes targets, fitted parameters, extracted actions, and metrics.
This tests representation capacity only; it does not train the shared-bank
observation policy or measure held-out performance / task success.
