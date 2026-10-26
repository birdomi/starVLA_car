# Direct action splatting

`ActionSplatHead` splats action vectors onto the future time grid. Each point
stores action[D], a time offset, radius, and importance: **D+3 parameters**.
There is no Gaussian covariance, probability density, NLL or time-path search.

```text
One shared bank of K action points + point identity queries
                    ↓ point encoder; shared K-token transformer
image/language VLM features + current state → cross-attention
                    ↓ predict offsets to the point bank
K action points → triangular softmax splat → one action chunk [B,H,D]
```

The splat at time t uses `k_i(t) = max(0, 1 - abs(t-tau_i)/r_i)`:

```text
action(t) = sum_i k_i(t) * exp(z_i / temperature) * action_i
            / sum_i k_i(t) * exp(z_i / temperature)
```

The implementation computes these weights in log space and FP32 under AMP.
Contributions outside a point's footprint are exactly zero. The small time grid
uses tensor contractions rather than a custom CUDA rasterizer. This is a 1D
adaptation of [softmax splatting](https://openaccess.thecvf.com/content_CVPR_2020/html/Niklaus_Softmax_Splatting_for_Video_Frame_Interpolation_CVPR_2020_paper.html),
equivalently learned local time interpolation; it is not a graphics renderer.

## Coverage and learning

Anchor i starts at `(i+0.5)/K`. Its learned time offset is bounded by `0.25/K`
by default. Radius is strictly larger than `0.8/K`, ensuring every time in
`[0,1]` has at least one supporting point. Initial radius is `1.5/K`. This
prevents holes and NaNs at endpoints. It limits large time warps; time remains
relative to the fixed-duration action chunk. There is no episode phase input.

The output chunk is supervised with action reconstruction loss (default L1) plus
`0.1 * L1(predicted first difference - target first difference)`. The final
action loss backpropagates through point values, time offsets, radii and
importance. Triangular splatting is piecewise differentiable, including through
the normalization; there is no `no_grad` path inside the training splatter.

There is one point bank and one decoder output. No candidate branches, WTA,
score head or chunk selection remain. Overlapping point values blend directly
into the output action. Shared framework WTA settings are unused by this head.

The optional head-level `action_mask` uses True=valid and supports `[B,H]` or
`[B,H,D]`. The existing framework supplies no mask; endpoint padding remains
supervised, matching the current dataset path. Task success is not directly
optimized without rollout feedback. Rotation/action semantics remain those of
the existing normalized action representation.

## Configuration

Use framework `QwenGR00T_Regressor` with `action_model.head_type: action_splat`.
Existing Gaussian and regression heads/checkpoints remain available.

Ready EgoDex configuration:
`examples/realRobots/EgoDex/train_files/starvla_qwengr00t_actionsplat_hf15_rel_part1_unfreeze_50k.yaml`

Defaults: K=32 points, one chunk, D=48, H=30, batch 32 × 4 accumulation ×
4 GPUs = 512. VLM trains with AdamW at 2e-5, head at 1e-4.
`moe_expert_layers` must be 0;
multi-window and past-action packing are not supported in this first version.

```bash
torchrun --nproc_per_node=4 --master_port=29661 \
  starVLA/training/train_starvla.py \
  --config_yaml examples/realRobots/EgoDex/train_files/starvla_qwengr00t_actionsplat_hf15_rel_part1_unfreeze_50k.yaml
```

## Verification

```bash
python -m unittest discover -s tests -p test_action_splat_head.py -v

python examples/modelExtensions/ActionSplat/fit_actions.py \
  --actions playground/gaussian_action_probe/fit_k32.npz \
  --num-points 8 32 --steps 500 --out playground/action_splat_probe
```

The fit tool also accepts normalized `[B,H,D]` `.npy` arrays. It fits a separate
point bank to each complete demo chunk, with target-based initialization. This
checks representation fit only, not shared-bank policy generalization or task
success. Predictions, points, targets and metrics are saved for inspection.

## Adaptive point count (3DGS-style densify / prune)

Set `num_splat_points_max` above `num_splat_points` to let the head grow and
shrink its point population during training, in the spirit of 3D Gaussian
Splatting's adaptive density control. The bank is allocated at capacity
`K_max`; only `K` slots start active (regular grid). Tensor shapes never
change, so DDP / DeepSpeed / optimizer state stay valid; inactive slots are
masked out of the splat, and any time slot that no active point supports
falls back to its nearest active point (no NaNs, no holes).

Statistics accumulated every training step (all-reduced across ranks):

- per-point **gradient**: batch-mean L2 norm of the loss gradient w.r.t. the
  per-sample decoded point parameters (the analogue of 3DGS's view-space
  positional gradient);
- per-point **usage**: mean splat weight over batch and time (uniform coverage
  gives `1 / K_active`).

Every `splat_densify_interval` optimizer steps inside
`[splat_densify_from, splat_densify_until]`:

1. **prune** active points with usage below
   `splat_prune_usage_ratio / K_active` (never below `splat_min_points`);
2. **densify** the active points whose gradient exceeds the
   `splat_densify_quantile` quantile (at most `splat_max_new_per_interval`,
   bounded by free slots): a point with radius above
   `splat_split_radius_ratio / K_active` is **split** into two children half a
   radius apart, both shrunk by `splat_split_shrink`; otherwise it is
   **cloned** with a jittered position. Children copy the parent's action
   values, importance and identity query (plus small noise); their time
   offsets reset and their Adam moments are zeroed.

`splat_importance_reset_interval` (0 = off) periodically resets importance,
like 3DGS's opacity reset. Anchors and the active mask are checkpointed;
statistics are not. Logged metrics: `splat/active`, `splat/radius_mean`,
`splat/n_split`, `splat/n_clone`, `splat/n_prune`.

Ready configuration (8 -> up to 64 points):
`examples/realRobots/EgoDex/train_files/starvla_qwengr00t_actionsplat_adaptive_hf15_rel_part1_unfreeze_50k.yaml`
