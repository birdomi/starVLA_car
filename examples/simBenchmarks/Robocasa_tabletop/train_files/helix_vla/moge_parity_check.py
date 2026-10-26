# MoGe encoder parity (CPU): starVLA MoGeEncoderBackbone (timm DINOv2 + fixed pos-embed resample + output projections)
# vs MoGe's own code (DINOv2 package of a microsoft/MoGe checkout, DINOv2Encoder.forward logic), on random + real frames.
#   PYTHONPATH=/mnt/DATA2/yyg/starVLA python moge_parity_check.py <MoGe checkout> <model.pt> [mp4 ...]
# 10-22: moge-3-vitl rel 5e-7 (CLS exact); moge-2-vitb-normal see memory moge3-vitl-eye.
import sys, time, torch, torch.nn as nn, torch.nn.functional as F
src, ck_path, videos = sys.argv[1], sys.argv[2], sys.argv[3:]
from starVLA.model.modules.dual_system.moge_encoder import MoGeEncoderBackbone
sys.path.insert(0, src + "/moge/model/modules")   # 'dinov2' as a top-level package (relative imports inside)
from dinov2.models import vision_transformer as vits
torch.manual_seed(0)
m = MoGeEncoderBackbone(ck_path, input_size=(256, 256), grid=(16, 16), trainable=False)
print(m.trainable_desc, "| tokens", m.num_tokens, "feat", m.feat_dim, "prefix", m.num_prefix, m.prefix_dim,
      "| params M", round(sum(p.numel() for p in m.parameters()) / 1e6, 2), "| requires_grad any", any(p.requires_grad for p in m.parameters()))
m.train(); assert not m.training
ck = torch.load(ck_path, map_location="cpu", weights_only=True); ecfg = ck["model_config"]["encoder"]
sd = {k[len("encoder."):]: v for k, v in ck["model"].items() if k.startswith("encoder.")}
bb = {k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")}
arch = {"dinov2_vits14": "vit_small", "dinov2_vitb14": "vit_base", "dinov2_vitl14": "vit_large"}[ecfg["backbone"]]
ref = vits.__dict__[arch](img_size=518, patch_size=14, init_values=1.0, ffn_layer="mlp", block_chunks=0, num_register_tokens=0,
                          interpolate_antialias=False, interpolate_offset=0.1)
print("ref load:", ref.load_state_dict(bb, strict=True)); ref.eval()
proj = nn.ModuleList([nn.Conv2d(ref.embed_dim, ecfg["dim_out"], 1) for _ in ecfg["intermediate_layers"]])
proj.load_state_dict({k[len("output_projections."):]: v for k, v in sd.items() if k.startswith("output_projections.")}); proj.eval()
mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

def reference(img):  # MoGe DINOv2Encoder.forward(image, 16, 16, return_class_token=True)
    x14 = (F.interpolate(img, (16 * 14, 16 * 14), mode="bilinear", align_corners=False, antialias=True) - mean) / std
    feats = ref.get_intermediate_layers(x14, n=ecfg["intermediate_layers"], return_class_token=True)
    return sum(p(f.permute(0, 2, 1).unflatten(2, (16, 16))) for p, (f, c) in zip(proj, feats)), feats[-1][1]

imgs = [("random", torch.rand(2, 3, 256, 256))]
for v in videos:
    import decord
    vr = decord.VideoReader(v); fr = vr.get_batch([0, len(vr) // 2]).asnumpy()  # [2, H, W, 3] uint8
    x = torch.from_numpy(fr).permute(0, 3, 1, 2).float() / 255.0
    imgs.append((v.split("/")[-4][:40] + "/" + v.split("/")[-1], F.interpolate(x, (256, 256), mode="bilinear", align_corners=False, antialias=True)))
for name, img in imgs:
    with torch.no_grad():
        t = time.time(); yr, cr = reference(img); tr = time.time() - t
        t = time.time(); out = m(img); tt = time.time() - t
    yr = yr.flatten(2).transpose(1, 2)
    nrm = out.tokens.norm(dim=-1)
    print(f"[{name}] tokens {tuple(out.tokens.shape)} prefix {tuple(out.prefix.shape)} valid {bool(out.valid.all())} | "
          f"max|d| {(yr - out.tokens).abs().max().item():.2e} rel {((yr - out.tokens).norm() / yr.norm()).item():.2e} "
          f"cls max|d| {(cr - out.prefix[:, 0]).abs().max().item():.2e} | token norm median {nrm.median().item():.1f} "
          f"max {nrm.max().item():.1f} (#>3x median {(nrm > 3 * nrm.median()).sum().item()}) | cpu ref {tr:.1f}s ours {tt:.1f}s")
try:
    m(torch.rand(1, 3, 224, 224)); print("WRONG INPUT SIZE ACCEPTED (bad)")
except ValueError as err:
    print("wrong input size -> ValueError OK:", str(err)[:80])
