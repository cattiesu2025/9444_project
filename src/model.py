"""Shared PMT ViT and BN neck. Architecture audited against hulu88/PMT.

Independent implementation with upstream-compatible backbone parameter names.
See docs/baseline.md for provenance and intentional numerical safeguards.
"""
from pathlib import Path
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class DropPath(nn.Module):
    def __init__(self, probability=0.):
        super().__init__()
        self.probability = probability

    def forward(self, x):
        if not self.training or self.probability == 0:
            return x
        keep = 1 - self.probability
        mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
        return x * mask / keep


class Attention(nn.Module):
    def __init__(self, dim, heads, dropout, attention_dropout):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attention_dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        b, n, d = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4).unbind(0)
        weights = ((q @ k.transpose(-2, -1)) / math.sqrt(d // self.heads)).softmax(-1)
        x = (self.attn_drop(weights) @ v).transpose(1, 2).reshape(b, n, d)
        return self.proj_drop(self.proj(x))


class Mlp(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.fc1 = nn.Linear(dim, 4 * dim)
        self.fc2 = nn.Linear(4 * dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(F.gelu(self.fc1(x)))))


class Block(nn.Module):
    def __init__(self, dim, heads, dropout, attention_dropout, drop_path):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)  # upstream epsilon = 1e-5
        self.attn = Attention(dim, heads, dropout, attention_dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, dropout)
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        return x + self.drop_path(self.mlp(self.norm2(x)))


class PatchEmbed(nn.Module):
    def __init__(self, image_size, patch_size, stride, dim):
        super().__init__()
        self.image_size = tuple(image_size)
        self.grid_size = tuple((s - patch_size) // stride + 1 for s in image_size)
        self.proj = nn.Conv2d(3, dim, patch_size, stride=stride)

    def forward(self, images):
        if tuple(images.shape[-2:]) != self.image_size:
            raise ValueError(f"Expected images of size {self.image_size}, got {images.shape[-2:]}")
        patches = self.proj(images)
        grid = tuple(patches.shape[-2:])
        return patches.flatten(2).transpose(1, 2), grid


class VisionTransformer(nn.Module):
    def __init__(self, image_size=(256, 128), patch_size=16, patch_stride=12,
                 dim=768, depth=12, heads=12, dropout=.03, attention_dropout=0., drop_path=.1, activation_checkpointing=False):
        super().__init__()
        self.activation_checkpointing = activation_checkpointing
        self.patch_embed = PatchEmbed(image_size, patch_size, patch_stride, dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, math.prod(self.patch_embed.grid_size) + 1, dim))
        self.pos_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(dim, heads, dropout, attention_dropout, p)
                                     for p in torch.linspace(0, drop_path, depth).tolist()])
        self.norm = nn.LayerNorm(dim)
        self.apply(self._initialize)
        nn.init.trunc_normal_(self.cls_token, std=.02)
        nn.init.trunc_normal_(self.pos_embed, std=.02)

    @staticmethod
    def _initialize(module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            nn.init.normal_(module.weight, std=math.sqrt(2 / (module.kernel_size[0] * module.kernel_size[1] * module.out_channels)))
            nn.init.zeros_(module.bias)

    def forward(self, images):
        patches, grid = self.patch_embed(images)
        tokens = torch.cat((self.cls_token.expand(images.shape[0], -1, -1), patches), dim=1)
        tokens = self.pos_drop(tokens + self.pos_embed)
        for block in self.blocks:
            tokens = checkpoint(block, tokens, use_reentrant=False, preserve_rng_state=True) if self.activation_checkpointing and self.training and torch.is_grad_enabled() else block(tokens)
        tokens = self.norm(tokens)
        return tokens[:, 0], tokens[:, 1:], grid

    def load_pretrained(self, path):
        state = torch.load(Path(path), map_location="cpu", weights_only=True)
        for key in ("model", "state_dict"):
            if key in state:
                state = state[key]
        state = {k.removeprefix("module."): v for k, v in state.items()
                 if not k.removeprefix("module.").startswith(("head.", "head_dist."))}
        if "pos_embed" in state and state["pos_embed"].shape != self.pos_embed.shape:
            pos = state["pos_embed"]
            side = math.isqrt(pos.shape[1] - 1)
            if side * side != pos.shape[1] - 1:
                raise ValueError("Pretrained positional embedding must contain one CLS and a square patch grid")
            patches = pos[:, 1:].reshape(1, side, side, -1).permute(0, 3, 1, 2)
            patches = F.interpolate(patches, size=self.patch_embed.grid_size, mode="bilinear", align_corners=False)
            state["pos_embed"] = torch.cat((pos[:, :1], patches.flatten(2).transpose(1, 2)), dim=1)
        self.load_state_dict(state, strict=True)  # never silently leave an uninitialized backbone


class PMT(nn.Module):
    def __init__(self, num_classes, model_config=None, pretrained=None):
        super().__init__()
        cfg = model_config or {}
        dim = cfg.get("global_feature_dim", 768)
        self.base = VisionTransformer(image_size=cfg.get("image_size", (256, 128)),
            patch_size=cfg.get("patch_size", 16), patch_stride=cfg.get("patch_stride", 12),
            dim=dim, depth=cfg.get("depth", 12), heads=cfg.get("heads", 12),
            dropout=cfg.get("dropout", .03), attention_dropout=cfg.get("attention_dropout", 0.),
            drop_path=cfg.get("drop_path", .1), activation_checkpointing=cfg.get("activation_checkpointing", False))
        self.bottleneck = nn.BatchNorm1d(dim)
        self.bottleneck.bias.requires_grad_(False)
        self.classifier = nn.Linear(dim, num_classes, bias=False)
        nn.init.normal_(self.classifier.weight, std=.001)
        if pretrained is not None:
            self.base.load_pretrained(pretrained)

    def forward(self, images):
        global_feature, patches, grid = self.base(images)
        bn_feature = self.bottleneck(global_feature)
        return {"global_feature": global_feature, "global_logits": self.classifier(bn_feature),
                "retrieval_feature": F.normalize(bn_feature, dim=1),
                "patch_features": patches, "patch_grid": grid,
                "local_features": None, "local_logits": None}
