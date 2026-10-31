"""AGW / CAJ external CNN baselines: two-stream ResNet-50, losses and augmentations.

Ported from mangye16/Cross-Modal-Re-ID-baseline@e817a2672c1e14e8802617acb980a79520ef9545
(model.py, loss.py, ICCV21_CAJ/{ChannelAug,data_loader,loss,train_ext}.py), MIT License,
Copyright (c) 2020 mangye16; notice in third_party_licenses/. Upstream behaviour is kept,
including Non_local inter_channels=1 and the CAJ squared-difference loss; see docs/agw_caj.md.
"""
import copy
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from .data import MODALITY_CODE, IMAGENET_MEAN, IMAGENET_STD, collate_batch, check_pmt_layout

BACKBONE = "ResNet-50 AGW two-stream"
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)  # Upstream ChannelRandomErasing fill value.
UPSTREAM_RESNET50_SHA256 = "19c8e3572231adff6824a2da93fd67b5986919a2e65f8b6007eab4edee220097"


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class NonLocal(nn.Module):
    """Upstream Non_local: dot-product affinity divided by N; W-branch BN starts at zero."""
    def __init__(self, channels):
        super().__init__()
        inter = 1  # Upstream reduc_ratio // reduc_ratio.
        self.g = nn.Conv2d(channels, inter, 1)
        self.theta = nn.Conv2d(channels, inter, 1)
        self.phi = nn.Conv2d(channels, inter, 1)
        self.W = nn.Sequential(nn.Conv2d(inter, channels, 1), nn.BatchNorm2d(channels))
        nn.init.zeros_(self.W[1].weight)
        nn.init.zeros_(self.W[1].bias)

    def forward(self, x):
        b = x.size(0)
        g = self.g(x).flatten(2).transpose(1, 2)
        theta = self.theta(x).flatten(2).transpose(1, 2)
        phi = self.phi(x).flatten(2)
        f = theta @ phi
        y = (f / f.size(-1)) @ g
        y = y.transpose(1, 2).reshape(b, -1, *x.shape[2:])
        return self.W(y) + x


def load_resnet50_state(path):
    """The upstream resnet50-19c8e357.pth uses the legacy tar format, which weights_only cannot read.
    Full unpickling is allowed only for that exact published file."""
    from .training import sha256
    if sha256(path) == UPSTREAM_RESNET50_SHA256:
        return torch.load(path, map_location="cpu", weights_only=False)
    return torch.load(path, map_location="cpu", weights_only=True)


def _stem(resnet):
    return nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)


class AGW(nn.Module):
    """Modality-specific stems (conv1..maxpool), shared layer1-4 with Non-local, GeM, BN neck.

    forward(images, modalities) routes each row by its modality code, so mixed and
    single-modality batches share one path. Retrieval uses the L2-normalized BN feature.
    """
    modality_specific = True

    def __init__(self, num_classes, cfg, pretrained=None):
        super().__init__()
        from torchvision.models import resnet50
        resnet = resnet50(weights=None)
        if pretrained is not None:
            resnet.load_state_dict(load_resnet50_state(pretrained), strict=True)
        # last_conv_stride=1: torchvision puts the stride on conv2 and the downsample conv.
        resnet.layer4[0].conv2.stride = (1, 1)
        resnet.layer4[0].downsample[0].stride = (1, 1)
        self.visible = _stem(copy.deepcopy(resnet))
        self.thermal = _stem(copy.deepcopy(resnet))
        self.layers = nn.ModuleList([resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4])
        # Upstream non_layers=[0,2,3,0]: after the last 2 blocks of layer2, last 3 of layer3.
        self.non_local_after = {1: (2, 3), 2: (3, 4, 5)} if cfg.get("non_local", True) else {}
        self.non_local = nn.ModuleDict({f"{layer}_{block}": NonLocal(256 * 2 ** layer)
                                        for layer, blocks in self.non_local_after.items() for block in blocks})
        self.gem_p = cfg.get("gem_p", 3.0)
        dim = cfg["global_feature_dim"]
        self.bottleneck = nn.BatchNorm1d(dim)
        nn.init.normal_(self.bottleneck.weight, 1.0, 0.01)
        nn.init.zeros_(self.bottleneck.bias)
        self.bottleneck.bias.requires_grad_(False)
        self.classifier = nn.Linear(dim, num_classes, bias=False)
        nn.init.normal_(self.classifier.weight, 0, 0.001)

    def forward(self, images, modalities):
        rgb = modalities == MODALITY_CODE["rgb"]
        parts, order = [], []
        for mask, stem in ((rgb, self.visible), (~rgb, self.thermal)):
            if mask.any():
                parts.append(stem(images[mask]))
                order.append(mask.nonzero().squeeze(1))
        x = torch.cat(parts)[torch.argsort(torch.cat(order))]
        for layer_index, layer in enumerate(self.layers):
            for block_index, block in enumerate(layer):
                x = block(x)
                if block_index in self.non_local_after.get(layer_index, ()):
                    x = self.non_local[f"{layer_index}_{block_index}"](x)
        if self.gem_p:
            pooled = (x.flatten(2).pow(self.gem_p).mean(-1) + 1e-12).pow(1 / self.gem_p)
        else:
            pooled = x.mean((2, 3))
        feature = self.bottleneck(pooled)
        return {"pool_feature": pooled, "logits": self.classifier(feature),
                "retrieval_feature": F.normalize(feature, dim=1)}


# ---------------------------------------------------------------------------
# Losses (FP32)
# ---------------------------------------------------------------------------

def pairwise_distance(x):
    square = x.pow(2).sum(1, keepdim=True)
    return (square + square.t() - 2 * x @ x.t()).clamp(min=1e-12).sqrt()


def softmax_weights(dist, mask, masked_max=False):
    """Upstream takes the row max over dist * mask, so masked-out zeros join it; with large distances every
    negative weight then underflows and closest_negative is ~0. masked_max=True (CAJ_fix) uses valid entries only,
    and sets masked entries to -inf before exp so they cannot overflow."""
    if masked_max:
        valid = dist.masked_fill(mask == 0, float("-inf"))
        diff = valid - valid.max(dim=1, keepdim=True)[0]
    else:
        diff = dist - torch.max(dist * mask, dim=1, keepdim=True)[0]
    z = torch.sum(torch.exp(diff) * mask, dim=1, keepdim=True) + 1e-6
    return torch.exp(diff) * mask / z


def weighted_extremes(features, labels, alpha=1.0, masked_max=False):
    """Softmax-weighted furthest positive / closest negative per anchor (self-match included, as upstream)."""
    dist = pairwise_distance(features)
    same = labels[:, None].eq(labels[None, :]).float()
    dist_ap, dist_an = dist * same, dist * (1 - same)
    furthest_positive = (dist_ap * softmax_weights(dist_ap * alpha, same, masked_max)).sum(1)
    closest_negative = (dist_an * softmax_weights(-dist_an * alpha, 1 - same, masked_max)).sum(1)
    return furthest_positive, closest_negative


def wrt(features, labels):
    """AGW weighted regularized triplet: SoftMargin(closest_negative - furthest_positive)."""
    positive, negative = weighted_extremes(features, labels)
    return F.soft_margin_loss(negative - positive, torch.ones_like(positive))


def adp(features, labels, alpha=1.0, gamma=1.0, square=True, masked_max=False):
    """CAJ TripletLoss_ADP; square=True is the enhanced squared difference used for SYSU."""
    positive, negative = weighted_extremes(features, labels, alpha, masked_max)
    if not square:
        return F.soft_margin_loss(gamma * (negative - positive), torch.ones_like(positive))
    diff = ((positive - negative).pow(2) * gamma).clamp(max=88)
    target = torch.where(positive > negative, -1.0, 1.0)
    return F.soft_margin_loss(diff, target)


class AGWLoss:
    """ID cross-entropy over every row + WRT (AGW) or ADP (CAJ) on the pre-BN pooled feature."""
    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, output, labels):
        features, logits = output["pool_feature"].float(), output["logits"].float()
        identity = F.cross_entropy(logits, labels)
        if self.cfg["method"] == "ID+WRT":
            triplet = wrt(features, labels)
        elif self.cfg["method"] == "ID+ADP":
            triplet = adp(features, labels, self.cfg["alpha"], self.cfg["gamma"], self.cfg["square"],
                          self.cfg.get("masked_max", False))
        else:
            raise ValueError(f"Unknown AGW/CAJ loss {self.cfg['method']}")
        return {"total": identity + triplet, "id": identity, "triplet": triplet}


# ---------------------------------------------------------------------------
# Optimizer and step schedule
# ---------------------------------------------------------------------------

def make_sgd(model, cfg):
    """Upstream groups: backbone at lr x0.1; BN neck and classifier at lr."""
    head = {id(p) for module in (model.bottleneck, model.classifier) for p in module.parameters()}
    groups = [[p for p in model.parameters() if p.requires_grad and id(p) not in head],
              [p for p in model.bottleneck.parameters() if p.requires_grad],
              list(model.classifier.parameters())]
    factors = [cfg["backbone_lr_factor"], 1.0, 1.0]
    return torch.optim.SGD([{"params": g, "lr": cfg["learning_rate"] * f} for g, f in zip(groups, factors)],
                           lr=cfg["learning_rate"], momentum=cfg["momentum"],
                           weight_decay=cfg["weight_decay"], nesterov=cfg["nesterov"])


class StepWarmupSchedule:
    """Upstream adjust_learning_rate(epoch), zero-based, called before each epoch:
    linear warmup (epoch+1)/warmup, then x0.1 at each milestone."""
    def __init__(self, optimizer, cfg):
        self.optimizer = optimizer
        self.factors = [g["lr"] / cfg["learning_rate"] for g in optimizer.param_groups]
        self.base = cfg["learning_rate"]
        self.warmup = cfg["warmup_epochs"]
        self.milestones = cfg["lr_milestones"]
        self.epoch = 0

    def lr_at(self, epoch):
        if epoch < self.warmup:
            return self.base * (epoch + 1) / self.warmup
        return self.base * 0.1 ** sum(epoch >= m for m in self.milestones)

    def step(self, epoch):
        self.epoch = epoch
        for group, factor in zip(self.optimizer.param_groups, self.factors):
            group["lr"] = self.lr_at(epoch) * factor

    def state_dict(self):
        return {"epoch": self.epoch, "factors": self.factors}

    def load_state_dict(self, state):
        if state["factors"] != self.factors:
            raise ValueError("Parameter-group learning-rate factors changed since checkpoint")
        self.step(state["epoch"])


# ---------------------------------------------------------------------------
# Augmentations: called as transform(img, gray=False) by data.SYSUTrainData
# ---------------------------------------------------------------------------

class ChannelRandomErasing:
    """Upstream ChannelRandomErasing on a normalized tensor, filled with CIFAR means."""
    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=CIFAR_MEAN):
        self.probability, self.sl, self.sh, self.r1, self.mean = probability, sl, sh, r1, mean

    def __call__(self, img):
        if random.uniform(0, 1) > self.probability:
            return img
        for _ in range(100):
            area = img.size(1) * img.size(2)
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < img.size(2) and h < img.size(1):
                x1 = random.randint(0, img.size(1) - h)
                y1 = random.randint(0, img.size(2) - w)
                for c in range(img.size(0)):
                    img[c, x1:x1 + h, y1:y1 + w] = self.mean[c]
                return img
        return img


def _replicate(img, channel):
    img[:] = img[channel].clone()
    return img


def _to_gray(img):
    gray = 0.2989 * img[0] + 0.5870 * img[1] + 0.1140 * img[2]
    img[:] = gray
    return img


class ChannelExchange:
    """CAJ ChannelExchange(gray=2): randint(0, 2) always picks one channel to replicate."""
    def __init__(self, gray=2):
        self.gray = gray

    def __call__(self, img):
        idx = random.randint(0, self.gray)
        return _replicate(img, idx) if idx < 3 else _to_gray(img)


class ChannelAdapGray:
    """CAJ ChannelAdapGray: 3/4 replicate one channel; else grayscale with this probability."""
    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, img):
        idx = random.randint(0, 3)
        if idx < 3:
            return _replicate(img, idx)
        if random.uniform(0, 1) > self.probability:
            return img
        return _to_gray(img)


class AGWTrainTransform:
    """Pad(10) → RandomCrop(H, W) → flip → ToTensor → Normalize → extra tensor ops.

    views > 1 returns a stacked [views, 3, H, W] tensor; each view draws its own crop/flip.
    """
    def __init__(self, image_size, extra=(), views=1):
        from torchvision import transforms as T
        self.base = T.Compose([T.Pad(10), T.RandomCrop(tuple(image_size)), T.RandomHorizontalFlip(),
                               T.ToTensor(), T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])
        self.extra = list(extra) if views == 1 else [list(e) for e in extra]
        self.views = views

    def _one(self, img, ops):
        out = self.base(img)
        for op in ops:
            out = op(out)
        return out

    def __call__(self, img, gray=False):
        if gray:
            raise ValueError("AGW/CAJ do not use sampler-driven grayscale")
        if self.views == 1:
            return self._one(img, self.extra)
        return torch.stack([self._one(img, ops) for ops in self.extra])


def make_transforms(cfg):
    """AGW: plain augmentation. CAJ: RGB → (erase, erase+exchange) views; IR → erase + adaptive gray."""
    size, erase = cfg["model"]["image_size"], cfg["augmentation"]["random_erasing"]
    if not cfg["augmentation"]["channel_augmentation"]:
        plain = [ChannelRandomErasing(erase)] if erase else []
        return {"rgb": AGWTrainTransform(size, plain), "ir": AGWTrainTransform(size, plain)}
    return {"rgb": AGWTrainTransform(size, [[ChannelRandomErasing(erase)],
                                            [ChannelRandomErasing(erase), ChannelExchange(gray=2)]], views=2),
            "ir": AGWTrainTransform(size, [ChannelRandomErasing(erase), ChannelAdapGray(0.5)])}


def collate_caj(items):
    """[RGB view 0 (PK), RGB channel-exchanged view 1 (PK), IR (PK)], matching upstream cat order."""
    rgb = [it for it in items if it["modality"] == MODALITY_CODE["rgb"]]
    ir = [it for it in items if it["modality"] == MODALITY_CODE["ir"]]
    batch = collate_batch([dict(it, image=it["image"][v]) for v in (0, 1) for it in rgb] + ir)
    batch["views"] = torch.tensor([0] * len(rgb) + [1] * len(rgb) + [0] * len(ir))
    return batch


def check_caj_layout(batch, P, K):
    n = P * K
    assert batch["labels"][:n].equal(batch["labels"][n:2 * n]), "两个 RGB 视图的身份顺序不一致"
    assert batch["row_indices"][:n].equal(batch["row_indices"][n:2 * n]), "两个 RGB 视图须来自同一图片"
    check_pmt_layout({k: v[n:] for k, v in batch.items()}, P, K)
