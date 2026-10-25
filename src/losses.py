"""PMT progressive hard triplet, modality discrepancy and center losses.

Formulas checked against hulu88/PMT at 0b31da9; no feature normalization
before metric losses. Degenerate DCL handling is documented in baseline.md.
"""
import torch
from torch import nn
from torch.nn import functional as F


def distances(x, y):
    # FP32 even when the backbone uses autocast: avoid half-precision cancellation.
    x, y = x.float(), y.float()
    return (x.square().sum(1)[:, None] + y.square().sum(1)[None, :] - 2 * x @ y.T).clamp_min(1e-12).sqrt()


def validate_layout(labels, k):
    if k < 2 or labels.ndim != 1 or len(labels) % (2 * k):
        raise ValueError("Expected balanced 2PK labels with K >= 2")
    a, b = labels.chunk(2)
    blocks = a.reshape(-1, k)
    if (not torch.equal(a, b) or not torch.all(blocks == blocks[:, :1])
            or len(torch.unique(blocks[:, 0])) != len(blocks) or len(blocks) < 2):
        raise ValueError("Expected distinct P >= 2 identities, contiguous K samples, RGB then IR")


def hard_triplet(features, labels, margin=.1):
    d = distances(features, features)
    same = labels[:, None] == labels[None, :]
    positives = same & ~torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    if not positives.any(1).all() or not (~same).any(1).all():
        raise ValueError("Triplet requires a non-self positive and a negative per anchor")
    hardest_positive = d.masked_fill(~positives, -torch.inf).max(1).values
    hardest_negative = d.masked_fill(same, torch.inf).min(1).values
    return F.relu(hardest_positive - hardest_negative + margin).mean()


def msel(features, labels, k):
    validate_layout(labels, k)
    a, b = features.chunk(2)
    target = labels[:len(a)]
    same = target[:, None] == target[None, :]
    nonself = same & ~torch.eye(len(a), dtype=torch.bool, device=labels.device)
    intra_a = (distances(a, a) * nonself).sum(1) / (k - 1)
    intra_b = (distances(b, b) * nonself).sum(1) / (k - 1)
    cross = distances(a, b)
    cross_a = (cross * same).sum(1) / k
    cross_b = (cross.T * same).sum(1) / k
    return ((cross_a - intra_a).square().mean() + (cross_b - intra_b).square().mean()) / 2


def dcl(features, labels, k):
    validate_layout(labels, k)
    identities = labels[:len(labels) // 2:k]
    same = identities[:, None] == labels[None, :]
    centers = torch.stack([features[labels == identity].mean(0) for identity in identities])
    d = distances(centers, features)
    negatives = d[~same].reshape(len(identities), -1)
    selected = (negatives < negatives.mean(1, keepdim=True)) & (negatives > 1e-6)
    # Equal distances yield no strictly-below-mean negatives in upstream code.
    # Fall back to all negatives for that center rather than returning NaN.
    selected = torch.where(selected.any(1, keepdim=True), selected, torch.ones_like(selected))
    negative_mean = ((negatives * selected).sum(1) / selected.sum(1)).mean()
    positive = d[same]
    positive = positive[positive > 1e-6]
    positive_mean = positive.mean() if positive.numel() else d.sum() * 0
    return positive_mean / negative_mean.clamp_min(1e-6)


class PMTLoss(nn.Module):
    def __init__(self, k=4, margin=.1, msel_weight=.5, dcl_weight=.5):
        super().__init__()
        self.k, self.margin = k, margin
        self.msel_weight, self.dcl_weight = msel_weight, dcl_weight

    def forward(self, output, labels, stage):
        if stage not in (1, 2, "V0"):
            raise ValueError("stage must be 1, 2 or V0")
        validate_layout(labels, self.k)
        features, logits = output["global_feature"].float(), output["global_logits"].float()
        a, b = logits.chunk(2)
        la, lb = labels.chunk(2)
        identity = F.cross_entropy(a, la) + F.cross_entropy(b, lb)
        zero = features.sum() * 0
        if stage == 1:
            fa, fb = features.chunk(2)
            triplet = hard_triplet(fa, la, self.margin) + hard_triplet(fb, lb, self.margin)
        else:
            triplet = hard_triplet(features, labels, self.margin)
        modality = msel(features, labels, self.k) if stage == 2 else zero
        center = dcl(features, labels, self.k) if stage == 2 else zero
        terms = {"id": identity, "triplet": triplet, "msel": modality, "dcl": center,
                 "weighted_msel": self.msel_weight * modality, "weighted_dcl": self.dcl_weight * center}
        terms["total"] = identity + triplet + terms["weighted_msel"] + terms["weighted_dcl"]
        return terms
