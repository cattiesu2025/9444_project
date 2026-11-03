"""D's horizontal auxiliary identity supervision; retrieval stays global-only."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class LocalHead(nn.Module):
    """Pool explicit row-major [N,H*W,D] patches into disjoint height bands.

    Boundary j is floor(j*H/regions). Every row belongs to exactly one band,
    including odd heights (the default 21x10 grid splits into 10 and 11 rows).
    Returns pre-BN features [N,q,D] and identity logits [N,q,C].
    """
    def __init__(self, feature_dim, num_classes, regions=2, seed=17):
        super().__init__()
        if any(type(v) is not int or v < 1 for v in (feature_dim, num_classes, regions)):
            raise ValueError("feature_dim, num_classes and regions must be positive integers")
        self.feature_dim, self.regions = feature_dim, regions
        # CPU construction and isolated generator leave the baseline RNG untouched.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.necks = nn.ModuleList([nn.BatchNorm1d(feature_dim) for _ in range(regions)])
            self.classifiers = nn.ModuleList([
                nn.Linear(feature_dim, num_classes, bias=False) for _ in range(regions)])
            for neck, classifier in zip(self.necks, self.classifiers):
                neck.bias.requires_grad_(False)
                nn.init.normal_(classifier.weight, std=.001)

    def forward(self, patches, grid):
        if len(grid) != 2 or any(type(v) is not int or v < 1 for v in grid):
            raise ValueError("grid must contain explicit positive integer H and W")
        h, w = grid
        if patches.ndim != 3 or patches.shape[1:] != (h * w, self.feature_dim):
            raise ValueError("Expected [N,H*W,feature_dim] patches without special tokens")
        if h < self.regions:
            raise ValueError("Each horizontal region needs at least one row")
        spatial = patches.reshape(patches.shape[0], h, w, self.feature_dim)
        features = torch.stack([
            spatial[:, j*h//self.regions:(j+1)*h//self.regions].mean(dim=(1, 2))
            for j in range(self.regions)], dim=1)
        logits = torch.stack([classifier(neck(features[:, j]))
                              for j, (neck, classifier) in enumerate(zip(self.necks, self.classifiers))], dim=1)
        return features, logits


def local_cross_entropy(logits, labels):
    """Mean identity CE over all samples and regions, computed in FP32."""
    if logits.ndim != 3 or labels.shape != (logits.shape[0],):
        raise ValueError("Expected logits [N,q,C] and labels [N]")
    return torch.stack([F.cross_entropy(logits[:, j].float(), labels)
                        for j in range(logits.shape[1])]).mean()


def add_local_loss(terms, output, labels, weight=.5):
    """Use after PMTLoss in BOTH stages; preserve the original total when off."""
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("Local loss weight must be finite and nonnegative")
    result = dict(terms)
    logits = output.get("local_logits")
    if logits is None or weight == 0:
        loss = terms["total"].new_zeros(())
    else:
        loss = local_cross_entropy(logits, labels)
        result["total"] = terms["total"] + weight * loss
    result.update(local=loss, local_weighted=weight * loss)
    return result


class PMTWithLocal(nn.Module):
    """Opt-in adapter; baseline state-dict keys remain unchanged.

    Construct from the already initialized PMT BEFORE creating the optimizer.
    Disabled mode creates no head. State includes local BN buffers for resume.
    """
    def __init__(self, baseline, regions=2, enabled=True, seed=17):
        super().__init__()
        self.base = baseline.base
        self.bottleneck = baseline.bottleneck
        self.classifier = baseline.classifier
        self.local_head = (LocalHead(self.classifier.in_features,
                                    self.classifier.out_features, regions, seed)
                           if enabled else None)
        self.train(baseline.training)

    def forward(self, images):
        global_feature, patches, grid = self.base(images)
        bn_feature = self.bottleneck(global_feature)
        features, logits = ((None, None) if self.local_head is None
                            else self.local_head(patches, grid))
        return dict(global_feature=global_feature,
                    global_logits=self.classifier(bn_feature),
                    retrieval_feature=F.normalize(bn_feature, dim=1),
                    patch_features=patches, patch_grid=grid,
                    local_features=features, local_logits=logits)
