"""Synthetic checks only: these are not retrieval or training results."""
import copy
import pytest
import torch
from src.model import PMT
from src.local_head import LocalHead, PMTWithLocal, local_cross_entropy, add_local_loss

torch.set_num_threads(1)


def test_default_grid_pooling():
    head = LocalHead(1, 3).eval()
    x = torch.arange(21.).reshape(1, 21, 1, 1).expand(1, 21, 10, 1).reshape(1, 210, 1)
    features, logits = head(x, (21, 10))
    torch.testing.assert_close(features.flatten(), torch.tensor([4.5, 15.]))
    assert logits.shape == (1, 2, 3)


@pytest.mark.parametrize('grid,count', [((21,10),211), ((1,10),10), ((0,10),0)])
def test_bad_layout(grid, count):
    with pytest.raises(ValueError):
        LocalHead(4,3)(torch.randn(4,count,4), grid)


def test_rng_isolation():
    before = torch.get_rng_state().clone()
    LocalHead(8,3)
    assert torch.equal(before, torch.get_rng_state())


def test_ce_and_disabled_loss():
    logits = torch.randn(4,2,3,requires_grad=True)
    labels = torch.tensor([0,1,2,0])
    expected = sum(torch.nn.functional.cross_entropy(logits[:,j],labels) for j in range(2))/2
    torch.testing.assert_close(local_cross_entropy(logits,labels), expected)
    original = torch.tensor(2.,requires_grad=True)
    for weight, output in [(0, {'local_logits':logits}), (.5, {'local_logits':None})]:
        assert add_local_loss({'total':original},output,labels,weight)['total'] is original
    result = add_local_loss({'total':original},{'local_logits':logits},labels)
    torch.testing.assert_close(result['total'], original + .5*expected)


def test_backbone_gradient_retrieval_and_resume():
    cfg = dict(image_size=[32,20],patch_size=8,patch_stride=6,
               global_feature_dim=24,depth=2,heads=3,dropout=0.,drop_path=0.)
    baseline = PMT(3,cfg).eval()
    x = torch.randn(6,3,32,20)
    expected = baseline(x)['retrieval_feature'].detach()
    model = PMTWithLocal(baseline).eval()
    torch.testing.assert_close(model(x)['retrieval_feature'],expected,rtol=0,atol=0)
    model.train()
    output = model(x)
    assert output['patch_grid'] == (5,3)
    loss = local_cross_entropy(output['local_logits'],torch.tensor([0,1,2,0,1,2]))
    loss.backward()
    for param in [model.base.patch_embed.proj.weight,
                  *[c.weight for c in model.local_head.classifiers],
                  *[n.weight for n in model.local_head.necks]]:
        assert param.grad is not None and torch.isfinite(param.grad).all() and param.grad.abs().sum() > 0
    model.eval()
    restored = PMTWithLocal(PMT(3,cfg)).eval()
    restored.load_state_dict(copy.deepcopy(model.state_dict()),strict=True)
    torch.testing.assert_close(restored(x)['local_logits'],model(x)['local_logits'])
    disabled = PMTWithLocal(PMT(3,cfg),enabled=False)
    assert disabled.local_head is None
    assert set(disabled.state_dict()) == set(PMT(3,cfg).state_dict())
