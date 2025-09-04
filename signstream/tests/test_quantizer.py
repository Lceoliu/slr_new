import torch
from signstream.models.rvq.quantizer import ResidualVectorQuantizer


def test_rvq_shapes():
    rvq = ResidualVectorQuantizer(levels=2, codebook_size=4, dim=3)
    x = torch.randn(2, 5, 3)
    q, indices, loss = rvq(x)
    assert q.shape == x.shape
    assert len(indices) == 2
    assert indices[0].shape == x.shape[:-1]
    assert loss.requires_grad
