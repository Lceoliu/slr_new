import torch
from signstream.models.rvq.quantizer import ResidualVectorQuantizer


def test_quantizer_roundtrip():
    q = ResidualVectorQuantizer(dim=4, levels=2, codebook_size=16)
    x = torch.randn(1, 3, 4)
    quantized, codes, loss = q(x)
    recon = q.dequantize(codes)
    assert recon.shape == x.shape
    assert isinstance(loss, torch.Tensor)
    assert quantized.shape == x.shape
