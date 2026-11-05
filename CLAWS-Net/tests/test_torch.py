import pytest
torch = pytest.importorskip("torch")
from claws_net.models import CLAWSNet, SimCLRNet, UNet
from claws_net.losses import nt_xent, CompositeLoss


@pytest.mark.parametrize("kw", [{}, {"use_ctf": False}, {"use_awdf": False}, {"n_views": 1}])
def test_forward_shapes(kw):
    nv = kw.get("n_views", 3)
    m = CLAWSNet(**kw); out = m([torch.rand(2, 3, 64, 64) for _ in range(nv)])
    assert out["mal"].shape == (2, 2) and out["sub"].shape == (2, 3) and out["view_w"].shape == (2, nv)


def test_loss_and_backward():
    m = CLAWSNet(); out = m([torch.rand(4, 3, 64, 64) for _ in range(3)])
    loss, _ = CompositeLoss()(out, torch.tensor([0, 1, 1, 0]), torch.tensor([-1, 0, 2, -1])); loss.backward()
    loss2, _ = CompositeLoss()(out, torch.tensor([0, 1, 1, 0]), torch.tensor([-1, -1, -1, -1]))  # no subtype labels
    assert torch.isfinite(loss2)


def test_simclr_and_unet():
    z = SimCLRNet()(torch.rand(4, 3, 64, 64)); assert z.shape == (4, 128) and torch.isfinite(nt_xent(z, z.flip(0)))
    assert UNet()(torch.rand(1, 2, 128, 128)).shape == (1, 1, 128, 128)
    enc_state = SimCLRNet().encoder.state_dict(); CLAWSNet().load_pretrained_encoder(enc_state)
