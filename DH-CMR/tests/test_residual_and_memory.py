import torch

from dh_cmr.chm.memory import ConditionalHyperedgeMemory
from dh_cmr.chm.projection import RelationResidual


def test_zero_memory_preserves_input_and_collinear_memory_removes_component():
    residual = RelationResidual(epsilon=1e-8)
    z = torch.tensor([[[2.0, 3.0, 0.0]]], requires_grad=True)
    assert torch.equal(residual(z, torch.zeros_like(z)), z)
    output = residual(z, torch.tensor([[[1.0, 0.0, 0.0]]]))
    torch.testing.assert_close(output, torch.tensor([[[0.0, 3.0, 0.0]]]))
    output.square().sum().backward()
    assert torch.isfinite(z.grad).all()


def test_conditional_partition_and_unified_pool_preserve_capacity():
    conditional = ConditionalHyperedgeMemory(8, 4, seed=7)
    unified = ConditionalHyperedgeMemory(8, 4, conditional=False, seed=7)
    assert conditional.prototypes.numel() == unified.prototypes.numel()
    z = torch.randn(2, 6, 8)
    partition = torch.arange(6)
    output = conditional(z, partition)
    changed = ConditionalHyperedgeMemory(8, 4, seed=7)
    with torch.no_grad():
        changed.prototypes[5].add_(10)
    torch.testing.assert_close(output.covered[:, :5], changed(z, partition).covered[:, :5])
    assert (output.indices[..., 0] != output.indices[..., 1]).all()


def test_random_memory_is_frozen_and_forward_is_read_only():
    memory = ConditionalHyperedgeMemory(8, 4, learnable=False)
    before = memory.prototypes.clone()
    assert not list(memory.parameters())
    memory.eval()
    memory(torch.randn(2, 3, 8), torch.tensor([0, 2, 4]))
    assert torch.equal(before, memory.prototypes)
