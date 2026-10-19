import pytest

torch = pytest.importorskip("torch")

from tokpremium.vocab.init_embeddings import mean_init_rows  # noqa: E402


def test_mean_of_sources_and_old_rows_untouched():
    w = torch.zeros(6, 4)
    w[0], w[1], w[2] = torch.ones(4), 3 * torch.ones(4), 5 * torch.ones(4)
    before = w[:3].clone()
    mean_init_rows(w, {3: [0, 1], 4: [2], 5: [0, 1, 2, 2]}, old_rows=3)
    assert torch.equal(w[:3], before)
    assert torch.allclose(w[3], 2 * torch.ones(4))
    assert torch.equal(w[4], w[2])  # N=1 is a copy
    assert torch.allclose(w[5], 3.5 * torch.ones(4))


def test_bf16_mean_computed_in_float32():
    w = torch.zeros(5, 2, dtype=torch.bfloat16)
    w[0], w[1] = 1.0, 2.0
    mean_init_rows(w, {4: [0, 1] * 50}, old_rows=3)
    assert w[4].dtype == torch.bfloat16
    assert torch.allclose(w[4].float(), torch.full((2,), 1.5))


@pytest.mark.parametrize("bad", [{1: [0]}, {3: [4]}, {3: []}])
def test_rejects_bad_decomposition(bad):
    with pytest.raises(ValueError):
        mean_init_rows(torch.zeros(6, 2), bad, old_rows=3)
