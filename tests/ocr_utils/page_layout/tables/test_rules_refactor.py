"""Извлечение ``fragment_layers`` и ``chain_groups`` из ``rules.py`` не меняет выхода прежних функций."""

from __future__ import annotations

import pytest

from ocr_utils.page_layout.tables import rules, ruling
from tests.ocr_utils.page_layout.tables import synthetic

DPI = synthetic.DPI


def _pages():
    """Ровная, наклонённая, изогнутая таблицы и изогнутая по S."""
    return [
        synthetic.make_table().image,
        synthetic.make_table(skew_deg=1.5).image,
        synthetic.bend(synthetic.make_table().image, 6.0),
        synthetic.make_curved_table().image,
    ]


@pytest.mark.parametrize("index", range(4))
def test_fragments_and_chain_are_unchanged(index: int) -> None:
    binary = ruling.binarize(_pages()[index])
    layers = rules.fragment_layers(binary, DPI)
    assert rules.fragments(binary, DPI) == (layers[0].fragments, layers[1].fragments)
    for layer in layers:
        # Номер метки у каждого фрагмента — именно его компонента: габарит меток совпадает с габаритом фрагмента.
        for fragment, label in zip(layer.fragments, layer.label_ids):
            ys, xs = (layer.labels == label).nonzero()
            assert (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1) == fragment.box.as_tuple()
        groups = rules.chain_groups(layer.fragments, DPI)
        assert rules.chain(layer.fragments, DPI) == [group.segment for group in groups]
        members = sorted(index for group in groups for index in group.members)
        assert len(members) == len(set(members))
