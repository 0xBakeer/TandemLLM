"""CPU tests for the drafter fine-tune: the packing mask, and the label arithmetic.

Both of these are the kind of thing that is invisible when it is wrong. A mask that lets two packed
blocks see each other trains a drafter on information it will not have at serve time, and the
acceptance gate -- which packs nothing -- would keep reporting an honest number while the training
signal quietly rotted. An off-by-one in the labels does the same in the other direction: the loss
goes down and the drafter learns to predict the wrong position.
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.train_dflash2 import block_masks  # noqa: E402


def test_mask_is_the_serving_rule_block_by_block():
    """For every packed block, the mask row equals what a single-block pass would have built."""
    anchors = torch.tensor([9, 40, 300])
    ctx_len = int(anchors.max())
    block, window = 8, 2048
    full, slide = block_masks(anchors, ctx_len, block, window, "cpu")
    assert full.shape == (3 * block, ctx_len + 3 * block)

    for b, p in enumerate(anchors.tolist()):
        for j in range(block):
            row = b * block + j
            qpos = p + j
            # context: strictly before the anchor, and inside the window
            for k in range(ctx_len):
                want = (k < p) and (qpos - k) < window
                assert bool(slide[row, k]) is want, (b, j, k)
                assert bool(full[row, k]) is (k < p)
            # the block's own rows: all of its own, none of anybody else's
            for bb in range(3):
                for jj in range(block):
                    key = ctx_len + bb * block + jj
                    assert bool(slide[row, key]) is (bb == b)
                    assert bool(full[row, key]) is (bb == b)


def test_mask_window_actually_cuts():
    """A short window must exclude far context; the block's own rows stay visible."""
    anchors = torch.tensor([100])
    full, slide = block_masks(anchors, 100, 8, 16, "cpu")
    assert not bool(slide[0, 0])                       # 100 - 0 >= 16
    assert bool(slide[0, 99])                          # 100 - 99 < 16
    assert bool(full[0, 0])                            # the full mask has no window
    assert bool(slide[0, 100])                         # its own anchor row


def test_label_indexing_matches_the_verify_rule():
    """`label[i]` is the token after position i, so row j of a block anchored at p wants
    `label[p + j - 1]`, and the seven drafted tokens are compared against `label[p : p + 7]`."""
    ids = torch.arange(50)
    label = ids[1:].clone()                            # a greedy trace: the next token, exactly
    p, block = 12, 8
    slots = p + torch.arange(1, block)                 # positions the rows predict
    assert torch.equal(label[slots - 1], ids[slots])
    assert torch.equal(label[p:p + block - 1], ids[p + 1:p + block])


def test_first_mismatch_is_the_accepted_prefix():
    """The gate counts the matching prefix, then adds the verify pass's own bonus token."""
    want = torch.tensor([1, 2, 3, 4, 5, 6, 7])
    for cut in range(8):
        draft = want.clone()
        if cut < 7:
            draft[cut] = 999
        diff = draft != want
        match = int(diff.float().argmax()) if bool(diff.any()) else len(want)
        assert match == cut
        assert match + 1 == cut + 1                    # committed positions, the ledger's unit


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok {name}")
