"""QWEN38_QUANT: load-time quantisation of chosen weights, as rules `scope:pattern=from>to`, `;` between rules.

  scope   draft   the MTP drafter's copy only. The verify reads the served weight, so the text is the plain greedy
                  text (the exactness rule holds); what changes is the drafts' acceptance and their cost.
          model   the served weight itself: the run is no longer the checkpoint's arithmetic. Refused until a run can
                  carry a not-exact mark (research/tandem/flashnext-fp4local-gap.md, section F).
  pattern a weight's name as the engine knows it (`lm_head`)
  from>to the stored format and the target: bf16>nvfp4 (two levels of scale, round to nearest; tools/nvfp4_linear
          `quantize_to_nvfp4`), bf16>int8 (one scale a row, `pack.Int8Rows`)

`draft:lm_head=...` builds the drafter's head over QWEN38_FN_DRAFT_HOT ids of ops/fn-hot-vocab.npy (0: every id).
NVIDIA FP4-Local's head is bf16, 1.27 GB: read by each of a round's draft steps it is most of the drafting time.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

FORMATS = ("nvfp4", "int8")


@dataclass(frozen=True)
class Rule:
    scope: str
    pattern: str
    src: str
    dst: str


def parse(text: str | None) -> list[Rule]:
    """`draft:lm_head=bf16>nvfp4;...` -> [Rule]; an empty text is no rules."""
    rules = []
    for part in (text or "").replace(",", ";").split(";"):
        part = part.strip()
        if not part:
            continue
        try:
            scope, rest = part.split(":", 1)
            pattern, conv = rest.split("=", 1)
            src, dst = conv.split(">", 1)
        except ValueError:
            raise ValueError(f"QWEN38_QUANT {part!r}: scope:pattern=from>to") from None
        r = Rule(scope.strip(), pattern.strip(), src.strip(), dst.strip())
        if r.scope not in ("draft", "model"):
            raise ValueError(f"QWEN38_QUANT {part!r}: scope is draft or model")
        if r.scope == "model":
            raise ValueError(f"QWEN38_QUANT {part!r}: model scope not available yet (a run that changes the served "
                             "weights must be marked not exact first)")
        if r.dst not in FORMATS:
            raise ValueError(f"QWEN38_QUANT {part!r}: target one of {FORMATS}")
        rules.append(r)
    return rules


class NVFP4Rows:
    """An NVFP4 weight (tools/nvfp4_linear.NVFP4Block) in the engine's `mul_rows` protocol: fp32 [M, N] out, the
    tensor-core skinny kernel up to 32 rows (tools/nvfp4_skinny), the dequantised weight above."""

    def __init__(self, block):
        self.block = block
        self.shape = (block.N, block.K)

    def mul_rows(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] <= 32:
            from tools.nvfp4_skinny import nvfp4_matmul_skinny
            return nvfp4_matmul_skinny(x.contiguous(), self.block).float()
        return x.float() @ self.block.dequant().float().T


def draft_head(w: torch.Tensor, ids: torch.Tensor, fmt: str):
    """The drafter's head over `ids` (rows of the served head `w`, bf16 [V, H]) in `fmt`."""
    rows = w.index_select(0, ids).contiguous()
    if fmt == "int8":
        from engine.models.flashnext.pack import Int8Rows
        return Int8Rows.of(rows.half())
    from tools.nvfp4_linear import quantize_to_nvfp4
    return NVFP4Rows(quantize_to_nvfp4(rows))


def apply(eng, rules: list[Rule], hot_ids=None, log=print) -> list[str]:
    """Apply the draft rules to an engine; returns what was done (for the load ledger)."""
    done = []
    for r in rules:
        if r.scope == "draft" and r.pattern == "lm_head":
            eng.set_draft_head(hot_ids, fmt=r.dst)
            n = eng.draft_ids.numel() if eng.draft_ids is not None else 0
            done.append(f"draft:lm_head {r.src}>{r.dst} over {n} ids")
        else:
            raise ValueError(f"QWEN38_QUANT {r}: no such weight in this engine")
    if log and done:
        log("[quant] " + "; ".join(done))
    return done
