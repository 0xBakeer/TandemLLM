"""The per-model cache state interface (ENG-221): what a continuation needs, in named sections.

The caches (`engine/cache.py`, `engine/disk_cache.py`, `engine/spill.py`) were written for the 27B
and read its `eng.state.S` and `eng.state.conv` directly. A model whose state has other parts
(Flash-Next: GDN and conv as the 27B, plus the PLE context and its dilated conv, and the QSA
indexer's pooled keys) cannot be cached that way. `ModelState` is the one shape they can share:

    sections()        name -> tensor, every part of the recurrent state (not the KV rows, which
                      the caches keep by length); the disk tier writes them under these names
    nbytes()          the sum of the sections
    clone()           an independent copy (a snapshot)
    restore(snap)     copy a snapshot's values into this state in place (captured CUDA graphs
                      keep their tensors)
    signature()       names, shapes and dtypes in order: two states with the same signature can
                      restore each other; a file written under another signature is refused

`Qwen38State` is the 27B's: the two tensors it always had, under the names "gdn.S" and "gdn.conv".
A model with no recurrent state has no sections (an empty dict is a valid state).

What is ported onto this interface is listed in the ENG-221 Memo note; the cache modules still
read the 27B's tensors directly until ENG-157 moves them (this module changes no caller).
"""

from __future__ import annotations

import hashlib
from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class ModelState(Protocol):
    def sections(self) -> dict[str, torch.Tensor]: ...

    def nbytes(self) -> int: ...

    def clone(self) -> "ModelState": ...

    def restore(self, snap: "ModelState") -> None: ...

    def signature(self) -> str: ...


def signature_of(sections: dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for name, t in sections.items():
        h.update(f"{name}:{tuple(t.shape)}:{t.dtype};".encode())
    return h.hexdigest()[:16]


class SectionState:
    """A ModelState over a dict of named tensors (the shared implementation)."""

    def __init__(self, sections: dict[str, torch.Tensor]):
        self._sections = dict(sections)

    def sections(self) -> dict[str, torch.Tensor]:
        return dict(self._sections)

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.sections().values())

    def clone(self) -> "SectionState":
        return SectionState({k: v.clone() for k, v in self.sections().items()})

    def restore(self, snap: ModelState) -> None:
        theirs = snap.sections()
        if signature_of(theirs) != self.signature():
            raise ValueError(f"state signature {signature_of(theirs)} cannot restore {self.signature()}")
        for k, v in self.sections().items():
            v.copy_(theirs[k])

    def signature(self) -> str:
        return signature_of(self.sections())


class Qwen38State(SectionState):
    """The 27B's recurrent state (`engine.model.GDNState`): S and conv of every GDN layer, as the
    same tensors (no copy), so a restore lands in the buffers its decode graphs captured."""

    def __init__(self, gdn_state):
        self.gdn = gdn_state
        super().__init__({"gdn.S": gdn_state.S, "gdn.conv": gdn_state.conv})

    def sections(self) -> dict[str, torch.Tensor]:
        # GDNState.swap() trades S with its spare between blocks: read the live one every time
        return {"gdn.S": self.gdn.S, "gdn.conv": self.gdn.conv}
