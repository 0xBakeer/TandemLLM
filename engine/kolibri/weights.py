"""Load a Kolibri-1 set onto the device: one `KLayer` per layer, the embedding, the final norm, the head.

Where each tensor comes from:

  * the set directory (as published on Hugging Face): `layers/{L}.safetensors` with keys
    `layers.{L}.<module>.weight[...]` -- routed experts, shared expert and (in the all-NVFP4 build)
    attention in NVFP4, norms and the router in BF16, the router bias in fp32 -- and
    `outside.safetensors` (embedding BF16, final norm, e4m3 head with fp32 row scales);
  * optionally Aleph Alpha's FP8 release (`fp8_dir`) for the four attention projections, as e4m3
    codes with fp32 `weight_scale_inv` per 128x128 block. Then the experts and the shared expert
    come from the set and attention from the release.

Which source attention takes is `attn` ("fp8": the release, "set": the set's own tensors). A
`manifest.json` in the set directory may say it (`{"attention": "fp8"|"set", ...}`; any other keys
are ignored here), so a new set is a directory change, not a code change. Formats are read from
the tensors themselves: uint8 codes + e4m3 `weight_scale` + `weight_scale_2` is NVFP4; e4m3 codes +
`weight_scale_inv` is FP8 block; anything else is dense BF16.

Page cache is dropped after each file is read (posix_fadvise DONTNEED): on a DGX Spark the page
cache and the GPU share one memory pool, and cached file pages next to the loaded weights would
push the board into swap.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass

import torch
from safetensors import safe_open

from engine.kolibri.config import KolibriConfig
from engine.kolibri.kernels import E4M3Head, ExpertBank, FP8Linear, NVFP4Linear

PROJ = ("gate_proj", "up_proj", "down_proj")


@dataclass
class KLayer:
    index: int
    sliding: bool
    qkv: object          # .matmul(x [T, H] bf16) -> [T, q+k+v] bf16
    o: object            # .matmul(x [T, q] bf16) -> [T, H] bf16
    q_norm: torch.Tensor
    k_norm: torch.Tensor
    n_in: torch.Tensor
    n_pa: torch.Tensor
    n_pal: torch.Tensor
    n_pf: torch.Tensor
    gate: torch.Tensor   # bf16 [E, H], as released (exact; logits are fp32)
    bias: torch.Tensor   # fp32 [E]
    G: ExpertBank        # E routed experts (+ the shared expert at index E when it is NVFP4)
    U: ExpertBank
    D: ExpertBank
    shared: object = None  # the shared expert in another format (`SharedExpert`), else None

    @property
    def nbytes(self) -> int:
        n = self.qkv.nbytes + self.o.nbytes + self.G.nbytes + self.U.nbytes + self.D.nbytes
        n += self.shared.nbytes if self.shared is not None else 0
        n += self.gate.numel() * 2 + sum(t.numel() * 2 for t in (self.q_norm, self.k_norm, self.n_in,
                                                                   self.n_pa, self.n_pal, self.n_pf))
        return n


def drop_cache(path: str) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)
    except (OSError, AttributeError):
        pass


def linear_of(get, has, base: str, device) -> object:
    """The projection `base` (`....q_proj`) in whatever format its tensors are: NVFP4 (uint8 codes,
    e4m3 `weight_scale`, `weight_scale_2`) or FP8 block (e4m3 codes with an fp32 128x128 scale table
    named `weight_scale_inv`, as in the release, or `weight_scale`)."""
    w = get(base + ".weight")
    if w.dtype == torch.uint8 and has(base + ".weight_scale") and has(base + ".weight_scale_2"):
        return NVFP4Linear(w.to(device), get(base + ".weight_scale").to(device),
                           get(base + ".weight_scale_2").float().reshape(-1).to(device))
    if w.dtype == torch.float8_e4m3fn:
        for sk in (".weight_scale_inv", ".weight_scale"):
            if has(base + sk):
                sc = get(base + sk)
                N, K = w.shape
                if sc.dim() == 2 and tuple(sc.shape) == (-(-N // 128), -(-K // 128)):
                    return FP8Linear(w.to(device), sc.float().to(device))
    raise ValueError(f"{base}: not a served format ({w.dtype}; dense BF16 projections are not served)")


class Concat:
    """Projections of the same input in different formats (q in NVFP4, k and v in FP8): one launch
    per run of equal formats, outputs concatenated along N. Same `.matmul`/`.dense`/`.nbytes`."""

    def __init__(self, parts: list):
        self.parts = parts
        self.N = sum(p.N for p in parts)
        self.K = parts[0].K

    @property
    def nbytes(self) -> int:
        return sum(p.nbytes for p in self.parts)

    def dense(self) -> torch.Tensor:
        return torch.cat([p.dense() for p in self.parts])

    def matmul(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda or os.environ.get("KOLIBRI_CONCAT_OUT", "1") == "0":
            return torch.cat([p.matmul(x) for p in self.parts], dim=-1)
        # every part writes its own columns of one output: no concatenation launch
        x2 = x.reshape(-1, self.K)
        y = torch.empty(x2.shape[0], self.N, dtype=torch.bfloat16, device=x.device)
        n0 = 0
        for p in self.parts:
            p.matmul(x2, out=y[:, n0:n0 + p.N])
            n0 += p.N
        return y.view(*x.shape[:-1], self.N)


def fuse(parts: list) -> object:
    """Adjacent projections of one format become one launch; a mix of formats a `Concat`."""
    runs: list[list] = []
    def joinable(a, b):
        if type(a) is not type(b):
            return False
        # FP8 members must end on a 128-row scale block to share one scale table
        return not isinstance(a, FP8Linear) or (a.N % 128 == 0 and b.N % 128 == 0)
    for p in parts:
        if runs and joinable(runs[-1][-1], p):
            runs[-1].append(p)
        else:
            runs.append([p])
    fused = [r[0] if len(r) == 1 else type(r[0]).cat(r) for r in runs]
    return fused[0] if len(fused) == 1 else Concat(fused)


@dataclass
class SharedExpert:
    """The shared expert when it is not NVFP4 (then it cannot ride in the expert bank):
    gate|up fused [2F, H], down [H, F]."""
    gu: object
    d: object
    F: int

    @property
    def nbytes(self) -> int:
        return self.gu.nbytes + self.d.nbytes

    def act(self, x: torch.Tensor) -> torch.Tensor:
        """The down projection's bf16 output (before `.float()`); SiLU * up in one launch."""
        from engine.kolibri.kernels import swiglu
        return self.d.matmul(swiglu(self.gu.matmul(x), self.F))

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        gu = self.gu.matmul(x).float()
        a = (torch.nn.functional.silu(gu[:, : self.F]) * gu[:, self.F:]).to(torch.bfloat16)
        return self.d.matmul(a).float()


class Release:
    """Aleph Alpha's sharded release (FP8 or BF16), tensors by name."""

    def __init__(self, d: str):
        self.dir = d
        self.map = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
        self.h: dict[str, object] = {}

    def has(self, n: str) -> bool:
        return n in self.map

    def get(self, n: str) -> torch.Tensor:
        f = self.map[n]
        if f not in self.h:
            self.h[f] = safe_open(os.path.join(self.dir, f), framework="pt")
        return self.h[f].get_tensor(n)

    def close(self) -> None:
        for f in self.h:
            drop_cache(os.path.join(self.dir, f))
        self.h.clear()


def read_manifest(set_dir: str) -> dict:
    p = os.path.join(set_dir, "manifest.json")
    return json.load(open(p)) if os.path.isfile(p) else {}


def load_layer(cfg: KolibriConfig, set_dir: str, L: int, device, rel: Release | None) -> KLayer:
    path = os.path.join(set_dir, "layers", f"{L}.safetensors")
    E = cfg.experts
    with safe_open(path, framework="pt") as f:
        keys = set(f.keys())

        def g(n):
            return f.get_tensor(n)

        def has(n):
            return n in keys
        p = f"layers.{L}."

        def bf(n):
            return g(p + n + ".weight").to(torch.bfloat16).to(device)
        if rel is not None:
            ap = f"model.layers.{L}.self_attn."
            qkv = fuse([linear_of(rel.get, rel.has, ap + n, device) for n in ("q_proj", "k_proj", "v_proj")])
            o = linear_of(rel.get, rel.has, ap + "o_proj", device)
        else:
            qkv = fuse([linear_of(g, has, p + "self_attn." + n, device) for n in ("q_proj", "k_proj", "v_proj")])
            o = linear_of(g, has, p + "self_attn.o_proj", device)
        sp = p + "mlp.shared_experts."
        shared_nv = all(g(sp + n + ".weight").dtype == torch.uint8 for n in PROJ)
        shared = None
        if not shared_nv:
            gl, ul = (linear_of(g, has, sp + n, device) for n in PROJ[:2])
            shared = SharedExpert(fuse([gl, ul]), linear_of(g, has, sp + "down_proj", device), gl.N)
        banks = []
        for n in PROJ:
            names = [p + f"mlp.experts.{e}.{n}" for e in range(E)] + ([sp + n] if shared_nv else [])
            codes = torch.stack([g(b + ".weight") for b in names])
            scale = torch.stack([g(b + ".weight_scale") for b in names])
            s2 = torch.stack([g(b + ".weight_scale_2").float().reshape(()) for b in names])
            banks.append(ExpertBank(codes.to(device), scale.to(device), s2.to(device)))
            del codes, scale
        lw = KLayer(index=L, sliding=cfg.sliding(L), qkv=qkv, o=o,
                    q_norm=bf("self_attn.q_norm"), k_norm=bf("self_attn.k_norm"),
                    n_in=bf("input_layernorm"), n_pa=bf("post_attn_norm"),
                    n_pal=bf("post_attention_layernorm"), n_pf=bf("post_ffn_norm"),
                    gate=g(p + "mlp.gate.weight").to(torch.bfloat16).to(device),
                    bias=g(p + "moe.router.expert_bias").float().to(device),
                    G=banks[0], U=banks[1], D=banks[2], shared=shared)
    drop_cache(path)
    if rel is not None:
        rel.close()
    return lw


def load_outside(set_dir: str, device):
    path = os.path.join(set_dir, "outside.safetensors")
    with safe_open(path, framework="pt") as f:
        emb = f.get_tensor("embed_tokens.weight").to(torch.bfloat16).to(device)
        norm = f.get_tensor("norm.weight").to(torch.bfloat16).to(device)
        head = E4M3Head(f.get_tensor("lm_head.weight").to(device), f.get_tensor("lm_head.weight_scale").to(device))
    drop_cache(path)
    return emb, norm, head


def resolve_attn(set_dir: str, attn: str | None, fp8_dir: str | None) -> str:
    """"fp8" or "set": the argument, else the manifest's `attention`, else "set" when the set has a
    manifest at all, else fp8 when a release is given."""
    if attn:
        return attn
    man = read_manifest(set_dir)
    m = man.get("attention")
    if m in ("fp8", "set"):
        return m
    if man:
        return "set"          # a set with a manifest carries every tensor in its chosen format
    return "fp8" if fp8_dir else "set"


#: Memory that must stay free beside a Kolibri load (GiB): the KV, the anchors, graph pools, the
#: page cache the reads pass through, and the OS. On a DGX Spark the GPU allocates from the same
#: memory as everything else, and running out of it can hang the driver instead of raising an
#: OOM error. So the loader refuses when MemAvailable cannot hold the set plus this.
MEM_MARGIN_GIB = float(os.environ.get("KOLIBRI_MEM_MARGIN_GIB", "20"))


def mem_available_gib() -> float | None:
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 2 ** 20
    except OSError:
        return None
    return None


def set_bytes(set_dir: str, fp8_dir: str | None, mode: str) -> int:
    """What the load will hold: the layer files and outside tensors (NVFP4 attention included even
    when the release's FP8 attention replaces it, so slightly high), plus the FP8 attention."""
    n = sum(os.path.getsize(os.path.join(set_dir, "layers", f)) for f in os.listdir(os.path.join(set_dir, "layers")))
    n += os.path.getsize(os.path.join(set_dir, "outside.safetensors"))
    if mode == "fp8":
        n += int(1.75e9)
    return n


def check_memory(need_bytes: int, device, log=print) -> None:
    """Refuse a load that would not fit beside what already runs on the board."""
    if str(device).startswith("cpu") or os.environ.get("KOLIBRI_SKIP_MEM_CHECK") == "1":
        return
    avail = mem_available_gib()
    if avail is None:
        return
    need = need_bytes / 2 ** 30 + MEM_MARGIN_GIB
    if avail < need:
        raise MemoryError(f"[kolibri] REFUSING to load: MemAvailable {avail:.1f} GiB < {need:.1f} GiB "
                          f"(weights {need_bytes / 2 ** 30:.1f} + margin {MEM_MARGIN_GIB:g}); is another "
                          f"engine or GPU job running?")
    log(f"[kolibri] memory check: MemAvailable {avail:.1f} GiB >= {need:.1f} GiB needed")


def load_all(set_dir: str, fp8_dir: str | None, device, attn: str | None = None, log=print):
    """(cfg, layers, emb, final_norm, head). The config is the release's when given, else the set's."""
    cfg_src = fp8_dir if fp8_dir and os.path.isfile(os.path.join(fp8_dir, "config.json")) else set_dir
    cfg = KolibriConfig.load(cfg_src)
    mode = resolve_attn(set_dir, attn, fp8_dir)
    rel = Release(fp8_dir) if mode == "fp8" else None
    if mode == "fp8" and not fp8_dir:
        raise ValueError("attention from the FP8 release needs fp8_dir")
    check_memory(set_bytes(set_dir, fp8_dir, mode), device, log)
    t0 = time.time()
    emb, norm, head = load_outside(set_dir, device)
    layers, nb = [], 0
    for L in range(cfg.layers):
        lw = load_layer(cfg, set_dir, L, device, rel)
        nb += lw.nbytes
        layers.append(lw)
        if L % 10 == 9 or L == cfg.layers - 1:
            log(f"[kolibri] loaded {L + 1}/{cfg.layers} layers, {nb / 1e9:.2f} GB, {time.time() - t0:.1f} s")
    nb += emb.numel() * 2 + head.nbytes
    log(f"[kolibri] weights {nb / 1e9:.2f} GB (attention from {mode}) in {time.time() - t0:.1f} s")
    return cfg, layers, emb, norm, head, {"attention": mode, "bytes": nb}
