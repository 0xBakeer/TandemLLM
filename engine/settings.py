"""Every QWEN38_* knob the engine reads, in one registry, read through one object.

The engine is configured by environment variables (`ops/serve.env` sets them). They used to be read
by ~124 `os.environ.get("QWEN38_...", default)` calls spread over 22 modules, each with its own copy
of the default. This module is now the only place a knob's default is written down, and
`SETTINGS.get(name)` is the only way the engine reads one.

What it does not change: WHEN a knob is read. `SETTINGS` views the live process environment, so a
module that read its knob at import still does (into the same module global) and a function that
read one per call still does -- every value, every parse, every default is the one the old call
produced, and `serve.env` keeps working as it is. `tests/test_settings.py` pins that: no engine
module reads a QWEN38_* variable any other way, and every registry default equals the default the
removed call carried.

What it adds:
  * one table (`KNOBS`: name -> (default, the module that reads it)) to document, audit and diff;
  * `EngineSettings(env=..., prefix=...)`: the same reads over another mapping or another prefix
    (a `QSE_` prefix for the extracted package), e.g. to print what a profile file sets
    (`describe()`) or to build a second configuration's values in a test;
  * `describe()`: every knob with its effective value and whether the environment set it.

The step after this one (not done here): modules that capture a knob into a global at import take
it from a settings object handed to the engine instead, so two engines in one process can differ.
"""

from __future__ import annotations

import os
from typing import Mapping

_MISSING = object()

# name (without the prefix) -> (default, first module that reads it). Generated from the reads it
# replaced on 2026-09-27; the defaults are the literal defaults those reads carried.
KNOBS: dict[str, tuple] = {
    'FP8_TILES': (None, 'tools/fp8_linear.py'),                 #
    # read under a computed name (engine/router.py tree_nodes, tools/nvfp4_skinny.py _table)
    'TREE_NODES': ('0', 'engine/router.py'),
    'TREE_NODES_NARROW': ('0', 'engine/router.py'),
    'SKINNY_TILES_B': (None, 'tools/nvfp4_skinny.py'),
    'SKINNY_TILES_C': (None, 'tools/nvfp4_skinny.py'),
    'SKINNY_TILES_WIDE_B': (None, 'tools/nvfp4_skinny.py'),
    'BLOCKING_SYNC': ('0', 'server/app.py'),
    'COMMIT_IN_VERIFY': ('0', 'engine/model.py'),
    'CORPUS': ('', 'engine/drafters/ngram.py'),
    'DECODE_ATTN': ('0', 'engine/model.py'),
    'DEEP': ('0', 'server/app.py'),
    'DEEP_AFTER': ('2', 'server/app.py'),
    'DF2_TEMP': ('1.0', 'engine/drafters/dflash2.py'),
    'DF2_TREE_MODE': ('paths', 'engine/drafters/dflash2.py'),
    'DFLASH2': (None, 'engine/drafters/dflash2.py'),
    'DRAFT_FC_NVFP4': ('0', 'engine/drafters/dflash2.py'),
    'DRAFT_GRAPH': ('0', 'engine/drafters/dflash2.py'),
    'DRAFT_HEAD': (None, 'engine/drafters/dflash2.py'),
    'DRAFT_HEAD_NVFP4': ('0', 'engine/drafters/dflash2.py'),
    'DRAFT_NVFP4': ('0', 'engine/drafters/dflash2.py'),
    'DRAFT_TEMP': ('1.0', 'engine/drafters/dflash2.py'),
    'DSPARK': (None, 'engine/drafters/dspark.py'),
    'DSPARK_NO_YARN': (None, 'engine/drafters/dspark.py'),
    'FP8_HEAD': (None, 'engine/loader.py'),
    'FUSED_ADDNORM': ('0', 'engine/model.py'),
    'FUSED_ATTN': ('0', 'engine/model.py'),
    'FUSED_ATTN_PREP': ('0', 'engine/model.py'),
    'FUSED_COMMIT': ('0', 'engine/model.py'),
    'FUSED_GDN': ('1', 'engine/model.py'),
    'FUSED_GDNBLOCK': ('1', 'engine/model.py'),
    'FUSED_GDNPRE': ('1', 'engine/model.py'),
    'FUSED_GDNPREFILL': ('0', 'engine/model.py'),
    'FUSED_GDNTREE': ('1', 'engine/model.py'),
    'FUSED_GDNVERIFY': ('0', 'engine/model.py'),
    'FUSED_HEAD': ('1', 'engine/model.py'),
    'FUSED_NORM': ('1', 'engine/model.py'),
    'FUSE_PROJ': ('0', 'tools/nvfp4_linear_v2.py'),
    'GDNV_BV': ('16', 'tools/gdn_verify_kernels.py'),
    'GDNV_CONV_BLOCK': ('256', 'tools/gdn_verify_kernels.py'),
    'GDNV_CONV_WARPS': ('4', 'tools/gdn_verify_kernels.py'),
    'GDNV_ONE_WARP': ('0', 'tools/gdn_verify_kernels.py'),
    'GDNV_TREE_PF': ('0', 'tools/gdn_verify_kernels.py'),
    'GDNV_WARPS': ('4', 'tools/gdn_verify_kernels.py'),
    'GDNV_WY': ('0', 'tools/gdn_verify_kernels.py'),
    'GDNV_WY_CHAIN_MAXT': ('32', 'tools/gdn_verify_kernels.py'),
    'GDNV_WY_FUSED': ('0', 'tools/gdn_verify_kernels.py'),
    'GDNV_WY_FUSED_MAXT': ('32', 'tools/gdn_verify_kernels.py'),
    'GDNV_WY_KC': ('0', 'tools/gdn_verify_kernels.py'),
    'GDNV_WY_MAXT': ('32', 'tools/gdn_verify_kernels.py'),
    'GDN_AB': ('0', 'engine/model.py'),
    'GDN_CHUNK': ('64', 'engine/model.py'),
    'GDN_MM': ('fp32', 'engine/gdn.py'),
    'GDN_MM_FROM': ('64', 'engine/gdn.py'),
    'GDN_PREFILL_BV': ('32', 'tools/gdn_prefill_kernels.py'),
    'GDN_PREFILL_PREC': ('bf16x3', 'tools/gdn_prefill_kernels.py'),
    'GDN_PREFILL_PREC_A': ('tf32', 'tools/gdn_prefill_kernels.py'),
    'GDN_PREFILL_PREC_SCAN': ('', 'tools/gdn_prefill_kernels.py'),
    'GDN_PREFILL_SERIES': ('0', 'tools/gdn_prefill_kernels.py'),
    'GDN_PREFILL_STAGES': ('1', 'tools/gdn_prefill_kernels.py'),
    'GDN_PREFILL_WARPS_INTRA': ('8', 'tools/gdn_prefill_kernels.py'),
    'GDN_PREFILL_WARPS_SCAN': ('8', 'tools/gdn_prefill_kernels.py'),
    'GQA_FROM': ('64', 'engine/model.py'),
    'GRAPH_MAX_CTX': ('32768', 'engine/verify_graph.py'),
    'HEAD_GEMM': ('', 'tools/head_gemv.py'),
    'HOST_ASYNC': ('0', 'engine/model.py'),
    'HOST_WALK': ('0', 'engine/drafters/dflash2.py'),
    'KV_FP8': ('0', 'engine/model.py'),
    'LAUNCH_FIRST': ('0', 'server/app.py'),
    'LOWER_RIGHT_FROM': ('64', 'engine/model.py'),
    'MODEL': (None, 'engine/config.py'),
    'NVFP4': (None, 'engine/loader.py'),
    'NVFP4_BLOCK_TILES': ('0', 'tools/nvfp4_linear.py'),
    'NVFP4_DEQUANT_FROM': ('512', 'tools/nvfp4_linear.py'),
    'NVFP4_PREFILL_V2': ('0', 'tools/nvfp4_linear.py'),
    'NVFP4_PREFILL_V2_UNTIL': ('1024', 'tools/nvfp4_linear.py'),
    'NVFP4_SKINNY': ('0', 'tools/nvfp4_skinny.py'),
    'NVFP4_SPLITK': ('0', 'tools/nvfp4_linear.py'),
    'NVFP4_V2': ('1', 'tools/nvfp4_linear_v2.py'),
    'NVFP4_V2_BN': ('0', 'tools/nvfp4_linear_v2.py'),
    'NVFP4_V2_DOTS': ('1', 'tools/nvfp4_linear_v2.py'),
    'NVFP4_V2_MAX': ('32', 'tools/nvfp4_linear_v2.py'),
    'NVFP4_V2_MIN': ('1', 'tools/nvfp4_linear_v2.py'),
    'NVFP4_V2_PREFETCH': ('0', 'tools/nvfp4_linear_v2.py'),
    'NVFP4_V2_W': ('0', 'tools/nvfp4_linear_v2.py'),
    'PREFILL_CAUSAL': ('1', 'engine/model.py'),
    'RANKK': ('1', 'engine/model.py'),
    'SCALE_ON': ('weight', 'tools/fp8_linear.py'),
    'SKINNY_LDW': ('0', 'tools/nvfp4_skinny.py'),
    'SKINNY_PDL': ('0', 'tools/nvfp4_skinny.py'),
    'SKINNY_SRUN': ('0', 'tools/nvfp4_skinny.py'),
    'SKINNY_SSTUB': ('0', 'tools/nvfp4_skinny.py'),
    'SKINNY_TILES': (None, 'tools/nvfp4_skinny.py'),
    'SKINNY_TILES_WIDE': (None, 'tools/nvfp4_skinny.py'),
    'SKINNY_XSTUB': ('0', 'tools/nvfp4_skinny.py'),
    'SUFFIX_STORE': ('~/.qwen38-spark-engine/suffix', 'server/app.py'),
    'TREE_ALIAS_STATE': ('0', 'engine/model.py'),
    'TREE_CHAIN_DELEGATE': ('1', 'engine/model.py'),
    'TREE_HOST_DEPTH': ('0', 'engine/model.py'),
    'TREE_MS': ('', 'engine/router.py'),
    'TREE_WIDE_AFTER': ('0', 'engine/lenrouter.py'),
    'TWO_STREAM': ('0', 'engine/model.py'),
    'UT_INVERSE': ('1', 'engine/gdn.py'),
    'VERIFY_GRAPH': ('0', 'engine/model.py'),
    'VERIFY_ROWS': ('16', 'engine/model.py'),
    'VERIFY_TILES': ('1', 'tools/nvfp4_verify_tiles.py'),
    'WY_BLOCK': ('8', 'tools/gdn_wy_kernels.py'),
    'WY_BV': ('32', 'tools/gdn_wy_kernels.py'),
    'WY_DBG': ('0', 'tools/gdn_wy_kernels.py'),
    'WY_PREC': ('ieee', 'tools/gdn_wy_kernels.py'),
    'WY_WARPS': ('4', 'tools/gdn_wy_kernels.py'),
    'WY_WARPS_PREP': ('4', 'tools/gdn_wy_kernels.py'),
}


class EngineSettings:
    """The engine's knobs over an environment mapping (the live `os.environ` by default)."""

    def __init__(self, env: Mapping[str, str] | None = None, prefix: str = "QWEN38_"):
        self._env = env
        self.prefix = prefix

    @property
    def env(self) -> Mapping[str, str]:
        return os.environ if self._env is None else self._env

    def get(self, name: str, default=_MISSING):
        """The raw string the environment holds for `name`, else its registry default (or the
        explicit `default`, for the one read whose default is a module constant)."""
        if default is _MISSING:
            if name not in KNOBS:
                raise KeyError(f"{self.prefix}{name} is not a registered knob (engine/settings.py KNOBS)")
            default = KNOBS[name][0]
        return self.env.get(self.prefix + name, default)

    def is_set(self, name: str) -> bool:
        return (self.prefix + name) in self.env

    def describe(self) -> dict[str, dict]:
        return {k: {"value": self.get(k), "default": d, "set": self.is_set(k), "read_in": where}
                for k, (d, where) in sorted(KNOBS.items())}


#: The process's settings: the live environment, prefix QWEN38_.
SETTINGS = EngineSettings()
