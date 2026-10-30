"""Recursive's Flint (`flint`, FlintForCausalLM) on the engine, from the BF16 checkpoint or our EXL3 pack
(ENG-E flint epic).

  config.py   the config, validated
  ref.py      the reference forward in torch, written from the published model description
  pack.py     the checkpoint or pack on disk, audited
  load.py     the weights on the device
  kernels.py  the row kernels Flint adds to LFM2's: sliding-window attention with sinks, its router
  engine.py   the board engine: decode step, verify block, commit, prompt chunks, CUDA graphs
  serving.py  the server's engine interface and builder
"""
