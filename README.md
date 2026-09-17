# qwen38-spark-engine

An inference engine for Qwen3.8-27B on a single DGX Spark (GB10), written for decode speed: FP8 and NVFP4 weights, speculative block verification across the model's Gated DeltaNet and attention layers, a lookup-memory drafter, and a per-step router over drafters.

Work in progress. Numbers appear in RESULTS.md only once measured.
