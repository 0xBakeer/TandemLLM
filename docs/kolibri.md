# Kolibri-1 (experimental)

Status: work in progress. This branch runs Aleph Alpha's Kolibri-1 behind the same OpenAI-compatible server as the main engine. It is a snapshot for people who want to try it, not a release, and the code will change.

The weights are [0xBakeer/TandemLLM-Kolibri-1-NVFP4](https://huggingface.co/0xBakeer/TandemLLM-Kolibri-1-NVFP4), a mixed NVFP4 and FP8 quantisation of Kolibri-1. They load only with the code on this branch.

## Hardware

We built and tested it on an NVIDIA DGX Spark (GB10, Blackwell, 128 GB of memory shared by CPU and GPU). The weights take about 45 GB. On top come the KV cache, about 20 KB per token of `--max-len`, and the in-memory prefix cache, up to 8 GiB by default (`--kolibri-anchor-gb`, `--kolibri-stash-gb`). The loader refuses to start when free memory cannot hold the weights plus a 20 GiB margin, so run one engine per board.

## Install and download

Install the packages from the Quick start in the [README](../README.md) (Python 3.11, PyTorch with CUDA, Triton, transformers, safetensors, numpy). Then:

```bash
git clone -b kolibri-experimental https://github.com/0xBakeer/TandemLLM.git
cd TandemLLM
hf download 0xBakeer/TandemLLM-Kolibri-1-NVFP4 --local-dir ./Kolibri-1-NVFP4
```

## Run

```bash
python server/app.py --kolibri --port 8001 --served-model Kolibri-1 --max-len 65536 \
    --kolibri-set ./Kolibri-1-NVFP4 --kolibri-tokenizer ./Kolibri-1-NVFP4
```

The downloaded folder is both the weight set and the tokenizer. It holds every attention tensor, so `--kolibri-fp8` stays empty. Loading and a short warm-up come first; the server listens after the log prints `[kolibri] warm-up done`:

```bash
curl -s localhost:8001/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"model": "Kolibri-1", "reasoning_effort": "none", "messages": [{"role": "user", "content": "Was ist die Hauptstadt von Deutschland?"}]}'
```

`reasoning_effort` takes `none`, `low`, `medium` or `high`. Sampling defaults to the model's `generation_config.json`; send `"temperature": 0` for greedy text. `python tools/kolibri_serve_check.py --base http://127.0.0.1:8001/v1` runs a few client checks (thinking, tool calls, prefix reuse) against a running server, and `python -m pytest tests/test_kolibri_*.py` runs the tests (`KOLIBRI_TOKENIZER=./Kolibri-1-NVFP4` adds the chat template ones).

## Not there yet

- No speculative decoding: every forward makes one token.
- No on-disk prompt cache. The prefix cache lives in memory; only a planned stop (SIGTERM) writes the live conversation to disk for the next start.
- Not in `install.sh`: set it up by hand as above.
