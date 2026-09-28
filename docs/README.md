# Documentation

Each page covers one part of TandemLLM and names the files that implement it. Start with the architecture page, a one-page tour that links to all the others.

| page | what it covers |
|-|-|
| [architecture.md](architecture.md) | the model, the byte budget, the decode loop, a map of the code |
| [quantisation.md](quantisation.md) | the checkpoint's FP8, NVFP4, the clip search, the quality gate, the three profiles |
| [kernels.md](kernels.md) | row invariance, the 4-bit and 8-bit matrix products, the recurrent-layer kernels, graphs |
| [speculative-decoding.md](speculative-decoding.md) | the two block drafters, the lookup drafter, the length router, tree verify through recurrent layers |
| [exactness.md](exactness.md) | the lossless promise, its one exception, and the checks behind it |
| [caches.md](caches.md) | the resident prefix, the state store, the persistent suffix store |
| [server.md](server.md) | the OpenAI API, tool calls, structured outputs, streaming, admission |
| [dashboard.md](dashboard.md) | the Live activity view, the other views, the API contract |
| [operations.md](operations.md) | running a profile, memory rules, holds, deploys, troubleshooting |
| [measurement.md](measurement.md) | the benchmark row, noise, the release gate, the comparison with vLLM |
| [adding-a-model.md](adding-a-model.md) | the seams a new model plugs into, and the tests it must pass |
| [roadmap.md](roadmap.md) | mixture of experts, parallel requests, other GPUs |

Two more references live next to the code. [server/METRICS.md](../server/METRICS.md) lists every Prometheus metric, and [contract/dashboard-v1/](contract/dashboard-v1/README.md) holds the dashboard API's schemas.

Every number on these pages was measured on one DGX Spark, unless the page calls it arithmetic or an estimate. [measurement.md](measurement.md) says how.

This documentation is licensed under CC BY 4.0 ([LICENSE](LICENSE)). The code is licensed separately (see the [README](../README.md)).
