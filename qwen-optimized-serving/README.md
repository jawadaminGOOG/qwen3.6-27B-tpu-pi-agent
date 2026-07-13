# Optimized Qwen3.6-27B-FP8 serving on Cloud TPU v6e-4

This recipe provides the optimized serving configuration for **`Qwen/Qwen3.6-27B-FP8`** on a **Cloud TPU v6e-4** (4 chips, 2x2 topology) using GKE and vLLM. 

It is designed for developer-focused, long-context agentic coding assistant workloads where low latency (TTFT) and high concurrency are required.

## Key Optimizations

1.  **Prefix Caching (`--enable-prefix-caching`):** Caches the KV-cache of shared prefix tokens (such as prompt templates, instructions, and static codebase context). For iterative agent sessions, this reduces TTFT by over **80%** (~500ms) by avoiding redundant prefill computation.
2.  **FP8 KV Cache (`--kv-cache-dtype=fp8`):** Compresses the key-value cache to FP8, doubling the maximum concurrency and context length headroom on the TPU VM.
3.  **Chunked Prefill (`--enable-chunked-prefill`):** Chunks large prefill operations to prevent them from blocking decoding steps of active streams, ensuring smooth inter-token latency (ITL).
4.  **Model-Specific Routing:** Configures Qwen-specific XML tool parsing (`--tool-call-parser=qwen3_xml`) and auto-tool choice.

## Workload Manifests

*   [qwen3.6-27b-tpu-optimized.yaml](file:///usr/local/google/home/jawadamin/Repos/tpu-recipes-1/inference/trillium/vLLM/Qwen3.6-27B/qwen-optimized-serving/qwen3.6-27b-tpu-optimized.yaml): Deploy optimized Qwen server.
*   [qwen-chat-template.yaml](file:///usr/local/google/home/jawadamin/Repos/tpu-recipes-1/inference/trillium/vLLM/Qwen3.6-27B/qwen-optimized-serving/qwen-chat-template.yaml): ConfigMap containing the Jinja2 chat template.

## Deploying the Server

1.  Create the namespace and Hugging Face secret:

    ```bash
    export HF_TOKEN=YOUR_TOKEN
    kubectl create namespace qwen-serving
    kubectl create secret generic hf-secret \
        -n qwen-serving \
        --from-literal=hf_api_token=${HF_TOKEN}
    ```

2.  Apply the chat template ConfigMap:

    ```bash
    kubectl apply -f qwen-chat-template.yaml
    ```

3.  Apply the optimized serving manifest:

    ```bash
    kubectl apply -f qwen3.6-27b-tpu-optimized.yaml
    ```

    *Note: Compilation for the full 32K context window (`--max-model-len=32768`) can take 10-12 minutes. The pod status will transition to `1/1 READY` once finished.*

## Concurrency Benchmarks

We evaluate the optimized server using two workloads representing developer coding patterns. The benchmark client runs inside the cluster using the following manifests:
*   [benchmark_agentic.yaml](file:///usr/local/google/home/jawadamin/Repos/tpu-recipes-1/inference/trillium/vLLM/Qwen3.6-27B/qwen-optimized-serving/benchmark_agentic.yaml): Benchmark execution pod.
*   [benchmark-prep-script.yaml](file:///usr/local/google/home/jawadamin/Repos/tpu-recipes-1/inference/trillium/vLLM/Qwen3.6-27B/qwen-optimized-serving/benchmark-prep-script.yaml): ConfigMap to download/generate coding datasets.

### Running the Benchmarks

1.  Apply the dataset preparation ConfigMap:
    ```bash
    kubectl apply -f benchmark-prep-script.yaml
    ```
2.  Deploy the benchmark pod (configured for Long Context by default):
    ```bash
    kubectl apply -f benchmark_agentic.yaml
    ```
3.  Watch the logs for progress and final results:
    ```bash
    kubectl logs -n qwen-serving vllm-benchmark-agentic -f
    ```
4.  To test a different configuration (e.g., Medium Context), modify the `command` args in `benchmark_agentic.yaml` to run `prepare_medium.py` and adjust `--max-concurrency` accordingly, then recreate the pod.

---

## Measured Performance Results (TP=4)

The tables below show performance under the best-optimized configurations on a native **TPU v6e-4** node.

### 1. Medium Context (~6.4K tokens)
*Simulates reading a few files or medium-sized diffs.*

| Concurrency | Mean TTFT | Mean TPOT | Request Throughput | Total TPS (Output) |
| :---: | :---: | :---: | :---: | :---: |
| **20** | **523.48 ms** | **21.76 ms** | **2.92 req/s** | **747.5 tokens/s** |

### 2. Long Context (~27K tokens)
*Simulates large codebase context or long conversation history.*

| Concurrency | Mean TTFT | Mean TPOT | Request Throughput | Total TPS (Output) |
| :---: | :---: | :---: | :---: | :---: |
| **10** | **503.79 ms** | **16.81 ms** | **2.08 req/s** | **532.5 tokens/s** |

---

## Cost & Concurrency Analysis

Pricing is based on standard GCP rates for **Cloud TPU v6e** as of mid-2026:
*   **TPU v6e-4 Slice (4 chips):** $2.70 * 4 = **$10.80 per hour** ($7,884.00/month on-demand, $3,547.80/month under a 3-Yr Committed Use Discount).

By sharing the TPU node capacity among developers up to the optimal concurrency level, we achieve the following monthly flat cost per developer:

| Workload Context | Concurrency (Max Devs) | Cost/Dev/Month (On-Demand) | Cost/Dev/Month (3-Yr CUD) |
| :--- | :---: | :---: | :---: |
| **Medium Context (~6.4K)** | **20** | **$394.20** | **$177.39** |
| **Long Context (~27K)** | **10** | **$788.40** | **$354.78** |
