# Qwen3.6-27B-FP8 Cloud TPU v6e Serving & Harness Integration Recipe

This repository contains modular recipes to deploy, optimize, and integrate the **`Qwen/Qwen3.6-27B-FP8`** model on Google Kubernetes Engine (GKE) targeting **Cloud TPU v6e-4** (4 chips, 2x2 topology) with **vLLM**.

To make deployment, performance tuning, and harness integration modular, the recipe is organized into four subfolders:

---

## 1. [GKE Base Infrastructure](gke-base-infra/)
* **Purpose:** Bootstrapping raw vLLM model serving on GKE.
* **Focus:** Core infrastructure setup, TPU node selection, PersistentVolumeClaim cache setup, and baseline verification (smoke test) with a short context window.
* **Path:** [gke-base-infra/](gke-base-infra/)

## 2. [Optimized Qwen Serving](qwen-optimized-serving/)
* **Purpose:** Boosting serving throughput and latency for developer-focused, long-context workloads.
* **Focus:** Configuring persistent **Prefix Caching** (KV-cache sharing), **FP8 KV Cache**, **Chunked Prefill**, model-specific chat templates, and executing concurrent load benchmarks.
* **Path:** [qwen-optimized-serving/](qwen-optimized-serving/)

## 3. [AGI Harness Integration](agi-harness-integration/)
* **Purpose:** Connecting Google's internal **AGI CLI** agent to the local TPU Qwen serving pod.
* **Focus:** Setting up the **Translation Proxy** (Gemini-to-OpenAI translation), configuring AGI daemon compaction thresholds, and troubleshooting context limits.
* **Path:** [agi-harness-integration/](agi-harness-integration/)

## 4. [Pi Harness Integration](pi-harness-integration/)
* **Purpose:** Configuring the **Pi open-source terminal coding agent** ([pi.dev](https://pi.dev)) with local TPU Qwen serving.
* **Focus:** Direct OpenAI API integration, configuring `~/.pi/agent/models.json` for custom endpoints, and session execution commands.
* **Path:** [pi-harness-integration/](pi-harness-integration/)

---

## Recommended Deployment Order

1. Complete **Base Infrastructure** setup to verify TPU pod scheduling and checkpoint downloading.
2. Apply **Optimized Serving** parameters to enable prefix caching and run concurrency benchmarks.
3. Deploy the **AGI** (internal) or **Pi** (open-source) client integration to link your CLI agent sessions to the model server.
