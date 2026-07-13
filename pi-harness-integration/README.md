# Pi Harness Integration with Qwen serving

This recipe documents how to configure the **Pi open-source terminal coding agent** ([pi.dev](https://pi.dev) / `@mariozechner/pi-coding-agent`) to connect directly to the optimized **`Qwen3.6-27B-FP8`** server running on Cloud TPU v6e.

## Direct OpenAI API Integration (No Proxy Required)

Unlike Gemini-native tools, **Pi natively supports custom OpenAI-compatible API providers**. 

Because vLLM exposes standard OpenAI-compatible endpoints (`/v1/chat/completions`), **no translation proxy layer is needed**. Pi connects directly to the vLLM serving endpoint on port `8000`.

```
+--------------------+         OpenAI API (/v1)         +-----------------------+
| Pi Coding Agent    | ==============================> | vLLM TPU Serving      |
| (Terminal / pi.dev)| <============================== | (GKE Cluster:8000)    |
+--------------------+   POST /v1/chat/completions      +-----------------------+
```

---

## Step-by-Step Configuration

### 1. Port Forward the GKE vLLM Service

Forward the GKE vLLM service port to your local machine:

```bash
kubectl port-forward -n qwen-serving svc/vllm-qwen-service-tp4 8000:8000
```

### 2. Configure Pi via `~/.pi/agent/models.json`

Add the custom TPU vLLM provider to your Pi agent configuration file at `~/.pi/agent/models.json`:

```json
{
  "providers": {
    "tpu-vllm": {
      "baseUrl": "http://localhost:8000/v1",
      "api": "openai-completions",
      "apiKey": "none",
      "models": [
        {
          "id": "Qwen/Qwen3.6-27B-FP8",
          "name": "Qwen 3.6 27B FP8 (Cloud TPU v6e-4)",
          "contextWindow": 32768,
          "maxTokens": 4096
        }
      ]
    }
  }
}
```

---

## Launching & Running Pi with Qwen

Once configured, launch Pi specifying the custom Qwen model ID:

```bash
pi --model Qwen/Qwen3.6-27B-FP8
```

Alternatively, launch Pi interactively and select the model using `/model`:
```bash
pi
```

---

## Optimizations & Performance for Pi Workloads

1. **Prefix Caching Advantage:** Pi issues tool execution loops (`read`, `write`, `edit`, `bash`) against your codebase. Because the GKE server runs with `--enable-prefix-caching`, prompt instructions and codebase context remain cached in TPU HBM, delivering Time to First Token (TTFT) latency under **~500ms** per tool step.
2. **Context Window Management:** Pi tracks context usage based on `contextWindow: 32768`, managing truncation cleanly within Qwen's physical limit.
