# AGI Harness Integration with Qwen serving

This recipe documents how to connect the **AGI CLI** (Google's internal agentic assistant daemon) to the optimized **`Qwen3.6-27B-FP8`** server running on Cloud TPU.

## Architecture

AGI interacts with model endpoints via the Gemini API standard. To connect it to a local open-weights model served by vLLM (which exposes an OpenAI API endpoint), a local **Translation Proxy** intercepts AGI's Gemini API requests, converts them into OpenAI format, queries the vLLM server, and streams responses back in Gemini format.

```
+------------+       Gemini API       +-------------------+       OpenAI API       +---------------------+
| AGI Client | =====================> | Translation Proxy | =====================> | vLLM TPU Serving    |
| (Terminal) | <===================== | (agi_proxy.py:8001)| <===================== | (GKE Cluster:8000)  |
+------------+   streamGenerateContent+-------------------+   v1/chat/completions  +---------------------+
```

## Setup & Configuration

### 1. Run the Translation Proxy
The proxy script [`agi_proxy.py`](agi_proxy.py) runs locally to bridge network traffic between the AGI daemon and the GKE vLLM service port-forward.

1. Port-forward the GKE service:
   ```bash
   kubectl port-forward -n qwen-serving svc/vllm-qwen-service-tp4 8000:8000
   ```
2. Start the proxy (runs on port `8001` by default):
   ```bash
   python3 agi_proxy.py 8001
   ```
   *Tip: Run this in a persistent process manager or multiplexer (`tmux new -s proxy 'python3 agi_proxy.py 8001 > /tmp/proxy.log 2>&1'`).*

### 2. Configure AGI Daemon
To route AGI requests to the proxy:

1. Point the `GEMINI_API_ENDPOINT` environment variable to the proxy:
   ```bash
   export GEMINI_API_ENDPOINT=http://localhost:8001
   agi stop && agi start --foreground
   ```
2. Configure the compaction threshold in `~/.agi/config.yaml`:
   ```yaml
   compact-threshold: 20000
   ```

---

## Important Tuning & Troubleshooting

### 1. The Safety Buffer & Max Tokens Capping
To prevent vLLM out-of-context errors, the proxy dynamically calculates available output token capacity:

```
available_output_tokens = 32,768 - estimated_prompt_tokens - 3,000 (Safety Buffer)
```

* If prompt tokens reach the upper context boundary, the proxy caps `max_tokens` to `1`.
* If an empty response occurs, run `/compact` to prune conversation history.

### 2. Token Estimation & Compaction
AGI's daemon estimates tokens using a standard character-to-token ratio (~4.0 chars/token), whereas Qwen's tokenizer is more dense (~3.2 to 3.4 chars/token).
* If `/compact` reports `"History too small to compact"`, send a short message turn (e.g., `"continue"`) to update the internal character count before retrying `/compact`.

### 3. Tool Declarations Overhead
Active MCP tools add definitions to every request (~4.5K tokens). Be selective with loaded tool extensions to maximize available conversation history space.
