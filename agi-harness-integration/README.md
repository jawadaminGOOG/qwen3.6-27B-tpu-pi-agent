# AGI Harness Integration with Qwen serving

This recipe documents how to connect the **AGI CLI** (Google's internal agentic assistant daemon) to the optimized **`Qwen3.6-27B-FP8`** server running on Cloud TPU.

## Architecture

AGI is designed to work with the Gemini API. To run it with a local, open-weights model like Qwen served by vLLM (which implements the OpenAI API), we run a local **Translation Proxy** that intercepts AGI's Gemini API requests, translates them to OpenAI API format, queries the vLLM server, and streams the responses back in the Gemini format.

```
+------------+       Gemini API       +-------------------+       OpenAI API       +---------------------+
| AGI Client | =====================> | Translation Proxy | =====================> | vLLM TPU Serving    |
| (Terminal) | <===================== | (agi_proxy.py:8001)| <===================== | (GKE Cluster:8000)  |
+------------+   streamGenerateContent+-------------------+   v1/chat/completions  +---------------------+
```

## Setup & Configuration

### 1. Run the Translation Proxy
The proxy script [`agi_proxy.py`](agi_proxy.py) must run in an environment that has network access to both your local machine (where you run AGI commands) and the GKE vLLM service port-forward.

1. Port-forward the GKE service:
   ```bash
   kubectl port-forward -n qwen-serving svc/vllm-qwen-service-tp4 8000:8000
   ```
2. Start the proxy (runs on port `8001` by default):
   ```bash
   python3 agi_proxy.py 8001
   ```
   *Tip: Run this in a persistent multiplexer like `tmux` (`tmux new -s proxy 'python3 agi_proxy.py 8001 > /tmp/proxy.log 2>&1'`).*

### 2. Configure AGI Daemon
To route AGI requests to the proxy:

1. Ensure the AGI daemon runs with `GEMINI_API_ENDPOINT` pointing to the proxy:
   ```bash
   export GEMINI_API_ENDPOINT=http://localhost:8001
   agi stop && agi start --foreground
   ```
2. Set the compaction threshold in your global AGI configuration file (`~/.agi/config.yaml`):
   ```yaml
   compact-threshold: 20000
   ```

---

## Important Tuning & Troubleshooting

### 1. The Safety Buffer & Max Tokens Capping
To prevent vLLM `400 Bad Request` out-of-context errors, the proxy dynamically calculates available output space:
$$\text{available\_output\_tokens} = 32,768 - \text{estimated\_prompt\_tokens} - 3,000\text{ (Safety Buffer)}$$

* If prompt tokens reach the upper context boundary, the proxy caps `max_tokens` to `1`.
* If an empty response occurs, run `/compact` to prune conversation history.

### 2. The Token Estimation Mismatch (Compaction Troubleshooting)
AGI's daemon estimates tokens using Gemini character ratios (~4.0 chars/token), whereas Qwen's tokenizer uses ~3.2 to 3.4 chars/token.
* **The Issue:** The history size in vLLM can physically cross 20K tokens *before* the daemon's internal estimate realizes it has crossed the `20000` threshold. If `/compact` fails with `"History too small to compact"`, send a short text turn (e.g., `"continue"`) to update the daemon's internal character count and retry `/compact`.

### 3. Tool Declaration Bloat
AGI sends all loaded tool declarations (like MCP server tools) to the model. Loaded MCP tools can add ~15,000 characters (~4.5K tokens) to every request. Be selective with loaded tools to preserve active context window space.
