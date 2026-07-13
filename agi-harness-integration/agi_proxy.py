import json
import urllib.request
import urllib.error
from http.server import HTTPServer, BaseHTTPRequestHandler
import sys
import os

OPENAI_ENDPOINT = "http://localhost:8000/v1/chat/completions"
PROD_BASE_URL = "https://daily-cloudcode-pa.sandbox.googleapis.com"

def estimate_tokens(messages, openai_tools):
    char_count = 0
    for m in messages:
        content = m.get("content")
        if content:
            if isinstance(content, str):
                char_count += len(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and "text" in part:
                        char_count += len(part["text"])
        if m.get("tool_calls"):
            for tc in m["tool_calls"]:
                char_count += len(tc.get("function", {}).get("name", ""))
                char_count += len(tc.get("function", {}).get("arguments", ""))
    
    if openai_tools:
        char_count += len(json.dumps(openai_tools))
        
    return int(char_count / 3.2)

class ProxyHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Print to stderr for debugging
        sys.stderr.write("%s - - [%s] %s\n" %
                         (self.address_string(),
                          self.log_date_time_string(),
                          format%args))

    def do_post(self):
        print(f"\n--- Incoming POST request to {self.path} ---", file=sys.stderr)
        
        # Read the request body
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length)
        with open("/tmp/last_request.json", "w") as lf: lf.write(post_data.decode("utf-8"))

        
        try:
            envelope = json.loads(post_data.decode('utf-8'))
        except Exception as e:
            print(f"Error parsing JSON: {e}", file=sys.stderr)
            self.send_response(400)
            self.end_headers()
            self.wfile.write(f"Invalid JSON: {e}".encode('utf-8'))
            return

        # Extract model
        model_id = envelope.get("model", "")
        req = envelope.get("request", {})
        
        print(f"Model ID from request: '{model_id}'", file=sys.stderr)

        # Decide whether to handle locally (Qwen) or forward to production Gemini
        is_qwen = False
        if model_id and ("qwen" in model_id.lower() or model_id == "qwen-27b"):
            is_qwen = True

        if not is_qwen:
            # Transparently forward non-Qwen requests to production Gemini
            if self.path.startswith("/v1internal"):
                base_url = PROD_BASE_URL
            else:
                base_url = "https://generativelanguage.googleapis.com"
                
            prod_endpoint = base_url + self.path
            print(f"Forwarding non-Qwen request for model '{model_id}' to {prod_endpoint}...", file=sys.stderr)
            print(f"Incoming headers:\n{self.headers}", file=sys.stderr)
            
            headers = {}
            for key in self.headers:
                if key.lower() == 'host':
                    continue
                headers[key] = self.headers[key]
                
            request_obj = urllib.request.Request(
                prod_endpoint,
                data=post_data,
                headers=headers,
                method="POST"
            )
            
            try:
                res = urllib.request.urlopen(request_obj)
            except urllib.error.HTTPError as e:
                err_body = e.read()
                print(f"HTTP Error from Production Gemini ({e.code}): {err_body}", file=sys.stderr)
                self.send_response(e.code)
                for key, val in e.headers.items():
                    self.send_header(key, val)
                self.end_headers()
                self.wfile.write(err_body)
                return
            except Exception as e:
                print(f"Error connecting to Production Gemini: {e}", file=sys.stderr)
                self.send_response(500)
                self.end_headers()
                self.wfile.write(str(e).encode('utf-8'))
                return

            self.send_response(200)
            print(f"Headers from Prod Gemini:\n{res.headers}", file=sys.stderr)
            for key, val in res.headers.items():
                if key.lower() == 'transfer-encoding':
                    continue
                self.send_header(key, val)
            self.end_headers()

            while True:
                chunk = res.readline()
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
            print("Finished forwarding response from Production Gemini.", file=sys.stderr)
            return

        # Map model name for vLLM
        model_id = "Qwen/Qwen3.6-27B-FP8"
        print(f"Handling Qwen locally. Using model ID: {model_id}", file=sys.stderr)

        # Map Google's GenerateContentRequest to OpenAI Chat Completions API
        print(f"Incoming request from AGI: {json.dumps(req)}", file=sys.stderr)
        contents = req.get("contents", [])
        system_instruction = req.get("systemInstruction", {})

        messages = []

        # 1. System instruction
        sys_parts = system_instruction.get("parts", [])
        sys_text = "\n".join([p.get("text", "") for p in sys_parts if "text" in p])
        if sys_text:
            messages.append({"role": "system", "content": sys_text})

        # 2. Content history
        for content in contents:
            role = content.get("role")
            parts = content.get("parts", [])
            
            # Check if it contains function response first, regardless of role (Gemini uses 'user' role for tool responses)
            has_func_response = any("functionResponse" in p for p in parts)

            if has_func_response:
                for p in parts:
                    if "functionResponse" in p:
                        fr = p["functionResponse"]
                        # Find matching tool call in history to get the ID
                        call_id = None
                        for prev_msg in reversed(messages):
                            if prev_msg.get("role") == "assistant" and "tool_calls" in prev_msg:
                                for tc in prev_msg["tool_calls"]:
                                    if tc["function"]["name"] == fr["name"]:
                                        call_id = tc["id"]
                                        break
                            if call_id:
                                break
                        if not call_id:
                            call_id = f"call_{fr['name']}"
                            
                        messages.append({
                            "role": "tool",
                            "tool_call_id": call_id,
                            "name": fr["name"],
                            "content": json.dumps(fr.get("response", {}))
                        })
            elif role == "model" or role == "assistant":
                msg = {"role": "assistant"}
                tool_calls = []
                text_parts = []
                for p in parts:
                    if "text" in p:
                        text_parts.append(p["text"])
                    elif "functionCall" in p:
                        fc = p["functionCall"]
                        # Generate a unique-ish ID for this tool call in history
                        call_id = f"call_{fc['name']}_{len(messages)}"
                        tool_calls.append({
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": fc["name"],
                                "arguments": json.dumps(fc.get("args", {}))
                            }
                        })
                if text_parts:
                    msg["content"] = "\n".join(text_parts)
                if tool_calls:
                    msg["tool_calls"] = tool_calls
                messages.append(msg)
            else:
                # Treat as user
                text = "\n".join([p.get("text", "") for p in parts if "text" in p])
                if text:
                    messages.append({"role": "user", "content": text})

        # Map tools if present (needed for token estimation)
        tools = req.get("tools", [])
        openai_tools = []
        for t in tools:
            decls = t.get("functionDeclarations", [])
            for decl in decls:
                openai_tools.append({
                    "type": "function",
                    "function": {
                        "name": decl.get("name"),
                        "description": decl.get("description", ""),
                        "parameters": decl.get("parameters", {})
                    }
                })

        # Max model len for Qwen in GKE config
        MAX_MODEL_LEN = 32768
        
        # Estimate input tokens
        input_tokens_est = estimate_tokens(messages, openai_tools)
        print(f"Estimated input tokens: {input_tokens_est}", file=sys.stderr)
        
        # Available tokens for output (with a safety buffer of 3000 tokens)
        available_output_tokens = MAX_MODEL_LEN - input_tokens_est - 3000
        if available_output_tokens < 0:
            available_output_tokens = 0

        generation_config = req.get("generationConfig", {})
        max_output_tokens = generation_config.get("maxOutputTokens")
        temperature = generation_config.get("temperature")

        target_max_tokens = 4096
        if max_output_tokens is not None and max_output_tokens > 0:
            target_max_tokens = max_output_tokens
            
        capped_max_tokens = min(target_max_tokens, available_output_tokens)
        # Ensure we request at least 1 token to avoid API error if available is 0
        capped_max_tokens = max(capped_max_tokens, 1)

        openai_req = {
            "model": model_id,
            "messages": messages,
            "stream": True,
            "max_tokens": capped_max_tokens,
            "stream_options": {"include_usage": True}
        }

        print(f"Capped max_tokens from {max_output_tokens} to {capped_max_tokens} (available: {available_output_tokens})", file=sys.stderr)

        if temperature is not None:
            openai_req["temperature"] = temperature

        if openai_tools:
            openai_req["tools"] = openai_tools

        print(f"Forwarding to OpenAI: {json.dumps(openai_req)}", file=sys.stderr)

        # Forward the request to the local vLLM instance
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream"
        }
        body_data = json.dumps(openai_req).encode('utf-8')
        
        request_obj = urllib.request.Request(
            OPENAI_ENDPOINT,
            data=body_data,
            headers=headers,
            method="POST"
        )

        try:
            res = urllib.request.urlopen(request_obj)
        except urllib.error.HTTPError as e:
            err_body = e.read().decode('utf-8')
            print(f"HTTP Error from vLLM ({e.code}): {err_body}", file=sys.stderr)
            self.send_response(e.code)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(err_body.encode('utf-8'))
            return
        except urllib.error.URLError as e:
            print(f"URL Error connecting to vLLM: {e.reason}", file=sys.stderr)
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": {"message": str(e.reason)}}).encode('utf-8'))
            return

        # Send headers for SSE
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Connection', 'close')
        self.end_headers()

        # Stream response
        print("Starting streaming back to AGI client...", file=sys.stderr)
        active_tool_calls = {}
        
        while True:
            line = res.readline()
            if not line:
                print("End of stream from vLLM (no more data)", file=sys.stderr)
                break
            
            line_str = line.decode('utf-8')
            print(f"Raw line from vLLM: {line_str.strip()}", file=sys.stderr)
            if line_str.startswith("data: "):
                data_body = line_str[len("data: "):].strip()
                if data_body == "[DONE]":
                    print("Received [DONE] from vLLM", file=sys.stderr)
                    break

                try:
                    chunk = json.loads(data_body)
                except json.JSONDecodeError:
                    continue

                # Handle usage metadata from vLLM
                if "usage" in chunk:
                    usage = chunk["usage"]
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    completion_tokens = usage.get("completion_tokens", 0)
                    total_tokens = usage.get("total_tokens", 0)
                    
                    response = {
                        "usageMetadata": {
                            "promptTokenCount": prompt_tokens,
                            "candidatesTokenCount": completion_tokens,
                            "totalTokenCount": total_tokens
                        }
                    }
                    print(f"Sending usage metadata to AGI: {json.dumps(response)}", file=sys.stderr)
                    self.wfile.write(f"data: {json.dumps(response)}\n\n".encode('utf-8'))
                    self.wfile.flush()
                    continue

                choices = chunk.get("choices", [])
                if not choices:
                    continue

                choice = choices[0]
                delta = choice.get("delta", {})
                finish_reason = choice.get("finish_reason")

                # Accumulate tool calls
                tool_calls_delta = delta.get("tool_calls", [])
                for tc in tool_calls_delta:
                    idx = tc.get("index")
                    if idx not in active_tool_calls:
                        active_tool_calls[idx] = {
                            "name": "",
                            "arguments": ""
                        }
                    if "function" in tc:
                        func = tc["function"]
                        if "name" in func:
                            active_tool_calls[idx]["name"] += func["name"]
                        if "arguments" in func:
                            active_tool_calls[idx]["arguments"] += func["arguments"]

                parts = []

                # Extract reasoning
                reasoning = delta.get("reasoning") or delta.get("reasoning_content")
                if reasoning:
                    parts.append({
                        "text": reasoning,
                        "thought": True
                    })

                # Extract normal content
                content = delta.get("content")
                if content:
                    parts.append({
                        "text": content,
                        "thought": False
                    })

                if parts:
                    response = {
                        "candidates": [
                            {
                                "content": {
                                    "role": "model",
                                    "parts": parts
                                },
                                "finishReason": finish_reason
                            }
                        ]
                    }
                    self.wfile.write(f"data: {json.dumps(response)}\n\n".encode('utf-8'))
                    self.wfile.flush()

        # Send accumulated tool calls at the end of the stream
        if active_tool_calls:
            parts = []
            for idx, tc in sorted(active_tool_calls.items()):
                name = tc["name"]
                args_str = tc["arguments"]
                try:
                    args = json.loads(args_str) if args_str else {}
                except json.JSONDecodeError as e:
                    print(f"Error parsing tool call arguments JSON: {args_str}, error: {e}", file=sys.stderr)
                    args = {"raw_arguments": args_str}
                
                parts.append({
                    "functionCall": {
                        "name": name,
                        "args": args
                    }
                })
            
            if parts:
                response = {
                    "candidates": [
                        {
                            "content": {
                                "role": "model",
                                "parts": parts
                            },
                            "finishReason": "FUNCTION_CALL"
                        }
                    ]
                }
                print(f"Sending accumulated tool calls: {json.dumps(response)}", file=sys.stderr)
                self.wfile.write(f"data: {json.dumps(response)}\n\n".encode('utf-8'))
                self.wfile.flush()

        # Send DONE to AGI client
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()
        
        print("Finished POST handler, closing connection.", file=sys.stderr)
        self.close_connection = True

    def do_POST(self):
        self.do_post()

    def do_GET(self):
        print(f"\n--- Incoming GET request to {self.path} ---", file=sys.stderr)
        if "streamGenerateContent" in self.path or "generateContent" in self.path:
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok", "models": ["Qwen/Qwen3.6-27B-FP8"]}).encode('utf-8'))
            return
            
        # Forward GET to production
        prod_endpoint = PROD_BASE_URL + self.path
        headers = {}
        for key in self.headers:
            if key.lower() == 'host':
                continue
            headers[key] = self.headers[key]
        request_obj = urllib.request.Request(
            prod_endpoint,
            headers=headers,
            method="GET"
        )
        try:
            res = urllib.request.urlopen(request_obj)
            self.send_response(200)
            for key, val in res.headers.items():
                self.send_header(key, val)
            self.end_headers()
            self.wfile.write(res.read())
        except urllib.error.HTTPError as e:
            self.send_response(e.code)
            for key, val in e.headers.items():
                self.send_header(key, val)
            self.end_headers()
            self.wfile.write(e.read())
        except Exception as e:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(str(e).encode('utf-8'))

def run_server(port=8080):
    server_address = ('', port)
    httpd = HTTPServer(server_address, ProxyHandler)
    print(f"AGI-to-vLLM translation proxy running on port {port}...")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down proxy...")

if __name__ == '__main__':
    port = 8080
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            pass
    run_server(port)
