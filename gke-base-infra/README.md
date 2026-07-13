# Base vLLM serving on GKE with Cloud TPU v6e-4

This recipe provides the base setup for deploying a model server using **vLLM** on Google Kubernetes Engine (GKE) targeting **Cloud TPU v6e-4** (4 chips, 2x2 topology).

This is a generic infrastructure bring-up recipe. It does not include model-specific optimizations (such as prefix caching or custom chat templates). It is intended as a starting point to verify your TPU node pool and GKE workload deployment.

## Prerequisites

1.  A GKE cluster with GCS Fuse CSI driver enabled.
2.  A TPU v6e-4 node pool configured in your zone.
3.  `kubectl` credentials configured locally.

## Deploying the Base Server

1.  (Optional) Create a Hugging Face secret if using restricted models (though the default `Qwen3.6-27B-FP8` is public, vLLM may still check):

    ```bash
    export HF_TOKEN=YOUR_TOKEN
    kubectl create namespace vllm-serving
    kubectl create secret generic hf-secret \
        -n vllm-serving \
        --from-literal=hf_api_token=${HF_TOKEN}
    ```

2.  Apply the base deployment manifest:

    ```bash
    kubectl apply -f vllm-tpu-base.yaml
    ```

    The default manifest is configured for a conservative **4K context** (`--max-model-len=4096`) to ensure fast compilation and verify basic functionality.

3.  Monitor the pod status:

    ```bash
    kubectl get pods -n vllm-serving -w
    ```

    Once the pod status is `Running`, check the startup and compilation logs:

    ```bash
    kubectl logs -n vllm-serving deployment/vllm-tpu-serving -f
    ```

    Compilation for a 4K context typically takes 2-3 minutes. The pod will report `1/1 READY` once the OpenAI server is listening.

## Verifying the Workload

1.  Port-forward the service to your local machine:

    ```bash
    kubectl port-forward -n vllm-serving svc/vllm-tpu-service 8000:8000
    ```

2.  Submit a simple curl test request:

    ```bash
    curl http://localhost:8000/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{
            "model": "Qwen/Qwen3.6-27B-FP8",
            "messages": [{"role": "user", "content": "Hello! Write a short sentence."}],
            "max_tokens": 32,
            "temperature": 0
        }'
    ```

    If the server returns a valid JSON response containing the model's text, your base TPU infrastructure is verified.
