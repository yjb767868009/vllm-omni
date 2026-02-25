# Quantization for Diffusion Transformers

vLLM-Omni supports quantization of DiT linear layers to reduce memory usage and accelerate inference.

## Supported Methods

| Method | Guide |
|--------|-------|
| FP8 | [FP8](fp8.md) |
| Int8 | [Int8](int8.md) |

## Device Compatibility for FP8

| GPU Generation | Example GPUs | FP8 Mode |
|---------------|-------------------|----------|
| Ada/Hopper (SM 89+) | RTX 4090, H100, H200 | Full W8A8 with native hardware |

Kernel selection is automatic.

## Device Compatibility for Int8

| NPU Generation | Int8 Mode |
|---------------|----------|
| Atlas A2/A3 | W8A8 |
