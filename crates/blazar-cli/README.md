# blazar

The `blazar` binary: a multi-engine local inference platform. One process
routes GGUF, safetensors, quantized, and MLX model directories to the right
engine lane (llama.cpp, mistral.rs, SGLang, MLX) and serves them through
OpenAI- and ollama-compatible APIs, with durable jobs, federation, speech,
vision, and image lanes behind the same CLI.

```sh
cargo install blazar
blazar serve
```

- [Repository and documentation](https://github.com/santanu20/blazar)
- License: MIT OR Apache-2.0
