---
title: "Deploying Quantised LLMs"
date: 2025-09-24T09:00:00+01:00
draft: false
ShowReadingTime: true
---

In the first two posts, I explained why quantisation matters and how it works. Here I’ll cover how to deploy quantised large language models (LLMs) on edge hardware, what workflows are available, and what to measure in practice.

---

## Deployment workflows

### 1. Hugging Face Transformers + bitsandbytes (GPU or desktop)

This is the most common path if you have a consumer GPU. `bitsandbytes` integrates with Transformers to load models directly in 8-bit or 4-bit precision.

Example:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)

bnb_config = BitsAndBytesConfig(load_in_4bit=True)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

inputs = tokenizer("Hello world", return_tensors="pt").input_ids
outputs = model.generate(inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
````

This loads the 7B model in 4-bit precision. On a consumer GPU with 12–16 GB VRAM, that can be the difference between failing to load and running smoothly.

---

### 2. llama.cpp + GGUF (CPU or edge devices)

For devices without a GPU, `llama.cpp` is a lightweight inference framework written in C++. It runs quantised models on CPU and supports Apple Silicon, ARM boards, and even mobile phones.

Steps:

1. **Convert the model**
   Convert a Hugging Face checkpoint into GGUF format with chosen quantisation (e.g. Q4).

   ```
   python convert-hf-to-gguf.py --model llama-2-7b --out llama-2-7b-q4.gguf
   ```

2. **Run inference**

   ```
   ./llama.cpp/llama-cli -m llama-2-7b-q4.gguf -p "Explain quantisation in simple terms"
   ```

The GGUF format supports a range of quantisation types (Q2, Q4, Q8). Lower-bit variants reduce memory further but can affect quality.

---

### 3. ONNX Runtime / TensorRT (accelerated inference)

For deployment on embedded GPUs or NPUs, ONNX Runtime and TensorRT both support quantised models. These are more common in production pipelines where integration with existing inference stacks is needed.

---

## What to measure

When deploying quantised models, I focus on four metrics:

1. **Latency**: tokens per second generated.
2. **Memory usage**: peak RAM/VRAM during inference.
3. **Output quality**: usually measured by perplexity or comparison against benchmarks.
4. **Power consumption**: relevant on mobile or embedded hardware.

A practical example:

* A 7B model in FP16 may run at \~10 tokens/sec on a laptop GPU and consume \~14 GB VRAM.
* The same model in 4-bit quantisation can drop to \~3.5 GB VRAM and \~20 tokens/sec, with only a small drop in fluency.

---

## Trade-offs and pitfalls

* **Accuracy loss**: INT8 is usually safe; INT4 sometimes reduces fluency or coherence. Hybrid precision (e.g. embeddings in FP16, attention weights in INT4) can help.
* **Hardware support**: INT8 kernels are common, INT4 less so. llama.cpp handles this internally, but other runtimes may not.
* **Calibration**: PTQ benefits from representative input data to compute scales. Poor calibration increases error.
* **Security risks**: research has shown that quantisation can hide adversarial triggers in weights. While not a barrier to use, it’s a factor for production systems.

---

## Future directions

Research is pushing quantisation further:

* **Mixed precision**: combining FP16, INT8, and INT4 within a model.
* **Outlier handling**: methods like SpQR preserve rare but important weights in higher precision.
* **Rotation-based schemes**: SpinQuant aligns weight spaces for better quantisation.
* **Integration with pruning and distillation**: combining techniques to maximise efficiency.
* **Hardware support**: new accelerators are emerging with native INT4 instructions.

---

## Closing thoughts

Quantisation is a trade-off: you exchange some model quality for a smaller footprint and faster execution. For edge deployment, it often makes the difference between “impossible” and “usable”. With careful choice of bit width, calibration, and tools, it’s possible to run models that once needed datacentres directly on laptops or embedded devices.


---
