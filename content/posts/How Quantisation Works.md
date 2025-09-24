---
title: "How Quantisation Works: From FP32 to INT4"
date: 2025-09-24T10:00:00+01:00
draft: false
ShowReadingTime: true
---


In the first post I explained why quantisation matters. Here I’ll focus on how it is applied, the main schemes in use, and some of the libraries that make it practical.

---

## Precision levels

- **FP32**: full precision, used in training. High accuracy, but memory-intensive.  
- **FP16 / bfloat16**: half precision. Often the default for training and inference on GPUs.  
- **INT8**: 8-bit integers. Reduces memory by 4× compared to FP32.  
- **INT4**: 4-bit integers. More aggressive, but often still usable for inference.  

Each step down saves memory and bandwidth but increases quantisation error.

---

## Mapping values to integers

A floating point value *x* is quantised using:

```

q = round(x / scale) + zero_point

```

- **scale**: defines how much of the float range is covered by the integer range.  
- **zero_point**: shifts the mapping so zero can be represented exactly.  

To recover the value:

```

x_approx = (q - zero_point) * scale

```

This means quantisation is essentially a rounding operation.

---

## Symmetric vs asymmetric

- **Symmetric**: centred around zero, same range for positive and negative values. Efficient, but not good for skewed distributions.  
- **Asymmetric**: introduces a zero point so the integer range can shift. Better for weights/activations that aren’t balanced around zero.

---

## PTQ vs QAT

- **Post-Training Quantisation (PTQ)**: apply quantisation after training. It’s quick and works without retraining, but may lose accuracy.  
- **Quantisation-Aware Training (QAT)**: simulate quantisation during training so the model adapts. More costly, but usually better accuracy.

---

## Techniques for reducing loss

Several methods go beyond naive quantisation:

- **SmoothQuant**: shifts outlier activations into weights to make ranges easier to quantise.  
- **SpQR**: leaves outliers at higher precision and quantises the rest to 3–4 bits.  
- **SpinQuant**: rotates weight subspaces before quantisation, reducing distortion.  

These are research methods, but they demonstrate how much engineering goes into retaining accuracy at low bit widths.

---

## Libraries and frameworks

- **bitsandbytes**: integrates with Hugging Face Transformers. Supports 8-bit and 4-bit loading, and QLoRA fine-tuning.  
- **GPTQ**: a post-training method tailored for LLMs, often used for 4-bit quantisation.  
- **AWQ**: activation-aware weight quantisation, another PTQ method.  
- **llama.cpp / GGUF**: CPU-friendly inference, supporting multiple quantisation levels, widely used for edge deployment.

---

## Practical notes

- Not all layers need quantising. Embeddings and normalisation often stay in FP16.  
- Calibration with representative input data improves PTQ accuracy.  
- Hardware support matters: some CPUs/NPUs handle INT8 well, but INT4 kernels are less standard.  

---

In the next post I’ll show what this looks like in practice: using `bitsandbytes` to load a model in 4-bit precision on a GPU, and converting a model to GGUF for inference with `llama.cpp` on a CPU.

---