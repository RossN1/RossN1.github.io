---
title: "Why Quantisation Matters for LLMs"
date: 2025-09-24T11:00:00+01:00
draft: false
ShowReadingTime: true
---

Large language models (LLMs) are powerful but expensive to run. A model with 7 billion parameters stored in FP32 requires around 28 GB of memory just for weights (7e9 × 4 bytes). For larger models such as LLaMA-70B, this rises into the hundreds of gigabytes. These numbers explain why LLMs usually run in datacentres on specialised GPUs with high memory bandwidth.

If we want to deploy models on consumer hardware or edge devices, these requirements are too high. Laptops and phones rarely have more than a few gigabytes available for AI tasks, and embedded systems often have less. 

**Quantisation is one of the main ways to close this gap.**


## What quantisation does

Quantisation reduces the precision of numbers stored in the model. Instead of FP32, we use FP16, INT8, or INT4. This has three direct effects:

1. **Smaller memory footprint**  
   - A 7B model at FP16 is ~14 GB.  
   - At INT8, it’s ~7 GB.  
   - At INT4, it drops to ~3.5 GB.  

   These differences are the difference between “cannot load” and “runs on a single GPU or CPU”.

2. **Faster inference**  
   Modern hardware executes low-precision arithmetic faster and moves less data through memory channels. INT8 or INT4 kernels can achieve higher throughput.

3. **Lower power use**  
   Memory access is expensive in terms of energy. With fewer bytes to move, battery life on mobile or embedded devices improves.

---

## Why edge deployment makes this important

For cloud providers, the trade-off is between efficiency and accuracy. For edge devices, it’s often **feasibility**:

- **Mobile apps**: chat assistants, transcription, or summarisation directly on a phone.  
- **IoT devices**: local reasoning without sending data to the cloud.  
- **Privacy-sensitive settings**: healthcare or finance, where keeping data local avoids exposure.  

Quantisation makes these scenarios realistic.

---

## The cost: accuracy trade-offs

Reducing precision introduces quantisation error. If a weight distribution has many outliers, compressing it to 4 bits can distort the range, leading to degraded fluency or higher perplexity. Accuracy loss depends on the scheme used and the model architecture.

Some approaches leave sensitive layers in higher precision (embeddings, normalisation layers) while quantising others. Others combine quantisation with retraining so the model learns to be robust.  

---

## Outlook

Quantisation is not the only approach — pruning and distillation are also active research areas — but it is the most direct route to shrinking a model while keeping the same architecture. In the next post, I’ll break down how quantisation works in practice, the different schemes used, and the tools available today.
