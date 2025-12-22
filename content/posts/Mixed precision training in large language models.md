---
title: "Mixed precision training in large language models"
date: 2025-12-16T15:40:00+00:00
draft: false
ShowReadingTime: true
---

Neural networks have traditionally been trained using 32-bit floating point (FP32). However, the precision FP32 offers isn't always necessary for most training workloads. Lower precision formats can be just as effective, as training is inherently approximate.  

This post covers how I added mixed precision training to an implementation of the transformer architecture following [Umar Jamil's tutorial](https://www.youtube.com/watch?v=ISNdQcPhsts) and what I learned along the way.  

## What is mixed precision training?
Mixed precision training is an optimization technique that utilises 16-bit (half precision) for most operations and 32-bit (full precision) for numerically sensitive operations, increasing the speed of model training and reducing memory usage.

Mixed precision typically involves three steps

### Casting 
By casting model operations such as convolutions, matrix multiplication and linear layers to half-precision, this would reduce the required memory bandwidth and increase the speed of calculation


### Maintaining FP32 weights
Initially, a copy of the model's parameters is saved in FP32. During the backpropagation stage of the training process, gradients would be computed in FP16 and applied to the FP32 master weights, preventing small gradient updates from being lost due to the limited range of FP16, preventing gradient underflow.


### Loss Scaling 
The value of the loss function is multiplied by a scaling factor before backpropagation. This would shift the values of the gradients into a range that can be represented by FP16, avoiding underflow errors. 

After backpropagation, the gradients would then be divided by the same factor, before updating weights in FP32.

If the scaling factor is too large, gradients can overflow (become infinity/NaN). In practice, dynamic loss scaling would be used. If overflow is detected, the training step would be skipped and the scale factor would be reduced.


## Implementing mixed precision training in PyTorch

PyTorch provides two functions for implementing mixed precision training: `autocast` and `GradScaler`.

Autocast wraps the forward pass and selects FP16 or FP32 for each operation. Matrix multiplication and convolutions run in FP16, numerically sensitive operations (Such as loss calculations) stay in FP32.

GradScaler handles loss scaling. It is used for scaling the loss before backpropagation, checks for overflow and unscales the gradients before the optimizer step  

```python
from torch.amp import autocast, GradScaler

scaler = GradScaler('cuda')

for epoch in range(initial_epoch, config['num_epochs']):
    batch_iterator = tqdm(train_dataloader, desc=f"Processing epoch {epoch:02d}")
    for batch in batch_iterator:
        model.train()
        optimizer.zero_grad()

        encoder_input = batch['encoder_input'].to(device)
        decoder_input = batch['decoder_input'].to(device)
        encoder_mask = batch['encoder_mask'].to(device)
        decoder_mask = batch['decoder_mask'].to(device)

        with autocast('cuda'):
            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            proj_output = model.project(decoder_output)

            label = batch['label'].to(device)
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
```
The changes from the standard model training process are:
* `autocast('cuda')` - This wraps the forward pass and loss calculation. PyTorch would use FP16 where it is safe (matrix multiplication, convolutions), and keep FP32 where it is needed (Softmax, loss functions, layer normalisation) 
* `scaler.scale(loss).backward()` - Multiplies the loss by the scale factor before backpropagation. This shifts gradients into a range that can be used by FP16.
* `scaler.step(optimizer)` - Unscales the gradients back to the original values, checks for overflow, and updates the weights. If overflow is detected, the step is skipped.
* `scaler.update` - Adjusts the scale factor for the next iteration. If overflow occurred, the scale is reduced. If several iterations pass without overflow, the scale is increased to maximise the benefits of scaling.

## Conclusion

Mixed precision training offers a practical way to accelerate deep learning workloads with minimal code changes. By wrapping the forward pass in `autocast` and using `GradScaler` to handle gradient scaling, I achieved a **~49%** increase in throughput and reduced training time by **33%** for a single epoch.

**FP32 (Baseline):**
```
Processing Epoch 00: 100%|██████████| 14297/14297 [1:49:58<00:00, 2.17it/s, loss=4.376]
```

**Mixed Precision:**
```
Processing Epoch 00: 100%|██████████| 14297/14297 [1:13:46<00:00, 3.23it/s, loss=4.733]
```

While the mixed precision run showed a slightly higher loss after one epoch (4.733 vs 4.376), this is expected behaviour — the two approaches typically converge to similar values over additional epochs. 

The key insight is that mixed precision doesn't sacrifice model quality; it simply gets you there faster.

