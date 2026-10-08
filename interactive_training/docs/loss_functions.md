# Loss Functions

Interactive Training computes losses **inside the Interactive Training training backend** from the token-level tensors you ship inside each `Datum.loss_fn_inputs`. You pick the loss with `loss_fn=<name>` on `forward_backward`; the required `loss_fn_inputs` keys depend on which one you pick. This page is the reference for what each built-in loss does, what it expects on the wire, and — for research and experimentation — how `forward_backward_custom` lets you bring your own differentiable loss with **no backend changes**.

> [!NOTE]
> The wire-format `Datum`/`TensorData`/`loss_fn_inputs` types are documented in [sdk-reference.md § Datum, ModelInput, TensorData](./sdk-reference.md#datum-modelinput-tensordata). The train-loop perspective on per-token masking lives in [training.md § Data format and loss masking](./training.md#data-format-and-loss-masking).

## Built-in loss functions

All built-ins require `target_tokens` in `loss_fn_inputs` (the input tokens shifted left by one — i.e. the label at each position). The other required keys vary. All formulas below are what the Interactive Training training backend computes for you:

| `loss_fn` | Reduction | Use case | Extra `loss_fn_inputs` keys | One-line formula |
|---|---|---|---|---|
| `cross_entropy` | **sum** | SFT, distillation, **building block for custom losses** | `weights` (float per token) | $L = -\sum_t w_t \cdot \log p_\theta(y_t \mid x_{<t})$ |
| `importance_sampling` | **sum** | RL — default for on-policy / near-on-policy | `advantages`, `logprobs` | $L = -\sum_t A_t \cdot \rho_t$ where $\rho_t = \exp(\log p_\theta - \log p_{\text{sampler}})$ |
| `ppo` | **configured normalization** | RL with clipped policy updates | `advantages`, `logprobs`; clipping configuration is backend-specific | Per-token $\ell_t=-\min\!\left(A_t\rho_t,\;A_t\,\mathrm{clip}(\rho_t,1{-}\epsilon_l,1{+}\epsilon_h)\right)$, then normalized reduction |
| `cispo` | **configured normalization** | RL — stabilized IS variant | `advantages`, `logprobs` | Uses a detached clipped ratio as the weight on $-A_t\log p_\theta$, rather than differentiating through that ratio |
| `sapo` | **configured normalization** | RL — soft adaptive gating | `advantages`, `logprobs` | Smooth sigmoid gate on the IS ratio with separate positive/negative advantage temperatures; sequence-mean normalization is a common configuration |

The PPO minimum is taken **after multiplying each branch by the signed advantage**; moving $A_t$ outside the minimum is wrong for negative advantages. Normalized losses can depend on the model/backend's token or sequence reduction. Do not infer a universal `loss_fn_config` key/default from an algorithm paper: confirm the selected service contract; unsupported overrides can be rejected.


> **Sign convention.** Interactive Training reports the **minimization** objective: every formula above is what the backend `.backward()`-s. For RL, advantages enter with their natural sign — positive $A_t$ pushes $\log p_\theta(y_t)$ up.

### Mixing loss functions within one optimizer step

Reduction affects both loss scale and accumulated gradients. The sum-reduced losses above differ from the configured normalized losses; service training configurations can normalize these groups differently:

| Reduction group | Loss functions | Loss contribution |
|---|---|---|
| **sum** | `cross_entropy`, `importance_sampling` | sums the weighted token contributions |
| **configured normalization** | `ppo`, `cispo`, `sapo` | normalizes by the applicable token/sequence rule; not necessarily the same scalar denominator for every backend |

> [!IMPORTANT]
> **For compatibility across Interactive Training models, calls accumulated before one `optim_step` should use the same reduction group.** Sessions that normalize the groups differently reject crossing groups without an intervening `optim_step`:
>
> ```
> Cannot accumulate a mean-reduced loss because sum-reduced gradients are already
> pending in this optimizer step. This session normalizes the two reduction
> groups differently. Call optim_step() before switching between sum-reduced
> losses (cross_entropy, importance_sampling) and mean-reduced losses
> (ppo, cispo, sapo).
> ```
>
> This is a guard against a silently wrong update, not an arbitrary restriction: applying a mean-normalization divisor to summed gradients (or failing to apply it to mean-reduced ones) corrupts the optimizer step.

The following is operation-order pseudocode, not runnable client code. In the
cookbook wrapper, await `forward_backward_async` / `optim_step_async` to obtain
each future, then await that future's `result_async()` for completion as shown below.

```text
# ❌ Rejected — sum-reduced then mean-reduced, no optim_step between
training_client.forward_backward(batch, loss_fn="cross_entropy")   # sum
training_client.forward_backward(batch, loss_fn="ppo")             # mean -> error

# ✅ Fine — accumulate freely within one group
training_client.forward_backward(batch, loss_fn="cross_entropy")
training_client.forward_backward(batch, loss_fn="importance_sampling")
training_client.optim_step(adam_params)

# ✅ Fine — optim_step flushes gradients and resets the group
training_client.forward_backward(batch, loss_fn="cross_entropy")
training_client.optim_step(adam_params)
training_client.forward_backward(batch, loss_fn="ppo")
```

If you are iterating over loss functions to compare them (a common first experiment), call `optim_step` — or start a fresh session — between each one.

`forward_backward_custom` uses `cross_entropy` for its surrogate backward pass, so it belongs to the **sum** reduction group for this accumulation rule.

See [training.md § Loss functions](./training.md#loss-functions) for picking between the RL losses in practice. The rest of this page is about the escape hatch: `forward_backward_custom`.

## When to reach for `forward_backward_custom`

**Prefer a built-in when it covers your loss.** `forward_backward_custom` costs ~1.5× the FLOPs of a built-in `forward_backward` (one extra forward pass — see "The math, in one identity" below) and trades a few small client-side constraints. Reach for it only when the built-ins above can't express your objective.

This is the **research escape hatch**: when you want to experiment with a loss that isn't one of the built-ins but is differentiable in the model's per-token log-probabilities. The public [DPO recipe](../interactive_training/recipes/preference/dpo/) is one concrete use; custom objectives can follow the same adapter contract. Representative use cases include:

- **Novel preference / pairwise losses** — compare $\log p_\theta$ of a preferred vs. a rejected completion on the same gradient step.
- **Distillation with a non-trivial teacher signal** — reverse-KL, top-$K$ matching, on-policy distillation.
- **Custom regularizers** — e.g. KL-to-reference penalties expressed in closed form on $\log p_\theta$.
- **Confidence / entropy shaping** — penalties or bonuses computed from the model's own logprobs.
- **Anything else** — any scalar loss you can express as a Python function of `logprobs` tensors and differentiate locally with autograd.

If your loss can be expressed as $L = \sum_t w_t \cdot (-\log p_\theta(y_t))$ for some fixed per-token weights $w_t$ (depending on the batch but **not** on $\log p_\theta$ itself), you don't need this — just use `cross_entropy` with the right `weights`. `forward_backward_custom` is for losses whose weights would depend on the model's own current logprobs.

## How it works

`forward_backward_custom` is **not a new backend route**. The cookbook adapter composes two ordinary backend calls around one local `loss.backward()`:

| Step | Where | What |
|---|---|---|
| 1. Forward | **Backend** (`forward(loss_fn="cross_entropy")`) | Runs the model, returns per-token logprobs $\ell_t = \log p_\theta(y_t \mid x_{<t})$. |
| 2. Local autograd | **Client** (Python, CPU) | Wraps each $\ell$ in a leaf tensor with `requires_grad=True`, calls your `loss_fn(data, logprobs_list)`, runs `loss.backward()`, reads $g_t = \partial C / \partial \ell_t$ from `.grad`. |
| 3. Surrogate backward | **Backend** (`forward_backward(loss_fn="cross_entropy")`) | Submits the same batch with `weights = -g`. Backend backprops cross-entropy through the model; staged parameter gradients are *exactly* $\partial C / \partial \theta$. |

### Why step 3 produces the right gradients

The backend's `cross_entropy` loss is

$$L_\mathrm{CE}(w; \theta) = \sum_t -\ell_t(\theta) \cdot w_t.$$

Differentiating with respect to $\theta$ via chain rule gives

$$\frac{\partial L_\mathrm{CE}}{\partial \theta} = \sum_t \frac{\partial \ell_t}{\partial \theta} \cdot (-w_t).$$

The user's loss $C(\ell_1, \ldots, \ell_N)$ has, by chain rule,

$$\frac{\partial C}{\partial \theta} = \sum_t \frac{\partial \ell_t}{\partial \theta} \cdot \frac{\partial C}{\partial \ell_t} = \sum_t \frac{\partial \ell_t}{\partial \theta} \cdot g_t.$$

Setting $w_t = -g_t$ makes the two expressions identical. **The identity is exact**, not an approximation, as long as the backend's cross-entropy is $L = \sum_t -\ell_t \cdot w_t$ with no clamping on $w$ — which it is in Interactive Training.

**Cost.** One extra forward pass over the model per custom forward/backward call (roughly 1.5× under idealized forward/backward FLOP arithmetic). This is **not a measured wall-clock or price guarantee**: request/polling latency, serialization, and the size/complexity of local autograd can dominate. The local backward differentiates token-logprob tensors, not the service's model parameters.

## Writing a custom loss

Your loss is a plain Python function `(data, logprobs_list) -> (loss, metrics)`. See `forward_backward_custom_async` in [interactive_training/rl/train_azure.py](../interactive_training/rl/train_azure.py) for the exact type and a worked example in [interactive_training/recipes/math_rl/train_azure.py](../interactive_training/recipes/math_rl/train_azure.py).

You hand it to the adapter, await the future, and step the optimizer:

```python
future = await training_client.forward_backward_custom_async(data, my_custom_loss_fn)
result = await future.result_async()
optimizer = await training_client.optim_step_async(adam_params)
await optimizer.result_async()

# result.metrics contains the dict your loss returned, merged with any
# backend-reported metrics from the surrogate forward_backward.
```

**What you never see.** The adapter takes `loss.backward()` on your behalf, reads off `g = ∂loss/∂logprobs` for each datum, builds the surrogate `Datum`s with `loss_fn_inputs = {"target_tokens": ..., "weights": -g}`, and submits the second `forward_backward`. You did not have to construct that weight vector; you only wrote the scalar `loss` from your Python function.

**What you do see.** `result.metrics` is the dict *you* returned, merged with any backend-reported metrics from the surrogate `forward_backward` **and** with three adapter-injected diagnostic stats on the surrogate weight vector:

| Metric | Meaning | What vanilla CE would show |
|---|---|---|
| `surrogate/weights_l2` | $\|\,w\,\|_2$, where $w_t = -\partial C/\partial \ell_t$ | $\sqrt{N}$ |
| `surrogate/weights_pos_frac` | Fraction of tokens with $w_t > 0$ | $1.0$ |
| `surrogate/weights_cos_uniform` | $\cos(w, \mathbf{1})$ — direct "is this just CE in disguise" signal | exactly $1.0$ |

These are batch-level numbers computed once per `forward_backward_custom_async` call on a tensor you already have in memory, so the cost is negligible. They are the most direct evidence in your training logs that the custom path is doing something differential from cross-entropy: if `weights_cos_uniform` stays near `1.0` your loss is collapsing to CE; if `weights_pos_frac` stays near `1.0` advantages aren't flowing through. User-returned metric keys override on collision, so a custom loss can replace any of these if it has a better measurement. The backend will also report a `loss` value from its `cross_entropy` computation on the surrogate batch — see constraint 4 below for why you should ignore it and trust your own metrics.

## Masking

Masking out prompt tokens, padding, or any other positions you don't want to contribute to the gradient is **free** — it falls out of the math, no separate wire-level mask is needed.

The surrogate weight is $-g_t = -\partial C / \partial \ell_t$. If your loss $C$ doesn't depend on $\ell_t$ at masked positions, then $g_t = 0$ there automatically, so the weight shipped to the server is $0$ at those positions and they contribute exactly nothing to the parameter gradient.

You have three equivalent places to apply the mask:

1. **Fold it into a precomputed aux quantity** in a repacker step before calling `forward_backward_custom_async` (e.g. precompute `w_t = mask_t * adv_t * exp(-lp_sampling_t)` so the Python loss becomes `C = -sum_t w_t * exp(lp_t)`).
2. **Multiply by the mask inside the Python loss**, carrying the mask through a named aux key on the `Datum`.
3. **Carry the mask as its own aux field** (e.g. `"mask": TensorData(...)`) and multiply by it inside the loss — the most readable option for pure SFT-style losses with no advantage to fold into.

Two things to keep in mind:

- **Do not rely on the `Datum.loss_fn_inputs["weights"]` field for masking.** The adapter owns that field on the surrogate-backward call and overwrites it with $-g$. Any mask you stash there in the original `Datum` is discarded; only its effect on $g$ (via your Python loss) matters.
- **Aux keys never reach the server.** As noted in constraint 1 below, the adapter strips everything except `target_tokens` (and the adapter-owned `weights`) from both the forward and the surrogate-backward payloads. Carry as many named per-token quantities (`"mask"`, `"reference_logprobs"`, `"dapo/sampling_lp"`, …) as you need; they exist purely for your closure.

## Constraints to be aware of

1. **`loss_fn_inputs` must contain `target_tokens`; arbitrary extras are permitted but never reach the server.** Any other keys (advantages, sampler logprobs, masks, reference logprobs, …) are passed through to your `loss_fn` closure but stripped from the wire payload. This is the **multi-aux-key pattern**: each per-token quantity gets its own named field, and your closure reads it back via `tensor_data_to_torch(data[i].loss_fn_inputs["my_key"])` (see [`interactive_training.tensor_utils`](../interactive_training/tensor_utils.py)). It is the only way to carry more than one per-token auxiliary signal, since `weights` is a single 1-D field. `weights` is optional on input — if absent, the adapter injects zeros for the forward call; only the surrogate-backward `weights` (`= -∂C/∂ℓ`) carry gradient information.
2. **`loss_fn` must compute a gradient for every datum's logprobs.** If `logprobs_list[i].grad` is `None` after `loss.backward()` — because your loss didn't reference `logprobs_list[i]` — the adapter raises `ValueError`. Silently substituting zeros would mask indexing bugs in your loss function.
3. **The custom loss must be differentiable in the logprobs.** Pure functions of $\theta$ that go around the logprobs (e.g. weight-decay-style penalties on raw parameters) cannot be expressed this way — the backend is the only thing that sees $\theta$, and the surrogate trick only routes gradients via $\ell_t$.
4. **The `forward` and `forward_backward` calls are submitted sequentially within the helper.** Keep the same model weights for both; an interleaved `optim_step` invalidates the surrogate-gradient identity. The helper awaits its forward result, but does not lock out other coroutines using that session. Serialize updates and custom-loss calls on the same session rather than assuming the helper provides cross-coroutine isolation.
5. **Custom losses can produce poorly-scaled `weights`.** If $\partial C / \partial \ell$ has large magnitudes, the backend-reported `cross_entropy` loss value on the surrogate call will look bizarre (it's $\sum -\ell_t \cdot (-g_t)$, a quantity with no meaningful interpretation). Parameter gradients are still correct; only the loss-value metric is misleading. Trust the metrics your `loss_fn` returns over the backend's auto-reported cross-entropy loss in this path.
6. **Numerical cost: ~1.5× a built-in `forward_backward`.** One extra forward pass per training step.
7. **`TensorData.data` is a flat `list[int]` or `list[float]`.** Token IDs use integers; weights, logprobs, and advantages use floats. The cookbook's custom-loss helpers use 1-D tensors. Carrying several separate per-token quantities is supported through auxiliary keys (constraint 1); arbitrary multi-dimensional server-side loss shapes are not.
8. **Host-RAM autograd runs on the full substep batch.** Step 2 materializes one 1-D logprobs tensor per datum and differentiates all of them together. Internal chunking would break custom cross-datum terms such as batch-level normalizers. Leaf storage is approximately `num_datums · avg_target_tokens · 4` bytes for float32, plus gradients and objective-dependent autograd intermediates; measure peak RAM rather than assuming a fixed multiplier. Raising RL `num_substeps` reduces each call's data, **but changes the update schedule**: the shared loop issues an optimizer step for each substep. It is not equivalent to full-batch gradient accumulation, and a cross-datum custom loss can also change when split. Re-evaluate the objective and training result before using it as a memory workaround.

## Related reading

- [training.md § Loss functions](./training.md#loss-functions) — picking between built-in losses in practice.
- [sdk-reference.md](./sdk-reference.md) — full API for `forward_backward_async`, `forward_async`, and `forward_backward_custom_async`.
- [`interactive_training/recipes/math_rl/`](../interactive_training/recipes/math_rl/) — production usage: the `custom_loss` CLI flag routes every `forward_backward` through `forward_backward_custom_async` with one of three registered closures (`is_py`, `squared_logprob`, `dapo`).
