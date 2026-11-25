# Dynamic Speculative Decoding

## Why is Dynamic SD needed?

SD methods need to verify K tokens for each sequence during decoding. As BS increases, the effective BS becomes BS\*K which increases the compute requirement during verification. When this BS\*K goes beyond a critical BS then SD negatively impacts the decode speed (TPOT). DSD helps by tuning the K to an optimal value such that we continue to reap the benefits from SD.

## Use cases

* Variable concurrency workload using same deployment. K would decrease as concurrency increases.
* During RL rollout where we start off with high BS but then end up with small BS due to very few long tail request which end up generating a lot of tokens stalling the progress of the current rollout. Here K would go up during the end of rollout.

## `--speculative-config` schema

To use Dynamic SD, add `num_speculative_tokens_per_batch_size` to the config of an SD method which is a list of list. Here, an entry is `[start_bs, end_bs, optimal_K]` which means when the concurrency is within range `[start_bs, end_bs]` then `optimal_K` number of draft tokens are used. For e.g.,

```bash
--speculative-config '{
    "method": "eagle",
    "model": "yuhuili/EAGLE-LLaMA3.1-Instruct-8B",
    "num_speculative_tokens": 3,
    "num_speculative_tokens_per_batch_size": [
      [1, 64, 3],
      [65, 128, 1],
      [129, 512, 0]
    ]
  }'
```

implies that:

* K=3 will be used when the concurrency is in range [1, 64]
* K=1 will be used when the concurrency is in range [65, 128]
* K=0 will be used when the concurrency is in range [129, 512], i.e., no draft tokens will be produced.

## Online Examples

### Dynamic SD Eagle Drafter

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --speculative-config '{
    "method": "eagle",
    "model": "yuhuili/EAGLE-LLaMA3.1-Instruct-8B",
    "num_speculative_tokens": 3,
    "num_speculative_tokens_per_batch_size": [
      [1, 64, 3],
      [65, 128, 1],
      [129, 512, 0]
    ]
  }'
```

### Dynamic SD Eagle3 Drafter

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --speculative-config '{
    "method": "eagle3",
    "model": "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B",
    "num_speculative_tokens": 3,
    "num_speculative_tokens_per_batch_size": [
      [1, 16, 5],
      [17, 32, 4],
      [33, 64, 3],
      [65, 128, 1],
      [129, 512, 0]
    ]
  }'

```

## Changing K at runtime

A server or `LLM` started with speculative decoding can lower its draft count without a restart, for example to stop drafting in an RL run once the policy has moved away from the drafter, and resume after a draft weight update. List the counts to switch to in `runtime_num_speculative_tokens` at startup; CUDA graphs are captured for each of them at every batch size, so a step at a listed count runs as fast as an engine started with it:

```bash
--speculative-config '{
    "method": "eagle3",
    "model": "RedHatAI/Qwen3-8B-speculator.eagle3",
    "num_speculative_tokens": 3,
    "runtime_num_speculative_tokens": [0, 1]
  }'
```

```python
llm.set_num_speculative_tokens(0)     # stop drafting
llm.set_num_speculative_tokens(1)     # 1 draft token per step
llm.set_num_speculative_tokens(None)  # back to num_speculative_tokens
```

With `VLLM_SERVER_DEV_MODE=1`, `POST /set_num_speculative_tokens` with `{"num_speculative_tokens": 0}` does the same (`null` restores the configured count); a count that is neither `num_speculative_tokens` nor listed returns HTTP 400. The count applies from the next scheduling step until it is changed again. With a batch-size schedule, it caps the schedule's count. With K=0 the drafter still processes the new tokens of each step, so its KV cache stays current and drafting resumes with no stale context.

## Limitations

* Tested with Eagle, Eagle-3, and DFlash. Other SD methods may or may not work out of the box
* Not compatible with data parallelism (`--data-parallel-size > 1`) for MoE models. Their DP ranks step together, but each schedules independently, so ranks can pick different K values, causing DP collective divergence and deadlocks. For MoE models with DP, vLLM automatically disables `num_speculative_tokens_per_batch_size` and falls back to the static `num_speculative_tokens` value. Dense models run each DP rank as an independent engine, so each rank uses its own schedule.
