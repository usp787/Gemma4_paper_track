# agent/: the agent and the request proxy behind every whole-task run

These are the two pieces of our main-track setup that the paper's whole-task runs depend on. They are copied unchanged from our main-track repo, which stays private until the main competition ends on 2026-12-02:

| Here | Source in the main-track repo | Commit |
|---|---|---|
| [baseline_nothink_args/](baseline_nothink_args/) | `agents/baseline_nothink_args/` | `ebc522f` (2026-10-10) |
| [token_proxy.py](token_proxy.py) | `baseline/token_proxy.py` | `bd82ee8` (2026-10-19) |

Neither file changed after those commits. Paths named inside them, such as `baseline/run_eval.py`, `eval_gpu.slurm` and `post_train/…`, are in the main-track repo.

## The agent: `baseline_nothink_args`

This is the control's agent in the paper (§2), called `off_args` in our records. It is a Kaggle submission for the competition's harness (`adk_submission`, Google ADK):

- **Model:** `gemma-4-31b-it-qat-w4a16-ct`, with thinking off (`thinking_budget: 0`).
- **Prompts:** the host's sample agent, with one change: rules for tool-call arguments in both prompts ("Tool Calls: Arguments That Work" in `prompts/system.md`, "Tool Calls" in `prompts/analyzer.md`). The header of `agent.yaml` gives the reasons, measured in an earlier run's traces.
- **Sampling** (`configs/sampling.yaml`): T = 0.2 and top-p 0.95. Top-k 64 comes from the model's generation config.
- **Budgets** (`eval_config.yaml`): 4 minutes, 60 tool calls and 100 turns per task.
- **Tools:** the harness's tools, plus a code-analyzer sub-agent (`sub_agents/code_analyzer.yaml`) that the 31B never called in our runs.

The runs used this directory zipped, as `baseline_nothink_args_20261010-1852_ebc522f.zip` in [../loops/arms.tsv](../loops/arms.tsv).

## The levers: how each arm differs from the control

| Arm (paper) | arms.tsv | The only change | Made by |
|---|---|---|---|
| Control | `ctrl` | none | the zip above |
| T = 1 | `ctrl_t1` | the temperature line of `configs/sampling.yaml` | [../loops/make_t1_zip.py](../loops/make_t1_zip.py) |
| Empty thought before every call | `block` | the chat template that vLLM serves (`--chat-template`) | [../loops/make_template.py](../loops/make_template.py) |
| Proxy: request held fixed | `dedup` | `token_proxy.py --dedup 2` between the harness and vLLM | [token_proxy.py](token_proxy.py) |
| Gemma 4 12B | `ctrl_12b` | the two `model:` lines (agent and sub-agent) | [../loops/make_model_zip.py](../loops/make_model_zip.py) |

## The proxy: `token_proxy.py`

It sits between the harness and vLLM and records every chat call's exact prompt and completion token ids in `calls.jsonl`. Two things read those records:
- the one-step replays, which start from them;
- [../loops/loop_report.py](../loops/loop_report.py), which uses them to check what the template and the proxy did. It computes the hazard from the harness's traces.

With `--dedup N`, before forwarding each request it collapses every run of N or more identical tool turns in a row to the run's first turn:
- A tool turn is an assistant message with tool calls, followed by one result per call.
- Identical means the same tool names, the same arguments (compared as JSON) and the same result contents.
- The harness still runs every call and keeps it in its trace. Only what the model sees changes.
- The paper's proxy arm uses `--dedup 2`. While a run lasts, the model gets the request that followed the run's first call, so every step of the run is a fresh draw from the same prompt.

The core is the pure function `dedup_messages(msgs, min_run)`, which has no dependencies beyond the standard library. The server needs `fastapi`, `uvicorn` and `httpx`, which come with vLLM:

```bash
python token_proxy.py --upstream http://127.0.0.1:8001 --port 8002 --out calls.jsonl --dedup 2
```

Point the harness's model endpoint at port 8002. Posting `{"instance_id": ...}` to `/_task` before each task tags the records with the task; our runner did this.

## Serving

We ran 4 tasks at once on one A100 80 GB: one vLLM server, with one proxy per task lane in front of it. vLLM came from the competition's wheelhouse v31 and was served with the scorer's flags:

```bash
python -m vllm.entrypoints.openai.api_server --model <checkpoint> --served-model-name gemma-4-31b-it-qat-w4a16-ct \
  --max-model-len 32768 --dtype auto --gpu-memory-utilization 0.90 --max-num-seqs 16 \
  --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4 \
  --reasoning-config '{"reasoning_start_str": "<|channel>", "reasoning_end_str": "<channel|>"}' \
  --no-scheduler-reserve-full-isl
```

- The 12B adds `--hf-overrides '{"architectures": ["Gemma4ForCausalLM"], "model_type": "gemma4"}'`.
- The `block` arm adds `--chat-template always_block.jinja`.

## Not included

- **The harness, the model checkpoints, the wheelhouse and the tasks:** they come from the competition's [data page](https://www.kaggle.com/competitions/gemma-4-developer-agent/data), and its rules don't allow redistributing them.
- **Our cluster launcher and task runner** (`eval_gpu.slurm`, `run_eval.py`): they wrap the host's harness for our SLURM cluster. Everything they change about a run is listed above.
