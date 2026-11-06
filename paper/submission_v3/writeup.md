## Abstract

We study why Gemma 4 31B, as a coding agent with thinking off, sends the same tool call again and again until its budget runs out. On 109 tasks in the harness of Kaggle's Gemma 4 Developer Agent competition, at least half the calls are repeats or malformed in a quarter of task runs, and these runs resolve 20% of the time, against 37%. Using one-step replays at fixed histories and whole-task runs under three levers, we obtain three findings. (1) The chance of one more identical call rises from 78% after two calls to 99% after ten even with a fixed request, showing that accumulating copies is unnecessary for increasing persistence. Fixed-history edits independently show that identical copies reinforce immediate repetition. (2) One-step escapes can be induced: where the model, free to think, chose not to, a forced thought lets the next call escape 58% of the time, against 11%. (3) None of the three whole-task interventions produced a detectable improvement in resolution under the tested budget.

## Paper and supporting code

[Full research paper (PDF)](https://github.com/usp787/Gemma4_paper_track/blob/main/paper/submission_v3/submission_v3.1.pdf)

[Code, experiment records, and reproduction instructions](https://github.com/usp787/Gemma4_paper_track)
