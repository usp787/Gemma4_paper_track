# Start prompt: the paper's outline and figure list

<!-- drafted 2026-10-21 for a new Claude Code session in this repo, when the experiment round ended. Paste everything below the line, or start the session with "Read paper/outline_prompt.md and follow it." -->

---

You are starting the planning round of our Kaggle paper. You have none of the earlier conversations, so this prompt and the files it names are your context. This prompt was written on 2026-10-21, when the last experiment (the 12B control) was recorded and the experiment round ended.

## How this round works (read this first)

This is a planning and discussion round, not a writing round. The user wants to agree on the plan with you before anything gets written.
- **Your first reply:** read the files below, then answer in the chat with the proposal described under "Your first reply". Write no files. Then stop and wait.
- **Later turns:** discuss and revise. When you recommend something, say why, and name the alternative you rejected. When you disagree with the user, say so once with your reason, then follow their decision.
- **Only when the user says the outline is agreed,** write `paper/outline.md` (its contents are at the end of this prompt).
- **Not in this round, unless the user asks:** paper prose, figures, plotting code, new analyses. Planning a figure doesn't mean drawing it.
- **Never:** run experiments, queue jobs, commit or push. The user commits by hand. Read-only checks on Explorer are allowed if a question needs them (memory note `explorer-access`), with light commands on the login node only.

## The paper's constraints

- **Venue:** Google's "Gemma 4 Developer Agent" paper track on Kaggle, entering for Overall Best Paper as an empirical paper. A writeup can win at most one award.
- **Length:** at most **3,000 words**. Earlier sessions assumed that the title, abstract, captions and references count. Nobody has checked this on Kaggle's pages, so raise it as an open question.
- **Deadline:** **2026-11-12 23:59 UTC**. The plan from 2026-10-13 was writing from Oct 27 to Nov 5, then review and corrections from Nov 6 to 10. The experiments ended early, so writing can start sooner.
- **Required content:** title and subtitle, abstract, introduction, a description of the research (methods and experiments), related work and citations.
- **Format:** either the Kaggle Writeup body (markdown, with figures as images) or a public link to an arXiv-ready PDF. Not decided yet.
- **Judging:** five criteria, scored equally:
  - Novelty: "new insights, deepen understanding, or highlight important properties of existing methods";
  - Quality, which the rubric defines as generality beyond the competition;
  - Relevance to software engineering and agentic learning;
  - Verifiability;
  - Clarity.
- **The host on AI-written text:** "Purely LLM generated 'slop' will not score well"; LLM editing used "tastefully" is fine. How the user and the assistant split the writing is the user's decision; ask.
- **Competition data can't be redistributed** (rule 2.4). That limits what a figure, an example or a code release may show. Ask before planning a figure around a real trace excerpt.
- **Where the rules are:** [README.md](../README.md), sections "What we submit", "How papers are judged" and "Rules that matter". Its "Goal" and "Primary direction" sections still describe the OPSD plan of 2026-10-07. The paper has been about the loop study since 2026-10-12.

## What we found

[findings.md](../findings.md) is the source for every claim and number. Read it in full. Each of its numbers is quoted from a record, and its §4 lists the claims withdrawn along the way, so don't revive those. The ten findings, one line each:
1. Loops are common (a quarter of task runs), and they mark lost runs (20% against 37% resolved).
2. They start where the model is already near certain of the call: with the request held fixed, the agent re-sends the call 78% of the time after its first copy and 99% by the 10th. Most of the lock-in curve is selection (G4).
3. The copies reinforce, on their exact tokens, and close the last escapes (G1, G4).
4. At T = 0.2, long runs end only from outside: at the budget warning, a compaction or the task's end (G3).
5. The pattern of the context drives the copy, not what the results say (G2).
6. The model skips thinking where a thought would help: a forced thought escapes 58% against 11% (T-open, one step ahead).
7. Every whole-task lever changes the loops, not the outcome: resolution stays at 71–73 of 218 (G3, G4).
8. At T = 0.2, which tasks loop is a property of the task (only in part across sizes).
9. The self-teacher shares the loop, which is why the study left OPSD (Gate 1).
10. Size changes how often loops start, not how firmly they hold (the 12B control).

In one sentence (findings.md): the loop is a symptom of a state the model is already sure about, and it is held in place by its own copies; removing the loop doesn't recover the task.

## What to read, in this order

1. [findings.md](../findings.md), all of it.
2. [../loops/literature.md](../loops/literature.md):
   - §1, the bottom line: loops are known, so the paper is a mechanism-and-levers study, not a discovery;
   - §5, the must-cite list;
   - §6, the implications: framing, risks, and what judges will expect;
   - its suggested name, "tool-call loops", with onset, return and lock-in (§2).
   - Fork-and-Flush (arXiv 2610.07447) isn't in it yet: memory note `fork-and-flush-paper`.
3. The answer at the top of each record: [ablation_G1.md](../ablation/ablation_G1.md) to [ablation_G4.md](../ablation/ablation_G4.md) and [ablation_control.md](../ablation/ablation_control.md). Open their sections only when a claim or a figure needs the detail.
4. [../loops/loops_reason_10_13.md](../loops/loops_reason_10_13.md) §5 (T-open), only for finding 6.
5. [../gate1/README.md](../gate1/README.md) and [../opsd_assessment.md](../opsd_assessment.md), only for finding 9.

## Where figure data lives

With `$R = /scratch/zha.j/gemma4` on Explorer (findings.md §8 has the full table):
- **Every closed-loop arm in one report:** `$R/paper/loops/report_12b_p2_1020-2212/`. It pools `ctrl` ×2, `block`, `ctrl_t1` ×2, `dedup` ×2 and `ctrl_12b` ×2, with round 1's v30 runs.
  - `loops.md` and `loops.json` hold the arm tables, the paired differences, lock-in by run length (all and clean steps), how runs end, and pass to pass.
  - `baseline_dev.md`, `baseline_rich.md` and their `.json` hold resolution per run and call hygiene.
- **G1 and G2:** `$R/paper/loops/dose_baseline_nothink_args_v30_zip/` and `$R/paper/loops/fb_baseline_nothink_args_v30_zip/`.
- **T-open:** gate1.md §4 in Gate 1's work dir.
- **Numbers that exist only in local splits:** G4's fully-collapsed curve, own exits by run length, and the 12B's splits. These came from scripts in session scratchpads, outside the repo.
  - A figure built on them needs its script brought into the repo first, which is a verifiability decision.
  - Prefer a report table where one is close enough. For example, the report's `dedup` clean row (0.79, 0.88, 0.93 and 0.99 at k = 2 to 10+) is close to the fully-collapsed curve (0.78, 0.87, 0.93 and 0.99).

## Your first reply

Fit it on about one screen, in short bullets. Bring:
1. **What the paper can claim,** in at most 5 bullets. Give each its status from findings.md: measured, observed or interpretation.
2. **Two or three framings,** each with a working title and a one-sentence thesis. Recommend one, judged against the five criteria. Some seeds; propose better ones if you see them:
   - *mechanism:* "a stuck agent is a confident agent". Loops start in states the model is sure of, its copies close the escapes, and only the outside ends them.
   - *measurement:* the lock-in hazard, with a fixed-request control, as the way to measure tool-call loops and to separate selection from reinforcement.
   - *levers:* "breaking the loop doesn't fix the task". Levers compared one step ahead and over whole tasks.
3. **A section skeleton with a word budget** that sums to the cap. Say what you counted as words.
4. **A figure and table list,** aiming at 4–6 items. For each item:
   - what it shows, and the finding it carries;
   - its source file, and whether it exists as a report table or needs a new script;
   - must or nice to have.
   Then say which numbers stay in the text.
5. **What to cut:** findings, arms or details that don't earn their words. Candidates: the OPSD story, round 1's v30 runs, the malformed-call slip, the always-block arm.
6. **The open questions for the user,** numbered, each with your default answer.

Then stop and wait.

## Questions the user will want to decide

Use these to seed item 6, and add your own:
- **Format:** Kaggle Writeup body, or an arXiv-ready PDF? This decides the figure tooling and how references look.
- **The word rule:** what counts toward the 3,000 words (captions, tables, references)? Re-check Kaggle's pages before budgeting to the word.
- **The lead result:** the hazard curve (selection and reinforcement), or the levers' null on resolution?
- **OPSD:** a motivating paragraph (finding 9), a footnote, or nothing?
- **The 12B:** a section of its own, or a size note on each claim?
- **The T-open result:** it is one step ahead only, with no closed-loop forced-thought arm. Lead with it, or keep it as a supporting result with its caveat?
- **Verifiability:** what goes public?
  - candidates: the repo with its scripts and report files, or a Kaggle notebook;
  - the constraint: no competition data;
  - the local-split scripts would need moving into the repo first.
- **Examples:** may a figure show a real loop, such as a tool-call sequence from one of our traces on a public upstream repo?
- **Terms:** keep "tool-call loops" with onset, return and lock-in?
- **Writing split and schedule:** who drafts which sections, the date the outline must be agreed, the first full draft, and when to create and save a Kaggle writeup early. A save bug in the Writeup editor was reported on 2026-09-24.

## What `paper/outline.md` should hold (only once the user agrees)

1. The title, the subtitle, and the one-sentence thesis.
2. The sections with their word budgets. For each section: the claims it makes (by finding number) and its evidence (record and §).
3. The figures and tables, with a one-line draft caption each, their source files, and what each one still needs.
4. The must-cite list (literature.md §5, plus Fork-and-Flush), mapped to sections.
5. The decisions taken, each with its date, and the items still open.
6. The writing schedule to 2026-11-12.

## Rules for references

- Resolve every article that has a DOI through the LibKey connector before listing it.
- arXiv-only preprints: LibKey doesn't index them (memory note `libkey-arxiv`). Write "Full text not available through your institution. Open copy: https://arxiv.org/abs/<id>".
- Present only the final metadata and links; don't narrate lookups.
- In this round, references are only mapped to sections. Re-verify them only if the user asks.

## When the outline is agreed

- Add a memory note on the paper's plan: the decisions and the outline's path.
- Point the "The paper" line in findings.md §7 at `paper/outline.md`.
