# Clarification: number of writeups per team and eligibility across awards

<p>Hi, thanks for hosting the Paper Track. A few questions about the rules so we can plan our submissions correctly:</p>
<ol>
<li>The rules say a team may select up to two Final Submissions for judging. For the Paper Track, does that mean a team can have at most two writeups judged, or can a team submit a separate writeup for each of the three awards (Best Paper, Best New Resource, Best New Application)?</li>
<li>Can a single writeup be considered for, or win, more than one of the three awards?</li>
<li>Are Best New Resource and Best New Application scored with the same five criteria (novelty, quality, relevance, verifiability, clarity) as Best Paper, or judged separately?</li>
<li>For a resource entry, is it acceptable to publish a dataset that is rebuilt from the public upstream GitHub repositories (fastapi, rich, requests, httpx), such as regenerated code graphs and embeddings, given the rule against redistributing Competition Data?</li>
</ol>
<p>Thank you!</p>

## Hersh Mittelman — comment 3542975, reply to topic
2026-10-22T18:13:20.360Z

<p><a href="/elanmark" target="_blank">@elanmark</a> Thanks for confirming that a team can submit at most two write-ups. When we create a second writeup, the form says: "You've already submitted to this competition. To submit this writeup, you'll first need to unsubmit your previous writeup." Is there a way for one team to have two writeups submitted at the same time, or should each team plan on one?</p>

## Shiva Shankar — comment 3542235, reply to topic
2026-10-21T16:19:25.920Z

<p>Hi <a href="/Elan" target="_blank">@Elan</a> Markowitz, thank you for your answer to Q4 on 29 Sep ("I believe that should be acceptable with citation and with linking to the Kaggle dataset … I will double check our exact rules on this front"). Before we publish, could you confirm whether the release below complies with rule 2.4.b?</p>
<p>The planned Kaggle dataset contains only:</p>
<ul>
<li>152 code graphs (JSON), one per parent commit of 152 merged public pull requests in fastapi, httpx, requests and rich, rebuilt from public GitHub with our open-source builder (module/class/function nodes with signatures and docstrings; calls, contains, imports, inherits and tests edges).</li>
<li>304 embedding files: CodeRankEmbed vectors (float16 .npz) plus metadata for those graphs.</li>
<li>pr_ids.tsv: repository, PR number and GitHub URL for the 152 PRs. No PR titles, bodies or other PR text.</li>
<li>pr_gold.jsonl: for each PR, the files and function/class names it changed. No PR text.</li>
<li>The builder, schema and benchmark scripts (Apache-2.0), with upstream licence notices (MIT, Apache-2.0, BSD-3-Clause).</li>
</ul>
<p>It contains no Competition Data: no task statements, gold or test patches, official graphs or embeddings, and no graphs built at any official task base commit. We excluded every PR whose parent or merge commit matches an official base commit, and an automated scan for official commit hashes and instance ids reports 0 hits. The dataset will cite and link the competition dataset.</p>
<p>Can you explicitly confirm that publishing this dataset under Apache-2.0 (with the upstream notices) complies with rule 2.4.b? Thank you.</p>

## Elan Markowitz — comment 3545676, reply to 3542235
2026-10-27T17:41:36.987Z

<p>Yep can confirm!</p>

## Anh-Vu Mai-Nguyen — comment 3532989, reply to topic
2026-10-05T08:06:48.267Z

<p>Thank you for the answers above. +1 to the open word-count question from Fasil and dval: do the title, abstract, figure/table captions and the reference list count toward the 3,000-word limit? One related case: if a paper is provided as a PDF through the Public Project Link rather than in the Writeup body, does the same limit and counting apply? Thank you.</p>

## Elan Markowitz — comment 3545675, reply to 3532989
2026-10-27T17:38:49.263Z

<p>The PDF will have the same word limit. All "judged" text is included in the word count except figures and references. The PDF will be judged on its own, ignoring the kaggle writeup component. But yes, title and abstract will, captions, and tables will count towards the word count. </p>

## Elan Markowitz — comment 3530171, reply to topic
2026-09-29T17:38:53.917Z

<p>Great questions! </p>
<ol>
<li>A team can submit at most two write-ups. There is only one submission track, and the judges will determine the eligibility for each prize. We are looking for quality over quantity. </li>
<li>Yes. A single write-up can be considered for more than one award, however, it will only be eligible to win one. If it is Best Paper it will win Best Paper and then no longer be eligible for the resource or application prize.</li>
<li>The same criteria will be applied to each, but in the context of that prize. Meaning the <em>novelty, quality, and relevance</em> will be scored in terms of the new resource or application. </li>
<li>I believe that should be acceptable with citation and with linking to the Kaggle dataset. If you publish new graphs and embeddings that link to the Kaggle dataset, you would not be redistributing the public dataset. I will double check our exact rules on this front. </li>
</ol>

## Fasil — comment 3531238, reply to 3530171
2026-10-01T14:38:02.713Z

<p>Thanks for these answers. Two follow-ups for my entry:</p>
<p>(1) Data release. May a public Kaggle dataset linked from my Writeup contain agent run traces from the public training tasks? The traces include code from the task repositories (fastapi, rich, requests, under their open-source licences) that the agent printed while working, plus the agent's own patches. I would remove every problem statement and hint, and include a script that restores them from the competition data for people who have accepted the rules. Gold patches and test patches would never be included. If that isn't acceptable, may I release only my labels, tables and scripts?</p>
<p>(2) Word count. How is the 3,000-word limit counted? Do references, tables and figure captions count, and is there an official counter?</p>
<p>Thank you!</p>

## Elan Markowitz — comment 3545672, reply to 3531238
2026-10-27T17:37:03.727Z

<p>Yes, it can include such traces. Please cite and reference the upstream license for the data. </p>

## dval — comment 3531983, reply to 3530171
2026-10-02T23:58:26.407Z

<p>Following up on your answer to question 4: has the permission to release regenerated graphs now been confirmed against the competition rules? We are preparing a Paper Track resource that audits code graphs and generates comparison graphs from public upstream repositories at recorded pre-fix commits. We would cite the competition and upstream licenses, and would not release the provided graphs or embeddings, problem statements, gold patches, or evaluation test patches. Could you confirm whether independently regenerated graph files (including upstream code text where its license allows this) may be made publicly available with the Writeup? If not, we would release only our own audit/generation code, synthetic fixtures, and aggregate results, with instructions for eligible participants to obtain the competition assets themselves. Are there additional restrictions on that code-and-aggregates-only release?</p>
<p>I also support Fasil’s question about how the 3,000-word limit counts references, tables and figure captions, and whether an official counter is available.</p>

## Elan Markowitz — comment 3545670, reply to 3531983
2026-10-27T17:36:28.593Z

<p>Yes regenerated graph files is allowable with appropriate licensing. </p>
<p>Word limit includes table and captions but not figures and references. </p>

## Shiva Shankar — comment 3533924, reply to 3530171
2026-10-07T03:41:22.310Z

<p>Thank you <a href="/elanmark" target="_blank">@elanmark</a> for the detailed and clear answers on team limits, award cross-eligibility, and upstream graph releases. This makes planning our two final write-ups much more straightforward.</p>
<p>Echoing the follow-ups from <a href="/Fasil" target="_blank">@Fasil</a>, <a href="/dval" target="_blank">@dval</a>, and <a href="/Anh-Vu" target="_blank">@Anh-Vu</a> Mai-Nguyen regarding the writeup submission mechanics:</p>
<ol>
<li><strong>Word Count Scope:</strong> Could you clarify whether references, figure/table captions, and mathematical appendices are excluded from the 3,000-word count, or if there is an official counter used during submission?</li>
<li><strong>PDF vs. Markdown Formats:</strong> For entries providing a complete publication-grade PDF via the Public Project Link (such as on arXiv or GitHub), does the Kaggle writeup body serve as an executive summary, with the full PDF considered as the primary artifact for judging?</li>
</ol>
<p>Thanks again to the Google Research team for organizing this track!</p>

## Elan Markowitz — comment 3545666, reply to 3533924
2026-10-27T17:32:04.903Z

<p>All words of the "judged" text count towards the word count including captions and tables (but not figures and references). We do not yet have an official word counter released. If the paper relies on mathematical proofs, please include those or shortened versions in the main paper. Supplementary material not count towards the word limit. </p>
<p>In the case of a PDF submission, we will just be judging the PDF as the writeup. Recommend using the Kaggle writeup as an abstract. However, it will not be used for judging and will not count towards the word count. </p>

## agrover112 — comment 3528198, reply to topic
2026-09-25T16:25:33.057Z

<p>These are interesting questions! I second this.</p>
