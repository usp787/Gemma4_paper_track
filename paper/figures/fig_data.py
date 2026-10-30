"""fig_data.py: every number the overview figures plot, with the record it is quoted from.

Written 2026-10-24 for paper/overview.html. These are transcriptions from the records in ../../ablation/,
../../loops/ and ../../findings.md, not new measurements. The paper's final figures should be rebuilt from the
report files on Explorer ($R/paper/loops/report_12b_p2_1020-2212/loops.json and baseline_*.json), which hold
the same tables; SOURCES says where each block below comes from and whether that is a report table or a
local split (a scratchpad analysis whose script is not in this repo yet).
"""

# Fig 1b. P(one more identical call | the last k calls were identical, the last one well formed), clean steps
# (no budget warning in the last result, no compaction before the next call). (rate, steps)
HAZARD_BINS = ["1", "2", "3–4", "5–9", "10+"]
HAZARD = {
    "ctrl_31b": [(0.024, 4985), (0.678, 115), (0.958, 142), (1.000, 290), (1.000, 1052)],
    # k >= 2: the "fully collapsed" row (the request exactly the one after the run's first call); k = 1 has no
    # run to collapse, so it is the all-clean-steps value.
    "dedup_31b": [(0.022, 5142), (0.779, 104), (0.869, 145), (0.930, 257), (0.993, 811)],
    "ctrl_12b": [(0.058, 4487), (0.687, 243), (0.950, 302), (0.990, 572), (0.997, 1805)],
}

# Fig 2. G1, the dose test: p(c) at the run's sampling, 60 fixed histories in 56 tasks (v30 histories).
DOSE_COPIES = [1, 2, 4]
DOSE = {
    "identical": [0.002, 0.098, 0.434],   # dose1, dose2, dose4
    "respelled": [0.002, 0.027, 0.038],   # dose1, para2, para4 (para shares dose1's start)
}
DOSE_EXACT = {"identical": [0.006, 0.052, 0.307], "respelled": [0.006, 0.009, 0.015]}
DOSE_DIFF = {  # paired, 95% intervals from 10,000 resamples of tasks
    "dose2 - dose1": (0.096, 0.040, 0.165),
    "dose4 - dose2": (0.336, 0.231, 0.443),
    "dose4 - para4": (0.376, 0.258, 0.496),
    "para4 - dose1": (0.036, -0.002, 0.092),
}
DOSE_CLOSED_LOOP = {"after 2 identical": 0.775, "at 3+": 0.957}  # ctrl_v30, the closed loop on the same build

# Fig 3. Paired differences against ctrl (lever - ctrl), 109 tasks, 95% intervals from 2,000 resamples of tasks.
# (estimate, low, high); stale and resolved in percentage points, run length in calls (runs of 3+).
LEVERS = {
    "T = 1": {"stale": (-18.2, -24.8, -11.0), "run_length": (-8.63, -12.76, -4.53), "resolved": (-0.5, -5.0, 4.1)},
    "always-block": {"stale": (1.6, -4.5, 7.6), "run_length": (-8.95, -12.63, -4.80), "resolved": (1.8, -3.7, 7.8)},
    "dedup": {"stale": (-1.3, -6.0, 3.1), "run_length": (-0.41, -4.73, 4.02), "resolved": (0.5, -3.7, 4.6)},
    "12B": {"stale": (14.8, 8.9, 20.7), "run_length": (-0.08, -3.92, 3.60), "resolved": (-22.0, -29.8, -15.1)},
}
RESOLVED_OF_218 = {"ctrl": 72, "T = 1": 71, "dedup": 73, "12B": 24}  # always-block: 38 of 109, one pass

# Fig 4. How runs end: (own exit, budget warning, compaction, task end).
EXIT_ARMS = ["31B control", "T = 1", "always-block", "dedup", "12B control"]
EXITS_3PLUS = {  # report tables "How runs of 3+ end", passes summed
    "31B control": (16, 58, 33, 9),
    "T = 1": (36, 20, 21, 7),
    "always-block": (62, 18, 11, 5),
    "dedup": (43, 41, 7, 17),
    "12B control": (25, 63, 60, 44),
}
EXITS_10PLUS = {  # local splits: all runs of 10+, ablation_G4.md §5.2 and ablation_control.md §5.2
    "31B control": (1, 43, 23, 5),
    "T = 1": (2, 10, 13, 3),
    "always-block": (18, 10, 6, 3),
    "dedup": (6, 31, 2, 15),
    "12B control": (4, 51, 34, 30),
}

# Fig 5. T-open, one step ahead: escape = the next call is neither stale nor malformed. (escapes, states)
TOPEN = {
    "skipped": {"T-think": (12, 114), "T-open": (66, 114), "diff": (47.4, 33.0, 61.1), "tasks": 36},
    "thought": {"T-think": (61, 99), "T-open": (62, 99), "diff": (1.0, -6.7, 9.1), "tasks": 34},
}

SOURCES = {
    "HAZARD": "ablation_G4.md §5.1 (ctrl; dedup's 'fully collapsed' row for k >= 2, its 'all clean steps' "
              "value at k = 1, changed 2026-10-30 to match the paper's fixed-request claim) and "
              "ablation_control.md §5.1 (ctrl_12b, both passes). Local splits.",
    "DOSE": "ablation_G1.md §3, from $R/paper/loops/dose_baseline_nothink_args_v30_zip/dose.md. Report table.",
    "LEVERS": "ablation_G3.md §4 (T = 1, always-block), ablation_G4.md §4 (dedup), ablation_control.md §4 (12B). "
              "Report tables (loops.md paired differences).",
    "EXITS_3PLUS": "ablation_G3.md §3, ablation_G4.md §3, ablation_control.md §3. Report tables.",
    "EXITS_10PLUS": "ablation_G4.md §5.2 and ablation_control.md §5.2. Local splits.",
    "TOPEN": "loops/loops_reason_10_13.md §5 (from gate1.md §4 in Gate 1's work dir). 20,000 resamples of tasks.",
}
