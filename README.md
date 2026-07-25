# EDA Agent: Verification Under Adversarial Testing
### Lab Report / Project Documentation

**Author's context:** Data science undergraduate. Originally built and
tested entirely on Google Colab, free-tier Groq API only. **Since §2.8
below: moved off Colab, running locally** (Windows laptop, MSI GF63,
Python venv), git repo on GitHub (`.gitignore` created before the first
commit — excludes `venv/`, `.venv/`, `__pycache__/`, a `Datasets/`
folder, and `datasets.ipynb`; source datasets deliberately not
committed). `GROQ_API_KEY` set as a system env var, never committed.
Still free-tier Groq API budget only — that constraint hasn't changed,
only where the code runs.

**Objective.** Build an autonomous exploratory-data-analysis agent — given
a pandas dataframe, it investigates independently (inspect → reason →
call a tool → observe → repeat → report), using an LLM with function
calling rather than a fixed script. The engineering artifact is
secondary. **The actual research question is whether a competent-looking
agent's claims can be trusted, and if not, how to catch it systematically
rather than by eyeballing output.** Every finding below came from
diffing the agent's claims against independently-computed ground truth —
never from the agent's report "seeming" plausible.

---

## 1. System Architecture

Four components as of the local move (was three on Colab), each with a
distinct responsibility:

| File | Role | LLM calls? |
|---|---|---|
| `eda_agent.py` | Agent loop: tool-calling model investigates a dataframe and produces a report | Yes |
| `report_verify.py` | Post-hoc checker: audits a completed report against its own tool-call trail | **No — pure Python** |
| `continue_conversation.py` | Sends one additional message into an already-completed conversation, reusing prior state | Yes |
| `main.py` | Driver script — replaces the old cell-by-cell Colab workflow; runs the full pipeline end to end and writes two output files (see §2.8) | No (orchestration only) |

The separation matters: `report_verify.py` is deliberately zero-LLM.
Every fix that has actually held up under repeated testing in this
project has been a **structural** one — a new tool, a stricter return
contract, a deterministic checker — never a prompt-level instruction
alone. That pattern recurs enough below that it's stated here as the
project's working thesis, not just a footnote.

### 1.1 Model choice

`openai/gpt-oss-120b` on Groq, `temperature=0`. Chosen over
`llama-3.3-70b-versatile` after the latter showed a specific, reproduced
text-formatted-tool-call bug. `gpt-oss-120b` is Groq's own reliability
recommendation — but is **not fully immune** to malformed tool calls
either; see §2.3.

Deliberately staying single-model until current findings are stable.
Groq's `qwen/qwen3-32b` (500K TPD vs. 200K TPD for `gpt-oss-120b`) is the
leading candidate for a second model once introduced — but adding a
second model now would confound "same model, repeated runs, compare
outputs," which is the whole basis for the findings log in §3.

### 1.2 Tools available to the agent

- **`execute_python(code)`** — sandboxed execution against `df`, with
  `pd`/`np` pre-provided. Namespace persists across all tool calls
  within a single run (see Finding #4).
- **`missingness_report(col)`** — returns non-null/missing count and
  percentage for one column in a single verified call. Added
  specifically to structurally close Finding #6 (§3, item 6) after
  prompt-only fixes failed to hold across repeated concrete forms of
  the same underlying failure.
- **`compute(expression)`** — evaluates a single Python expression (or a
  batch via `expressions=[...]`) against the same shared namespace, so
  derived ratios/percentages have a real, logged tool call behind them
  instead of being computed in the model's head. See §5c/§5e for full
  design history. **Updated this round:** now includes a syntax
  auto-repair step for bare comprehensions; see §2.9.

### 1.3 Known environment quirks

- `pd.np = np` shim required — some model-generated code still uses the
  deprecated pre-1.0 `pd.np.X` pattern; without the shim this raises
  `AttributeError`.
- `.skew()` discrepancies of 3rd–4th decimal place between the agent's
  numbers and independently-computed ground truth are **expected, not
  fabrication** — pandas uses bias-corrected Fisher–Pearson skew by
  default, scipy does not.

---

## 2. Bugs Found and Fixed in the Harness Itself

Distinct from the *agent's* behavioral findings (§3) — these are bugs in
the surrounding code that would have invalidated testing if left
unfixed.

### 2.1 Tool dispatch bug
Tool-call dispatch was originally routed by an implicit assumption that
every call was `execute_python`, regardless of `tool_call.function.name`.
This silently broke `missingness_report` entirely — every call to it
returned a bogus "malformed tool call" error. Fixed by dispatching on
the actual function name.

### 2.2 Audit logging completeness
`report_verify.py` only trusts numbers that appear in `audit_log`. A
tool whose calls aren't logged causes the checker to falsely flag its
(actually verified) output as unverified. All tools now log
unconditionally on every call.

### 2.3 Malformed tool-call salvage (two distinct shapes)
`extract_code_from_tool_use_error` handles:
1. **Original (llama-3.3 era):** a JSON-quoted `"code"` key with broken
   outer structure.
2. **Discovered on `gpt-oss-120b`, follow-up-turn context only:** the
   model dumps raw, unquoted Python directly after `"arguments":` with
   no `"code"` key at all. `gpt-oss-120b` is lower-frequency on this
   failure than `llama-3.3`, not immune — worth continued monitoring.

### 2.4 `run_eda_agent` / `continue_conversation` return contract
Both functions originally returned only `(report, audit_log)`. Fixed to
return `(report, audit_log, messages)`, so a completed conversation's
real state — not a hand-reconstructed approximation of it — can be fed
into `continue_conversation`. This was a functional blocker for testing
Finding #9 under realistic follow-up conditions (§4).

### 2.5 `report_verify.py`'s own bug history
The checker has been iterated on as much as the agent itself:

- **Unicode thousands-separator normalization** — reports use `\u202f`,
  not commas, as a thousands separator; was splitting `2,748`-style
  numbers into bogus fragments.
- **Markdown heading/list-number exclusion** — `### 6. Title` was being
  read as a data claim of "6".
- **Dtype-digit exclusion** — `float64` was being read as containing
  "64".
- **Threshold-reference exclusion** — `cardinality > 10` was being read
  as a claim about "10".
- **Nearest-preceding-mention attribution** (check [3]) — numbers are
  now attributed to whichever column name appeared most recently
  *before* them in reading order, not whichever is closest by raw
  character distance. The distance-based version broke on tightly
  packed enumerations ("Category 63%, Stock 31%,") where a *following*
  column name could be marginally closer than the correct preceding one.
- **Semantic-type tagging** (check [3]) — numbers are tagged by a nearby
  keyword (`missing`, `non-null`, `skew`, `unique`, `mean`, `std`,
  `median`) before comparison; only two numbers sharing the same tag
  *and* the same column can be flagged as contradictory. Without this,
  `missingness_report`'s bundled non-null%/missing% output (two
  different, correctly-non-matching metrics that sum to ~100) was
  flagging **every column, on every run**, as self-contradictory —
  escalating from an edge case to a guaranteed false positive.
  Independently smoke-tested (not just re-asserted): confirmed 0 false
  positives on the exact bundled pattern, and confirmed the fix didn't
  neuter genuine-contradiction detection (still correctly flags a real
  same-tag, same-column conflict).
- **`flag_stale_numeric_recall`'s flat-pool false match** — found on a
  live 12-column run: a stated `10.95%` (an actually-unverified, silently
  hand-computed figure — see Finding #11 below) was marked as
  "genuinely appeared before" purely because `10.95` sits numerically
  close to an entirely unrelated fact (`num_arrests` having `11` unique
  values), under the original flat-pool matching's loose tolerance
  (`abs_tol=0.5`, tuned for large-number rounding drift, not
  percentage-scale precision). Confirmed via direct test before
  fixing. **Fix:** pool entries and stated numbers are now matched as
  `(column, claim-type, value)` triples — same nearest-preceding-column
  and keyword-tag machinery check [3] already uses — so an unrelated
  fact can no longer verify an unrelated claim just by numeric
  proximity. **The first fix attempt itself regressed a previously
  passing test** (attributing an entire line's numbers to whichever
  column name appeared first in it, rather than per-number
  nearest-preceding attribution) — caught only because a small
  regression suite of prior validated cases was re-run after the change,
  not assumed to still pass. A second, smaller bug (a shared claim-type
  keyword stated once for an entire comma-separated list, e.g. "...,
  Discount 9%, ... missing.", sitting outside the per-number tagging
  window) required a further fallback: if a line contains exactly one
  claim-type keyword anywhere, apply it line-wide rather than leaving
  distant numbers untagged. Final state re-validated against: the
  original bug case, both previously-passing stale-recall tests, and
  the real transcript that surfaced the bug — all four correct.
- **Unicode threshold-symbol gap in `THRESHOLD_CONTEXT_RE`** — the
  regex (`[<>]=?\s*$`) only matched ASCII `<`/`>`; the model routinely
  writes `≤`/`≥` instead, so "cardinality ≤ 10" was slipping through as
  a false-positive unverified number on check [1]. Root cause was
  confirmed before fixing (character class literally excluded the
  glyphs in use). **Fix:** character class extended to
  `[<>\u2264\u2265]=?\s*$`. Directly tested against the exact
  `≤ 10` case: no longer flagged; a plain unverified number in the same
  report is still flagged, confirming the fix didn't over-widen the
  exclusion.
- **Check [3] shared-bucket-range false positive** — "Moderate
  missingness (10-20%): victim_age, victim_gender, weapon_used,
  case_status" was flagging `victim_age` as self-contradictory (10 vs.
  20). Root cause: neither number has a *preceding* column mention (all
  four columns come after the range), so the old fallback picked
  whichever column was nearest by raw distance — victim_age, for both
  numbers — fabricating a conflict out of a range describing four
  columns collectively, not two claims about one column. **Fix:**
  column attribution (`_resolve_owner_column`, now shared across checks
  [3] and the stale-recall check) only auto-resolves a number with no
  preceding column when exactly ONE column follows it; if multiple
  columns follow with none preceding, the number is left unattributed
  rather than guessed. Directly tested: the exact 4-column range case no
  longer flags, a genuine same-column same-tag conflict (`victim_age`
  missing 12% vs. 25% in different sentences) still flags, and the
  previously-validated bundled non-null/missing-% pattern still doesn't
  false-positive.
- **Check [1] ratio-jargon false positive** — "an 80/20 train-test
  split" flagged both `80` and `20` as unverified numbers; this is
  standard ML methodology vocabulary, not a claim about the dataset —
  the numeric-token counterpart to the already-known "Recommendations
  section" gap below, which so far only had a carve-out concept for
  check [2]'s text tokens. **Fix:** `JARGON_NUMBER_CONTEXT_RE` matches
  an N/M ratio immediately followed by train/test/validation/split
  wording and excludes numbers inside that span from check [1].
  Deliberately narrow — a superficially similar but genuine data claim
  ("80/300 rows had missing values") is unaffected, confirmed by direct
  test, since the exclusion requires the specific split-idiom wording,
  not just an N/M shape.
- **Check [1] false NEGATIVE — magnitude blind spot let a fabricated
  statistic through completely undetected.** Found on a live 12-column
  run: the final report stated `num_arrests skewness = 0.12`, a number
  no tool call ever computed (step 5 only computed skew for
  `suspect_age`, `victim_age`, `property_loss_usd`) and which was
  independently confirmed WRONG against ground truth (`-0.5988`, opposite
  sign) — a genuine Finding #9-class fabrication, not just an unverified
  claim, in a new concrete form (a skew statistic rather than a
  class-balance count). `verify_report` flagged nothing. Root cause was
  two compounding bugs: (1) `ignore_below=1.0` discarded ANY number under
  1.0 regardless of shape, including small-magnitude statistics exactly
  like this one — fixed by exempting anything with a decimal point,
  since a decimal is never the list-index/rank-marker noise the cutoff
  was meant for; (2) even past that, `_in_pool`'s flat `abs_tol=0.5`
  falsely "verified" `0.12` against a completely unrelated `0` from some
  other column's missing-count, since 0.5 is enormous relative to a
  near-zero statistic — fixed by scaling `abs_tol` down to 0.02 for
  values under magnitude 5. Both fixes tested directly against this
  transcript's numbers (fabricated `0.12` now flags; the report's
  genuinely-verified `-0.8364` property-loss skew still passes) and
  against the full prior regression suite, with nothing broken.
  **This is the more consequential bug of everything fixed this round —
  a false negative in the checker is a direct hole in the project's core
  trust claim, worse than any false positive.**
- **Check [1] false positive on a genuinely-correct negative statistic** —
  found on a follow-up 12-column run where the agent got everything
  right: `num_arrests skew` was actually computed (-0.5990, matching
  ground truth -0.5988), but the report's markdown wrote it as
  `**\u20110.60**` using U+2011 (non-breaking hyphen) as the minus sign,
  not ASCII `-`. `NUMBER_RE_STRICT`'s `-?` only matches ASCII, so the
  sign silently vanished, the claim parsed as *positive* 0.60, stopped
  matching the pool's negative value, and a correct number got flagged
  as fabricated. Same root cause as the `≤/≥` bug and the
  Unicode-thousands-space bug before it — the model uses "nicer"
  unicode punctuation the regexes didn't expect — just landing on a
  minus sign this time.
- **Consolidation: three separate one-glyph-at-a-time fixes (thousands
  space, comparison operator, minus sign) replaced with one deliberately
  broad normalization pass.** Fixing exactly the one glyph that broke,
  each time, is itself the pattern this project's own 3-strikes rule
  says to stop doing. `_normalize_number_spacing` now widens each
  character class to the full common set for its role — all of
  `\u2000`–`\u200a`/`\u202f`/`\u205f`/`\u3000` as thousands-space
  variants, all of `\u2010`–`\u2015`/`\u2212` (+ small/fullwidth forms)
  as minus-sign variants, `\u2264`–`\u2267`/`\u2a7d`/`\u2a7e` as
  comparison-operator variants — rather than only the specific
  characters already observed in a transcript. Verified this isn't
  cosmetic: tested against unicode variants that have NOT yet appeared
  in any transcript (em-dash and true minus-sign as negative signs,
  `≦`/`⩽` as threshold operators, en-space as a thousands separator) —
  all correctly handled — alongside the full existing regression suite,
  which still passes unchanged.

**Known, deliberately deferred limitations of the checker:**
- Check [3] only compares numbers within the same table cell/sentence —
  cannot catch a contradiction split across distant locations in a long
  report. Needs report-wide per-column fact aggregation; out of scope
  for now.
- Check [2] has a persistent false-positive rate on "Recommendations"
  sections — proposed encoding schemes, model names (`StandardScaler`),
  etc. get flagged as fabricated when they're recommendation vocabulary,
  not data claims. Needs section-scoping; not yet attempted. (Check
  [1]'s narrower train-test-split-ratio slice of this same gap is now
  fixed, above — the general section-scoping problem for check [2] is
  not.)
- Occasional formula constants (e.g. `100` in `Price * (1-Discount/100)`)
  get flagged as unverified by check [1]. Minor, low priority.
- Check [1]'s `HEADING_NUMBER_RE` exclusion only matches numbered-list
  markers at the *start of a line* — a numbered list embedded mid-line
  inside a markdown table cell (`| **Next steps** | 1. Finalize... <br>2.
  Convert...`) isn't caught, since it's not a line-starting heading in
  raw text form. Distinct from the "Recommendations vocabulary" gap
  above — same section, different mechanism (list markers vs.
  recommendation nouns).
- **Check [3] positional column/number pairing false positive.** A
  sentence naming two columns together followed by two numbers meant to
  pair positionally — `"Age fields (suspect_age, victim_age) – ~21% and
  ~11% missing"`, first number for the first column, second for the
  second — gets both numbers attributed to whichever column is nearest,
  manufacturing a false contradiction against the real missingness
  table stated earlier in the same report. A cousin of the already-fixed
  range-bucket bug (one range shared by several columns), but a distinct
  mechanism (several columns paired positionally with several numbers,
  not one range applied to all). Logged rather than fixed: found once,
  the underlying data was correct (the false positive is noise in
  review output, not a trust problem with the report itself), and
  deliberately not opening another patch cycle on check [3] at this
  point in the project.
- **Check [3] "columns-before-shared-range" false positive — mirror
  image of the bug above.** `"weapon_used, severity, case_status also
  show non-trivial missingness (≈6-14%)"`: three columns named together,
  then ONE range meant to describe the group loosely, attributed entirely
  to the last-named column (`case_status`), manufacturing a false
  contradiction against that column's real value (13.92%) stated
  elsewhere in the report. The already-fixed range-bucket case was
  "range, then several columns, no preceding column to anchor to." This
  is "several columns, then one range, too MANY preceding columns to
  anchor to unambiguously" — same root fragility (per-column attribution
  by nearest-mention proximity), opposite direction. Explicitly NOT
  fixed, on purpose — see the write-up below on why this and its mirror
  image are treated as one open-ended category rather than two bugs to
  patch.
- **Deeper gap this second case revealed, not yet caught by anything:**
  the model's claim itself was imprecise, not just ambiguously
  attributed — `weapon_used` (actually 18.48% missing) doesn't belong in
  a "≈6-14%" bucket at all; only `severity` (6.76%) and `case_status`
  (13.92%) do. `report_verify.py` checks whether a stated number
  matches something real anywhere in the log — it has no concept of
  whether a number's *grouping/bucketing* is sensible, and the current
  per-column-attribution design can't be extended to catch this without
  becoming a different, much more semantic kind of checker. Logged, not
  attempted.

**On why the attribution bugs above are logged, not patched — the actual
reasoning, not just "we're tired of patching":** both are instances of
one unbounded problem, not two bugs. Regex-based attribution has to
guess which column a number belongs to from surrounding prose, and
English has no upper bound on the ways it can describe "these N columns
have these N values" — every fix only narrows one phrasing out of an
effectively infinite space. This is the same lesson as the earlier
unicode-glyph whack-a-mole (§2.5), generalized from "which characters"
to "which sentence shapes." The structural fix isn't a better regex —
it's not needing to parse prose for this fact at all. See §5g below.

### 2.6 Wide-dataset scaling limitation and fix
Discovered running the agent against a 33-column real-world dataset
(police incident records) for Finding #4 stress-testing — the run **hit
`MAX_ITERATIONS` (15) without ever producing a report.**

Root cause: `missingness_report` originally accepted exactly one column
per call, so the number of tool-call iterations needed scaled 1:1 with
column count. A 33-column dataframe needs at minimum ~33 iterations
just for missingness alone, before any other analysis step — structurally
impossible under a 15-iteration cap. This is a genuinely new limitation,
distinct from every finding above: those were about whether a claim was
*true*; this is about whether the agent can *finish at all* on a wide
dataset, independent of correctness.

A secondary, smaller waste was observed in the same run: the model tried
writing a Python `for` loop over `df.columns` calling `missingness_report`
*from inside* `execute_python`, and failed, because the tool function
isn't exposed inside the sandboxed execution namespace — it burned a full
iteration printing column names only, then fell back to one-by-one tool
calls. Not a bug exactly, but a real inefficiency the fix below also
addresses.

**Fix:** `missingness_report` now accepts a `cols: list[str]` parameter
alongside the original single-column `col` — passing a list returns all
of their reports in one call, removing the 1:1 column-to-iteration
scaling entirely. Backward compatible; single-column calls are unaffected.
Independently smoke-tested: single-column mode, batch mode across three
columns, and the no-argument error path all return correct results.

**Caveat, originally stated before this was tested — now resolved.** This
removes the *ceiling*, but adoption still needed to be confirmed, not
assumed: the system prompt only *nudges* the model to prefer `cols` over
repeated single-column calls, the same category of fix that has failed
elsewhere in this project (Finding #9's soft-phrasing rule,
`tool_choice="required"`, §4.3–4.4). **Confirmed on the 12-column
re-run**: the model used `missingness_report(cols=[...])` for all 12
columns in a single call, unprompted, on its very first tool call of the
run. Finding #10 marked confirmed in §3.

### 2.7 Dataframe mutation leak — ground truth was not actually independent
Discovered on the 12-column police-data run, and arguably the most
consequential bug found in this project: `run_eda_agent`'s internal
namespace held **the caller's actual dataframe object**, not a copy
(`namespace = {"df": df, ...}`). The agent's own tool calls can mutate
`df` in place (e.g. `df['property_loss_usd_num'] = pd.to_numeric(...)`),
and because it's the same object, that mutation persisted on the
caller's own notebook variable after the run finished.

**Concretely, in the actual run that surfaced this:** the agent added a
derived column mid-run. `ground_truth_summary(df_c)`, called *after*
`run_eda_agent(df_c)` with the same variable, then reported the
dataframe's shape as `(5250, 13)` — one column more than the true
original 12-column dataset — because it was computing "independent"
ground truth on a dataframe the agent itself had already altered.

**This time the contamination was benign** — the added column was
computed correctly, so the numbers still matched. But the entire
project's methodology (§5: never trust an agent's self-report, always
diff against an independent source) has a real hole here: the ground
truth stops being independent the moment the agent can mutate the
object it's computed from. Had the agent added a *wrong* derived column,
the "independent" check would have validated against the corrupted
data and caught nothing.

**Fix:** both `run_eda_agent` and `continue_conversation` now build their
execution namespace from `df.copy()`, not `df` directly. Verified
directly: constructed the exact mutating call from the live transcript
against both the old and new namespace construction, confirmed the
caller's dataframe is left untouched only after the fix.

**Re-confirmed on a later run:** ground truth shape stayed at the true
original `(5250, 12)` despite the agent again deriving new columns
mid-run — no contamination, on a run that also produced the Finding #4
multi-step evidence above.

### 2.8 `main.py` and the missing audit-trail capture

Moving off Colab meant the cell-by-cell workflow (`run_eda_agent` →
`ground_truth_summary` → `verify_report` → optional
`continue_conversation`) needed a real driver script instead of manually
run notebook cells. `main.py` was built to do this, and — per direct
user pushback ("why should an analyst read raw checker debug output") —
deliberately splits output by audience into two files:

- `full_run_report.md` — the agent's report plus a short trust banner
  (`build_trust_banner`), nothing else. What an analyst reads.
- `verification_details.md` — ground truth, full raw verifier output,
  and (as of this fix) the raw tool-call trail. Read only if the banner
  flags something, or to validate the harness itself.

**Bug found:** `ground_truth_summary` and `verify_report` were both
wrapped in `capture_stdout` so their printed output gets saved to
`verification_details.md`. `run_eda_agent(df)` — running with its own
`verbose=True` default — was the one call **not** wrapped. Its
step-by-step `[Step N] MODEL CALLED ... / RESULT: ...` trail printed
straight to the terminal and was never written to either file. Not a
deliberate exclusion — the raw trail belongs in `verification_details.md`
by the file's own stated purpose — just a gap introduced when the
Colab-cell workflow (where "printed" and "saved" were effectively the
same thing, since the notebook itself was the artifact) became a
standalone script (where they're not).

**Fix:** `render_audit_log(audit_log)` builds a markdown rendering
directly from the structured `audit_log` list (`step`/`code`/`result`
keys) rather than capturing `run_eda_agent`'s printed stdout — more
robust, and doesn't depend on `verbose` staying `True`. Added as the
first section of `verification_details.md` (chronologically the
earliest artifact of a run). `eda_agent.py`'s own `verbose` printing was
left untouched — still useful for watching a run live in the terminal;
this just stops it from being the *only* copy.

### 2.9 `compute()` syntax auto-repair for bare comprehensions

**Recurring error, confirmed across multiple runs:**
```
compute(expressions=["df[col].nunique() for col in df.columns"])
ERROR: SyntaxError: invalid syntax
```
Root cause: a bare generator expression (`X for Y in Z` with no
enclosing brackets) is valid Python only as the sole argument to a
function call, not as a standalone `eval()` expression. The model
clearly intends "give me this value for every column" and reaches for a
comprehension shorthand on wide dataframes, but drops the required
enclosing `[...]`.

**Fix:** on `SyntaxError`, if the expression contains `" for "` and
doesn't already start with a bracket, `compute()` auto-wraps it in
`[...]` and retries once before giving up, returning
`(auto-repaired unparenthesized comprehension -> [...])` alongside the
result. This is safe because bracket-wrapping only fixes syntax — it
cannot change what gets computed — unlike auto-correcting the actual
logic, which would be a substance change and is deliberately not
something this function does silently. Falls through to a clearer
hint-based error message if auto-repair itself fails.

**Confirmed working in production, not just in isolation:** multiple
later runs hit the exact malformed pattern and the audit log shows the
auto-repair firing and returning correct results each time — including
one case that closed, in the same run, a long-standing gap where
`fnlwgt`'s uniqueness percentage had been stated with no
percentage-producing tool call behind it at all.

### 2.10 Structural completion gates added to `run_eda_agent`

Three related bugs, found and fixed in sequence, all in the same part
of the loop (the `if not message.tool_calls:` branch where the model
decides it's done):

**2.10a — Premature stop with no coverage requirement.** The model was
finalizing a report after as few as 4 tool-call steps, skipping class
balance and/or ID-checking entirely, then backfilling the missing
numbers from memory (a fifth Finding-#9-style fabrication, confirmed
against a real Adult/Census-Income run where the recited numbers were
the well-known published dataset statistics, not something this run's
`df` had actually produced). Rule 7 ("stop when you have enough
evidence") gave no operational floor for what "enough" meant.

**Fix:** `REQUIRED_CHECKS` + `missing_coverage(audit_log)`, a
module-level, crude-but-cheap pattern match on `audit_log`'s logged
`code` strings (does *a* `missingness_report` call exist; does *a*
`nunique(`/`.skew(` call exist; etc.). If gaps exist and iterations
remain, the loop pushes back with an explicit message naming the
missing categories and forces continuation instead of accepting the
report. Rule 7 in the system prompt was reworded to match ("do not stop
until each category is covered — the loop will reject an incomplete
report").

**Implementation bug found in the first version of this fix, before it
ever ran:** the person implementing it defined `REQUIRED_CHECKS` and
`missing_coverage` correctly but never called `missing_coverage` inside
the `if not message.tool_calls:` branch — the original unconditional
`return` was still there, untouched, right below the new dead code. The
gate would have done nothing on a real run had it not been caught before
testing. Fixed by moving the checks to module level (defined once, not
redefined every loop iteration) and actually wiring `gaps =
missing_coverage(audit_log)` into the return-vs-continue decision.

**2.10b — Bundled check let the model satisfy class-balance coverage
without ever checking class balance.** The original
`"class_balance_or_id_check"` accepted *any* `compute()` call as
satisfying it — including the routine cardinality-percentage `compute()`
calls the model already makes for a different section. This meant the
gate reported "all clear" before the model had ever touched `income`,
so the "please verify this" nudge never fired, and the model recited the
same well-known Adult-dataset percentages from memory a second time —
not a defiance of a correction, but a correction that was never sent in
the first place. **Confirmed via two live runs before the fix, one
clean run after.**

**Fix:** split into two independent checks — `cardinality/skew`
(unchanged) and a standalone `class_balance` requiring `"value_counts"`
specifically in the logged code, whether it appears inside an
`execute_python` call or a `compute()` expression string. Gaming one
category can no longer silently satisfy the other. `id_check` doesn't
need a separate flag: its evidence (nunique counts) is already required
by `cardinality/skew`.

**2.10c — Fabricated step citations recurring across multiple runs on
the same claim shape.** A third, independent occurrence (following two
earlier ones logged separately) of the model appending a second,
invented `{{step:N}}` tag to an ID-like-column claim (`{{step:7}}
{{step:8}}` where step 8 never ran). Three recurrences of an identical
pattern against clear prose (rule 10 already explicitly forbids
fabricated citations) is this project's own stated threshold for
switching from prompt wording to a structural gate — the same reasoning
already applied to 2.10a/2.10b.

**Fix:** `invalid_citations(report_text, audit_log)` extracts every
`{{step:N}}` tag from the model's draft final report and checks the
step number against real entries in `audit_log`, before the report is
accepted. Wired into the same finalization branch as the coverage
gate — a draft with either gaps or fabricated citations gets pushed
back with a message naming the specific problem(s), rather than two
separate rejection paths. Deliberately does **not** attempt the more
precise version of this check (does the cited step's output actually
*contain* this number) — that's what `report_verify.py`'s existing
`flag_citation_mismatches` already does, correctly, after the fact;
duplicating that logic mid-loop would be a much larger, riskier change
for a check that already has a working home downstream.

**Important limitation confirmed after these gates shipped, see §3
Finding #16 and §5f below:** the completion gates check whether the
*right category* of tool call happened (a `value_counts` call exists
somewhere in the log) — they have no concept of whether that call's
*output was correctly used* afterward (e.g. whether a returned ratio was
converted to a percentage inside the tool call, per rule 6c, or by hand
outside it). The gate answering "did the investigation happen" and the
gate answering "was every downstream number derived compliantly" are
different questions; only the first exists right now.

### 2.11 `report_verify.py` — bare "step N" prose mentions

**Third confirmed instance of a false-positive contradiction**, and the
first one with a fully identified mechanism rather than a suspected
pattern. A report stated `"fnlwgt ... (≈66.48% unique, step 7) ..."` —
note: plain prose, not the `{{step:7}}` brace convention the system
prompt actually specifies. `strip_citation_tags`'s regex only matched
the braced form, so the bare `7` survived into `flag_internal_
contradictions`, got tagged `nunique` (the word "unique" sits right next
to it), attributed to the nearest preceding column (`fnlwgt`), and
flagged as a second, conflicting uniqueness value against the real
`66.48`. Compounding factor: `flag_unverified_numbers` also missed the
bare `7` entirely, because `7` coincidentally *is* a real value elsewhere
in the log (`marital_status` has 7 uniques) — exactly the
"coincidentally-matching number" limitation this file's own docstring
already warns about, now observed live rather than theoretical.

**Fix:** `BARE_STEP_MENTION_RE` (`\(?\bstep\s*#?\s*\d+(?:\s*,\s*\d+)*\)?`,
case-insensitive) added alongside the existing brace-tag regex; both are
stripped by `strip_citation_tags` before any downstream check sees the
text, treating a bare mention as citation metadata (not a second data
claim) the same way a `{{step:N}}` tag already is. Verified directly:
the exact failing sentence no longer produces a contradiction; a
genuine, unrelated same-column same-tag contradiction (values 8+ apart)
still fires correctly; real `{{step:N}}` tags still strip as before.

**Known residual risk, not yet observed in practice:** the regex will
also strip an unrelated sentence that happens to contain the literal
phrase "step N" outside a citation context (e.g. "the third step 2 hours
into preprocessing"). Considered unlikely in this report's vocabulary
and not worth guarding against preemptively without a real case to test
against.

---

## 3. Findings Log — Agent Behavior

Numbered in discovery order. "Confirmed" means re-tested across multiple
repeated runs of the *same* dataset before being trusted, with
generalization to a different dataset tested only once the fix was
stable — conflating those two tests weakens both.

| # | Finding | Fix type | Status |
|---|---|---|---|
| 1 | Unverified-but-correct claim (Penguins dataset): stated an ID column's uniqueness without a tool call to check it | Prompt rule | Confirmed fixed |
| 2 | True binary (0/1) columns misclassified: `.skew()` applied uniformly, recommending log-transforms inapplicable to Bernoulli columns | Prompt rule (`nunique()` check before skew interpretation) | Confirmed fixed |
| 3 | Low-cardinality non-binary discrete columns (2–10 uniques) inconsistently included/excluded from transform recommendations across runs | Prompt rule (three-tier cardinality classification) | Confirmed fixed across 3 repeated runs |
| 4 | Tool statelessness: `execute_python` recreated its namespace every call | Structural (persistent namespace) | **Confirmed** — 12-column run showed genuine cross-step reuse: a derived column added in one step was directly reused (not recomputed) in two later, separate tool calls. (This run also surfaced §2.7's mutation-leak bug, now fixed; the persistence mechanism itself is validated independently of that bug.) **Strongest evidence to date, from a follow-up run:** the agent hit a real error (`df[cont_cols].skew()` failed on malformed strings like `'35446.9.0'`), diagnosed it via a boolean mask (139 bad rows identified), fixed it with a regex stripping the trailing `.0`, reverified 0 remaining NaNs, and only then recomputed skew successfully — a genuine multi-step investigate → diagnose → fix → reverify loop across separate tool calls, not just variable reuse in the weaker sense the original confirmation showed. |
| 5 | Skewness formula mismatch vs. `ground_truth_summary` | N/A — not a bug | Documentation note only |
| 6 | Silent hand-computed derived statistics: non-null counts for non-numeric columns computed via mental subtraction instead of a tool call, recurring in new concrete forms after repeated prompt patches | **Structural** (`missingness_report` tool added) | Confirmed working across multiple runs since — the clearest evidence in the project that "give it a better tool" beats "tell it not to" once a failure shape recurs 3+ times |
| 7 | Fabricated non-numeric content: invented illustrative category values (`"Electronics"`, `"Clothing"`) never retrieved by any tool, and factually wrong (real values were `A`/`B`/`C`/`D`) | Prompt rule | Confirmed fixed |
| 8 | Self-contradictory restatement: same fact stated with two different values in different report sections | Prompt rule | Confirmed fixed on cases tested — checker's own detection of this pattern required multiple rounds of its own bug-fixing (§2.5) before it could surface real instances without drowning in false positives |
| 9 | Fabricated numeric values (not just unverified — actually wrong): stated Stock class-balance counts (`1,510`/`1,500`) that were factually incorrect (cross-confirmed real values: `1,513`/`1,497`) for a column never re-queried that run | Prompt rule → structural (see §4) | **See §4 — full case study** |
| 10 | `MAX_ITERATIONS` ceiling on wide datasets: a 33-column dataframe hit the 15-iteration cap without ever producing a report, since `missingness_report` originally allowed one column per call — an availability/completion failure, not a correctness failure | **Structural** (`cols: list[str]` batch mode added to `missingness_report`, see §2.6) | **Confirmed** — re-run on the 12-column slice adopted batch mode unprompted on the very first tool call (a single `missingness_report(cols=[...])` covering all 12 columns) |
| 11 | Correct-but-unverified derived arithmetic on top of already-tool-verified numbers: report stated `≈96% uniqueness` (`5050/5250`) and `139 conversion failures` (`575−436`) — both arithmetically correct, neither itself printed by any tool call | Prompt rule (6c) — see §5f/§3 Finding #16 for reopened status | Same root behavior as Finding #6 (silent mental computation) recurring in a shape `missingness_report` doesn't cover. **Was marked resolved after 4/4 clean runs (§5f) — that status is now known to be premature; see Finding #16.** |
| 12 | Model recites well-known public-dataset statistics from memory (Adult/Census-Income class-balance split, 75.92%/24.08%) rather than computing them from `df` — a more dangerous variant of Finding #9, since it doesn't require looking at the data at all | Prompt rule (extended rule 9) + structural (coverage gate, §2.10a/b) | Confirmed fixed after the class-balance check was decoupled from the general-purpose `compute()` check (§2.10b) |
| 13 | Fabricated `{{step:N}}` citation appended alongside a real one, specifically on ID-like-column claims (`{{step:7}} {{step:8}}`) — 3 confirmed recurrences across separate runs | Structural (`invalid_citations` gate, §2.10c) | Confirmed fixed |
| 14 | `compute()` called with a bare, unbracketed comprehension — `SyntaxError`, recurring across runs, wide-dataframe pattern | Structural (auto-repair, §2.9) | Confirmed fixed in production (not just isolated testing) |
| 15 | Verifier false positive: a count and its own percentage for the same column, or a bare "step N" prose citation, both misread by `flag_internal_contradictions` as two conflicting data values | Structural (checker fix, §2.11 — bare-mention normalization; the count-vs-percentage variant remains a known but lower-priority gap) | Bare-mention variant confirmed fixed; count-vs-percentage variant logged, not yet fixed |
| 16 | **Reopened Finding #11, precise mechanism confirmed:** `compute()` called as `value_counts(normalize=True).to_dict()` — a raw ratio, no `*100` — then the report states the percentage anyway (75.92%/24.08%), hand-multiplied outside any tool call. Confirmed on **two consecutive live runs**, both post-dating the rule-6c fix and both post-dating the class-balance coverage gate (§2.10b). The coverage gate does not catch this: it verifies *a* `value_counts` call happened, not that its output was converted compliantly. The citation gate (§2.10c) also doesn't catch it directly — it flags the resulting `{{step:N}}` mis-citation as a symptom, not the underlying hand-multiplication as the cause. | Prompt rule (6c) — **confirmed NOT holding** | **Open, actively recurring.** §5f's "resolved... treating this as closed" status was wrong; the fix's own "4/4 clean" evidence did not generalize past the batch it was measured on. Not being re-patched at the prompt level — see decision note in §5f. |

**Rule 0 (explicit "branch on anomalies, don't run a fixed checklist"
instruction) — status softened from an earlier draft of this document.**
Multiple runs after the rule was added, and again after few-shot worked
examples were added on top of it, show the model checking
`(df['capital_gain']==0).mean()*100` / the equivalent for `capital_loss`
immediately after computing skew, unprompted beyond the system prompt.
**However:** re-checking transcripts predating the few-shot addition
shows this same branch already happening in several of them — the
baseline rate for this *specific* branch was already high before the
few-shot examples were added, which means the post-few-shot successes
are weak evidence for the few-shot fix specifically; they may just
reflect a branch the model was rarely failing at in the first place. An
earlier version of this document stated Rule 0 was "confirmed NOT
reliably working" — that framing was based on limited evidence and is
now understood to be too strong for this particular branch-case. What's
genuinely unconfirmed either way: whether few-shot examples improve
branching on cases the model *was* reliably missing, since no clearly-
failing-before case has been re-checked post-fix yet. Not resolved,
not confirmed broken — logged as an open measurement gap, see §6 item 1.

---

## 4. Case Study: Finding #9, End to End

This finding is documented separately because it's the one place the
project's methodology (§5) was tested most directly, including a case
where the "fix" itself had a bug, and a case where the intended
structural fix turned out not to be viable at all.

### 4.1 Initial fix and its limits
A prompt rule was added: never restate a specific count/percentage from
memory, even if confident — always re-verify with a fresh tool call.
Tested 3 times via `continue_conversation`, **passed all 3** — but all 3
trials used phrasing that explicitly demanded a new derived comparison
("compare X and Y, note the difference"), which naturally cues tool use.
Rule 9 was correctly logged as **promising but not fully validated**
until tested under phrasing that doesn't cue computation.

### 4.2 The soft-phrasing test
Follow-up prompt: *"Summarize the key missingness findings in one
sentence."* Result: the model made **zero tool calls**, answered
entirely from memory. The five recalled numbers happened to all be
correct.

This is not a pass. The rule under test is about the *act* of
re-verifying, not about landing on a correct value by chance — a
stale-but-accidentally-correct number is the exact case that would
otherwise slip through undetected. Logged as: **soft phrasing does not
trigger re-verification; accuracy was coincidental, not enforced.**

### 4.3 First structural fix attempt: `tool_choice="required"`
Rationale: rather than another prompt clause, force verification at the
API level — the model literally cannot return a text-only answer until
it has called a tool.

**Implementation bug (round 1):** `tool_choice` was gated on loop-step
index (`step == 0`) rather than on whether a tool call had actually
succeeded. When the model failed to comply with `"required"` on step 0
(see §4.4), the failure was caught by the malformed-call salvage path,
which retries — but the retry landed on `step == 1`, where the
constraint had already silently reverted to `"auto"`. The fix held for
exactly one attempt, then collapsed, undetected until the next full
test run.

**Fix to the fix (round 2):** replaced step-index gating with an
explicit `tool_used_this_turn` flag that only clears once a tool call
actually succeeds (excluding malformed/`ERROR`-result calls), with a
capped retry count to prevent infinite forcing against a
non-compliant model.

### 4.4 `tool_choice="required"` abandoned — confirmed non-viable
Live testing showed `gpt-oss-120b`, in a `continue_conversation`
context (long prior history + soft follow-up prompt), **fails to comply
with `tool_choice="required"` 100% of attempts** — it consistently
generates a full prose answer instead of a tool call, which Groq
rejects with `tool_use_failed`. This was not intermittent noncompliance
that more retries would resolve; four consecutive attempts produced
near-identical prose failures. The retry loop itself then burned enough
tokens to trigger a **TPD (tokens-per-day) rate limit** mid-run,
preventing the run from ever reaching a completed state.

**Decision: abandoned entirely**, not tuned further. Re-testing variants
of "make `required` work" against a wall with a 100% failure rate would
repeat the same methodological mistake Finding #6 already taught this
project to avoid (§5) — continuing to patch an approach after its
failure mode has been reproduced, rather than switching approach.

### 4.5 Working fix: post-hoc structural detection
`report_verify.py` gained a fourth check,
`flag_stale_numeric_recall(new_report_text, audit_log, boundary_index)`:
capture `boundary_index = len(audit_log)` immediately before calling
`continue_conversation`; afterward, any number in the new response that
matches a **pre-boundary** audit-log entry but no entry **at or after**
the boundary is flagged as recalled-not-reverified — independent of
whether the recalled value happens to be correct.

Independently validated, not just asserted:
- Against the exact real soft-phrasing transcript from §4.2 (5 numbers,
  0 fresh tool calls): **5/5 correctly flagged.**
- Against a control case with one number freshly re-verified and one
  not: **only the non-reverified number flagged** — confirms the check
  doesn't just flag everything indiscriminately.
- Costs **zero additional API calls** — pure post-hoc analysis of
  existing state.

`continue_conversation` now runs with plain `tool_choice="auto"` (no
forcing); `flag_stale_numeric_recall` is the sole enforcement mechanism
for Rule 9, to be run after every call.

**Current status of Finding #9:** prompt-level rule confirmed to fail
under soft phrasing; API-level forcing confirmed non-viable for this
model/context; deterministic post-hoc detection confirmed working
end-to-end in a live run — including a harder case where the model
restated **rounded** values (`31%`, `4%`) rather than the exact
recalled figures (`30.99%`, `3.99%`), and the checker's number-matching
still correctly resolved them back to the underlying audit_log entries.
Zero false positives, zero false negatives across all live and
synthetic tests to date.

**Open gap, not yet closed:** detection is not the same as correction.
The underlying agent behavior is unchanged — it still answers
soft-phrased follow-ups from memory with zero tool calls, in every
trial run so far. `flag_stale_numeric_recall` must currently be called
manually after every `continue_conversation` invocation; nothing
enforces that it will be. See §6, item 5.

---

## 5. Methodology Notes

- **Never declare a fix confirmed off a single clean run.** This
  project's own history includes fixes that looked solid after one run
  and then failed differently on the next (Finding #6 recurred in at
  least 4 distinct concrete forms before a structural fix actually
  closed it; the `tool_choice="required"` fix looked plausible in design
  and failed 4/4 in practice). **Reconfirmed the hard way this round:**
  the rule-6c ratio-vs-percentage fix was declared "resolved... treating
  this as closed" after 4/4 clean follow-up runs (§5f) — and then
  recurred on the next two runs checked after that. Four clean runs in
  a row was treated as passing this project's own 3-strikes bar for
  calling something durably fixed; in hindsight the batches weren't
  independent enough evidence, and this note itself is being updated to
  say so rather than quietly revising the earlier claim.
- **When the same root-cause behavior resurfaces in a new concrete shape
  after a prompt patch, stop patching prompts.** Make the violation
  structurally impossible instead — a new tool (Finding #6), a stricter
  return contract (§2.4), or a deterministic checker (§4.5) — rather
  than writing another prohibition clause. Every fix in this project
  that has actually held up under adversarial re-testing has been
  structural, not prompt-level. The one confirmed exception to "holds
  up" this round is rule 6c itself (Finding #16) — a prompt-level fix
  that looked structural-adjacent (a specific, mechanical instruction
  rather than an abstract compliance ask) but still didn't hold,
  reinforcing rather than weakening the general rule.
- **Distinguish severity classes explicitly:** "unverified" (no tool
  call backs this number, but it might still be correct) vs.
  "fabricated" (specific content invented with no basis) vs. "wrong" (a
  specific number that's actually incorrect) vs. "stale recall"
  (correct, but arrived at without re-verification, per §4.5) are
  different failure modes. Initially conflated under one finding before
  being split across #1, #7, #9.
- **Don't burn API quota on undirected runs.** Each full run resends
  the entire growing conversation history at every step, so a 10–12
  step run costs far more than 10–12× a single call. Spend runs
  deliberately on the single most valuable untested question; use
  quota-dead periods for checker/harness code work that needs zero API
  calls.
- **A structural fix is only as good as its retry/edge-case logic.**
  The `tool_choice="required"` saga (§4.3) shows a correctly-motivated
  structural fix can still fail from an implementation bug (step-index
  gating) independent of whether the underlying approach is sound —
  and, separately, that even a bug-free implementation of a structural
  fix can turn out to target a constraint the model won't honor at all.
  Both possibilities need to be checked before calling something fixed.
  **Reconfirmed this round on the coverage gate (§2.10a):** the first
  implementation was syntactically clean and semantically well-designed
  but simply never wired into the return path — caught only because it
  was checked before being trusted, not because the design was reviewed
  and looked right.
- **`python3 -m py_compile` proves syntax, not correctness.** An edit
  accidentally dropped the `def flag_internal_contradictions(...)`
  header line while adding an unrelated helper function above it. With
  no `def` line to reset indentation, the entire orphaned function body
  silently became unreachable dead code *inside* the preceding
  function — syntactically valid Python, so `py_compile` passed clean,
  and the break was invisible until the function was actually called
  (`NameError` in the notebook). Every edit needs its actual behavior
  re-tested, not just its syntax — a clean compile is necessary, not
  sufficient. **Same lesson, new instance this round:** the dead
  coverage-gate code (§2.10a) compiled cleanly and would have looked
  identical to a working gate in a diff review; only tracing the actual
  control flow (does `missing_coverage` get called anywhere?) caught it.
- **A crashed reassignment leaves the old variable alive — don't mistake
  stale state for a new result.** A `RateLimitError` raised mid-call on
  `report2, audit_log, messages = continue_conversation(...)` prevented
  the assignment from ever completing, so `report2` silently retained
  its value from the *previous successful run* (a different, smaller
  dataset). Output printed afterward looked like a fresh result but was
  actually leftover state — caught only by noticing the column names in
  the printed text belonged to the wrong dataset. Any check run
  immediately after an exception should be treated as suspect until the
  exception itself is understood, not just "check output for
  plausibility."
- **Isolate confounded findings before drawing conclusions from a run.**
  The 33-column stress-test run surfaced two unrelated things at once —
  the `MAX_ITERATIONS` ceiling (Finding #10) and the intended namespace
  test (Finding #4) — in the same trace. Neither could be cleanly
  attributed without first fixing/isolating the other. Decided to slice
  the dataset down to a 12-column subset for the Finding #4 test
  specifically, rather than debug both limitations from one confounded
  run (see §6, item 1, for the column selection and reasoning).
  **Same category of mistake nearly repeated this round with the
  few-shot branching experiment** — post-fix runs showing correct
  branching were initially read as evidence the few-shot fix worked,
  before checking whether pre-fix runs already showed the same branch
  succeeding, which they did. Confound caught before being written up
  as a confirmed result, not after.
- **RAG was considered and rejected for adding "EDA domain knowledge" to
  the agent.** The domain knowledge in question (skew thresholds,
  cardinality-based classification rules, zero-inflation handling) is
  small, static, and doesn't change per-dataset or over time; it's
  already encoded directly in the system prompt, which is strictly more
  reliable than retrieving it (deterministic presence vs. a retrieval
  step that could miss or misretrieve). No failure mode found in this
  project's entire history was ever caused by the agent lacking domain
  knowledge — every one was *behavioral* (fabricating, hand-deriving,
  mis-citing, not branching). RAG doesn't address any of those. Logged
  as a decision, not left implicit, since the question came up directly
  and deserved a real answer rather than a reflexive no.

---

## 5b. The `{{step:N}}` Citation-Tag Convention

Motivated by two converging problems: (a) every check in `report_verify.py`
has to *guess* whether a report number traces to a real tool call via fuzzy
matching, which is exactly where the recurring glyph/regex bugs (§2.5) keep
originating; (b) Finding #11 (silent derived arithmetic, e.g. "≈96%" from
`5050/5250` never run through a tool) has recurred 4+ times with no
structural fix yet.

**Explicitly NOT another `tool_choice="required"` attempt.** §4.3–4.4
already confirmed forcing tool calls at the API level fails 100% of the
time for this model under soft follow-up conditions and burns tokens into
a TPD rate limit. This convention is a **prompt-level ask only** — system
prompt rule 10 in `eda_agent.py` — the model is never blocked from
answering without a tag. If it ignores the convention under soft phrasing
(plausible, per §4.2's precedent), coverage does not regress: an untagged
number just falls through to the existing whole-log checks, unchanged.

**Mechanism:** when the model states a number sourced from a specific tool
step, it appends `{{step:N}}` (N = the step index shown as `[Step N]` in
its own conversation history). `report_verify.py` gained:
- `strip_citation_tags(text)` — removes tags before the existing checks
  run, so a tag's own digits are never mistaken for a second, bare
  unverified number claim. **Extended this round (§2.11)** to also strip
  bare, un-braced "step N" prose mentions, not just the `{{step:N}}`
  form.
- `flag_citation_mismatches(report, audit_log)` — for each tag, finds the
  nearest preceding number and does an **exact lookup** (not fuzzy pool
  matching) against the specific audit_log entry recorded under that step
  index (matched by the entry's own `"step"` field, not list position —
  `continue_conversation` entries use string steps like `"followup-0"`,
  so position-based indexing would silently misalign). Returns three
  distinct signals (see below), treated as stronger than an ordinary
  unverified number because the model asserted a specific, checkable
  provenance and it was false.

Tested directly (not just asserted): a correctly-cited number passes and
does not double-flag after tag-stripping; a citation to a step index that
never ran is caught as `invalid_step_refs`; a citation to a real step
whose actual output doesn't contain the claimed number is caught as
`miscited_numbers`; an uncited number still falls through to the
pre-existing `flag_unverified_numbers` path unchanged; a full
`verify_report` run against a mix of all three produces the expected
combined count.

**Update after 3 live runs:** citation adoption turned out to be high —
the model tagged nearly every sourced number across all three runs,
against the earlier prediction that soft-phrased prompt conventions
would likely be ignored (§4.2 precedent). But the initial version of
`flag_citation_mismatches` had its own bug, found by these runs: the
model consistently cites the step **immediately after** the one that
actually produced the value (nunique computed at Step 2 cited as
`{{step:3}}`, missingness at Step 1 cited as `{{step:2}}`, etc.) — a
systematic off-by-one, not fabrication. The original two-tier check
(real step / fake step) couldn't tell that apart from genuine
fabrication and inflated issue counts to 17–28 per run, almost entirely
noise. **Fixed:** `flag_citation_mismatches` now returns three tiers —
`invalid_step_refs` (cited step doesn't exist), `miscited_verified_elsewhere`
(cited step is wrong, but the number IS verified under some other step —
an indexing slip, low severity, excluded from the headline count),
and `miscited_unverified` (cited step is wrong AND the number appears
nowhere in the whole log — genuine fabrication, same severity as an
invalid step ref). Verified directly against reconstructed fragments of
the real off-by-one case (correctly downgraded) and a synthetic genuine
fabrication (still caught as serious).

**This round's structural extension (§2.10c):** `invalid_citations` now
also runs *before* a report is even accepted by `run_eda_agent`'s own
loop, not just after the fact in `verify_report` — closing a third
recurrence of a fabricated second `{{step:N}}` tag on ID-column claims
before it can ever reach a saved report at all.

## 5c. The `compute(expression)` Tool

Built in response to Finding #11 recurring a 5th+ time across three live
runs (§5b), all with the exact same shape: `incident_id` "≈96% unique"
stated with no tool call behind the division at all.

**Design decision made explicitly:** arbitrary Python expression via
`eval`, not a fixed set of pre-named operations (`ratio()`, `percent()`,
etc.). Trades a small amount of safety surface for not limiting the model
to cases anticipated in advance — accepted as reasonable since
`execute_python` already permits full arbitrary code execution in this
same environment, so `compute()`'s `eval` of one expression doesn't
meaningfully widen what's already possible.

**What it is:** third tool alongside `execute_python` and
`missingness_report`. Evaluates a single expression (not a statement —
assignments/imports/loops go through `execute_python` instead) against
the SAME shared `namespace` dict `execute_python` already uses, so it
sees every variable the model has defined so far this run (consistent
with Finding #4's confirmed namespace persistence). Returns
`"<expression> = <result>"` so both the expression and its value are
preserved in `audit_log`. Non-scalar results (a Series/DataFrame/array)
get a warning appended rather than being treated as an error, nudging
the model to refine the expression rather than silently returning
something the report shouldn't quote directly. **This round: gained a
syntax auto-repair step for bare comprehensions — see §2.9.**

**Explicitly NOT enforced.** Consistent with the §4.3-4.4 lesson and the
citation-tag convention (§5b): no `tool_choice="required"`, nothing
blocks the model from still doing arithmetic in its head. The system
prompt's rule 6 was generalized (previously missingness-specific) to
name `compute()` for any other derived ratio/percentage/difference,
using the exact `incident_id` uniqueness case as the concrete example.
**This has a direct, confirmed consequence, not just a theoretical one —
see Finding #16: nothing enforces that `compute()`'s output is actually
used correctly downstream either, which is exactly the gap the
ratio-vs-percentage recurrence exploits.**

**Tested before calling it done:** a derived-percentage expression
evaluates correctly; an expression referencing a variable defined by an
earlier `execute_python` call resolves correctly (proving shared-namespace
access works); a statement (not an expression) errors cleanly instead of
crashing; a non-scalar result returns the refine-your-expression warning
instead of dumping a raw Series into the report. End-to-end: a
`compute()`-logged result for the exact `incident_id` uniqueness case,
cited with `{{step:N}}`, passes `flag_citation_mismatches` cleanly — the
whole pipeline (tool → citation → verification) closes the loop on this
specific recurring finding, provided the model uses it *and* uses it
correctly — the second half of that condition is the part now known to
fail (Finding #16).

## 5d. Proactive Request Token-Budget Compaction

A live follow-up-turn transcript produced a **413** ("Request too large"),
distinct from every prior 429: the request alone was estimated at 8649
tokens against an 8000 TPM cap -- already over the entire per-minute
budget before any other usage, unlike a genuine 429 (`Used + Requested`
exceeds the limit, but `Requested` alone doesn't) which is transient and
already handled by `call_with_retry`'s existing sleep-and-retry. No retry
count fixes a single request that's larger than the whole cap.

**Root cause:** `continue_conversation` appends one message to the full
accumulated `messages` history of an already-completed main run before
calling `call_with_retry` again -- the single most likely place to tip
over the cap, since it inherits the entire prior conversation for free.

**Decision: proactive, not reactive.** Check and compact `messages`
toward a token budget *before* every request, rather than catching the
413 after the fact and trimming in the exception handler. Reasons this
was preferred over the reactive alternative:
- A reactive catch would still waste a round-trip on a request already
  guaranteed to fail, then have to decide how much to trim under
  exception-handling time pressure -- exactly the kind of ad hoc
  heuristic that tends to become the next bug.
- `messages` (what's sent to the model) and `audit_log` (what
  `report_verify.py` actually checks) are already separate structures
  in this codebase, confirmed by inspection -- compacting `messages`
  cannot touch `audit_log`, so it cannot make a previously-verified
  number look unverified. The one real side effect (the model can't
  re-read a compacted step's full output, so a `{{step:N}}` citation to
  it becomes a mismatch) is already handled gracefully by the existing
  `miscited_verified_elsewhere` / `miscited_unverified` distinction --
  not a new failure mode, the same machinery absorbs it for free.
- Proactive budget-checking is trivial to unit test deterministically
  (build an oversized fake `messages` list, assert the estimate is under
  budget after compaction); reactive-recovery-from-413 is not (would
  require mocking the API's rejection).

**Implementation:** `_estimate_tokens` (rough ~4-chars-per-token
approximation, not exact tokenization) and `_compact_messages_to_budget`
(replaces the CONTENT of the OLDEST tool-role messages with a short
placeholder, one at a time, until under `REQUEST_TOKEN_BUDGET` -- never
removes a message, since removing a tool-role reply would break the
API's requirement that every `tool_call` has a matching tool response;
never touches the system message or most recent messages) live in
`eda_agent.py`. Called from inside `call_with_retry` itself -- the single
function both `run_eda_agent` and `continue_conversation` already share
-- so both call sites get the fix automatically without duplicating the
logic into `continue_conversation.py` separately.

Tested directly: an artificially oversized `messages` list compacts to
under budget; the system message and most recent tool result are left
untouched; compaction proceeds oldest-first; an already-under-budget list
is a no-op; `call_with_retry` is confirmed (via a mock) to invoke
compaction before every request, not just some.

**Bug found and fixed while doing this:** `continue_conversation.py`
never had a dispatch branch for `compute` -- only `execute_python` and
`missingness_report`, falling through to `"ERROR: unknown tool 'compute'."`
for any `compute()` call attempted during a follow-up turn. The tool
schema and system prompt both advertise `compute` for follow-up-turn
arithmetic too (per §4.5's Finding #9 handling), but the dispatch never
supported it. Added, mirroring `run_eda_agent`'s branch exactly. Verified
by reproducing the dispatch logic directly against `compute()`.

## 5e. `compute()` Batch Mode — Finding #11 vs. Finding #10 Tradeoff

A live run (`MAX_ITERATIONS` hit, no final report produced) showed
`compute()`'s original single-expression-only design pushing directly
against Finding #10: the model called it 4 separate times just to get
`nunique()` for 4 categorical columns, one call per column, when the
same numbers would have cost one step via a loop in `execute_python`.
The single-expression restriction was originally deliberate (keep each
derived number unambiguous), but it meant fixing Finding #11
(fabricated arithmetic) was directly increasing pressure toward Finding
#10 (running out of steps) — a real tradeoff surfaced by data, not
theoretical.

**Decision, and why the alternative was rejected:** add batch mode
(`compute(expressions=[...])`, mirroring `missingness_report`'s
established `col`/`cols` precedent) rather than relying on a prompt
instruction alone to make the model batch its calls under the existing
single-expression tool. A prompt-only fix was considered and rejected:
this project has already directly confirmed soft prompts fail under
comparable pressure (§4.2, §4.3-4.4, 100% noncompliance on "verify
before restating"). The one distinction worth being precise about:
those confirmed failures were all cases where noncompliance produces
**wrong data**; ignoring a "prefer batching" prompt produces **wasted
steps**, which is a lower-stakes, already-logged failure mode (Finding
#10), not a new correctness risk. Still, giving the model actual batch
capability removes the dependency on compliance entirely rather than
gambling on it working this time — the prompt nudge is added as a cheap
assist on top of the real fix, not as the fix itself.

**Implementation:** `compute()` now accepts `expression` (single,
backward-compatible with every existing call site and transcript) or
`expressions` (list, evaluated independently — one bad expression in a
batch errors individually without losing the results of the others).
Both dispatch sites (`eda_agent.py` and `continue_conversation.py`)
and the tool schema were updated together, since these two files must
stay in sync on tool support (the same class of gap as the missing
`compute` dispatch found in §5d).

Tested directly: the exact scenario (4 `nunique()` calls) reproduced
in one batch call; single-expression calls still work unchanged; a
batch with one invalid expression still returns correct results for the
valid ones; missing arguments handled cleanly; end-to-end, a
`{{step:N}}` citation to a multi-result batched `compute()` call still
resolves correctly through `flag_citation_mismatches`.

## 5f. Stopping-Bar Check — Original Result and Reopened Status

A stopping bar was set explicitly before running a batch of test runs
(3 transcripts, checked against: (1) no `MAX_ITERATIONS` cutoff, (2)
zero serious issues, (3) any remaining flags are repeats of
already-logged classes, not new ones). Applying it honestly to that
original batch: **not met**, specifically on (2).

**All three of those runs had a genuine Finding #11 recurrence, in a
precise shape:** `compute("df['incident_id'].nunique() / len(df)")`
returns the raw ratio (0.9619...); the model converts it to "96.19%" /
"96.2%" in the final report by multiplying by 100 in its head, outside
any tool call. One earlier validated transcript showed the model CAN do
this correctly — folding the `*100` directly into the expression
(`compute("... / len(df) * 100")`) — but doesn't do so consistently.

**Fix attempted:** both the `compute()` tool schema and system prompt
rule 6 were found inconsistent with each other — the tool's main
description showed the uniqueness example as a bare ratio (no `*100`),
while the `expression` parameter's own example showed it with `*100`.
Fixed both to agree, and added an explicit new rule (6c) stating the
exact failure mode by name: computing a ratio via `compute()` and then
multiplying by 100 by hand afterward is the SAME violation rule 6
already prohibits, since the multiplication itself never went through
the tool.

**~~Resolved, not a limitation~~ — REOPENED. The original text here read:**
*"Confirmed clean on 4 of 4 returned follow-up runs after the fix — zero
recurrences of the hand-multiplication pattern across two separate
batches... a fix confirmed working 4 times in a row is not something to
keep re-testing; treating this as closed."* **That status was wrong, and
is corrected here rather than silently edited away, per this project's
own methodology note on not hiding a reversed conclusion.** Two further
runs, checked after this document's own local-move update, both show
the identical hand-multiplication pattern: `compute()` called as
`value_counts(normalize=True).to_dict()` with no `*100` anywhere in the
expression, and the report stating 75.92%/24.08% anyway. See Finding
#16 (§3) for the full mechanism, including why the coverage/citation
gates built afterward (§2.10) don't catch this particular failure —
they verify a category of investigation happened, not that its output
was used compliantly.

**Decision on further action, made explicitly rather than defaulting to
"patch again":** not attempting a rule-6c rewording (a "6d"). Per this
project's own 3-strikes framing, rule 6c has now failed on recurrence,
the same threshold that has triggered a structural fix everywhere else
in this project. The reason this one is being logged as an accepted,
currently-open limitation rather than immediately structurally patched:
the only structural fix that actually closes this class of gap is the
one already scoped in §5g (structured output — a `class_balance` field
sourced directly from a tool return that includes the percentage,
removing the "was it converted compliantly" question entirely rather
than checking for it after the fact). Building a narrower one-off
structural check specifically for this single rule (e.g. extending the
coverage gate to also verify `*100` appears inside the cited `compute()`
call) was considered and rejected — it would fix this one instance and
leave the next rule-6c-shaped gap uncaught, the same whack-a-mole this
project already moved away from for the checker's own regex bugs
(§2.5). Left open pending the §5g redesign.

---

## 5g. Future Direction: Structured Output Instead of Free Prose

**Not built. Documented here as a deliberate design direction for a
future phase, not a task in progress.**

Every attribution false positive logged above — the range-bucket bug,
its positional-pairing cousin, its columns-before-range mirror image —
is the same underlying tradeoff coming due repeatedly: this project
chose (§5b) to keep the final report as human-readable free-form
markdown, with a citation-tag convention (`{{step:N}}`) layered on top
as a lighter-weight compromise, rather than requiring the model to
return literal structured data. That was a deliberate choice, not an
oversight — it keeps the report readable without redesigning the output
format — and these recurring false positives are its cost, paid in
small installments across many runs rather than once up front.

The actual fix for the whole category, if it's ever worth doing: the
model returns structured claims directly —
`{"column": "weapon_used", "missing_pct": 18.48, "source_step": 2}` —
instead of a sentence a regex has to parse to recover that same
information. There is no attribution ambiguity to resolve, because the
column and its value are paired by construction, not inferred from
proximity in prose. This is the "Option B" design considered and
deliberately deferred earlier in this project in favor of the
citation-tag compromise; nothing has changed about the tradeoff, it's
just accumulated enough small cost that it's worth naming explicitly as
a real future option rather than leaving it implicit.

**Additional evidence accumulated this round, reinforcing rather than
changing the original assessment:** the bare-"step N" false positive
(§2.11) and — more significantly — the reopened ratio-vs-percentage gap
(Finding #16, §5f) are both instances of the exact same underlying
tradeoff. The ratio-vs-percentage case in particular is the clearest
argument yet for this redesign specifically: a structured `class_balance`
field sourced directly from a tool call that already includes the `*100`
step makes "did the model convert this compliantly" a non-question,
rather than something requiring an ever-growing set of checks to catch
after the fact. This is now the strongest concrete example motivating
§5g of anything found so far.

This would be a genuine redesign (report generation, the verifier, and
likely the prompt all change shape) — not a patch, and not something to
bolt on reactively. If pursued, it belongs as a deliberate, scoped
decision in its own right.

---

## 6. Open Items, Priority Order

1. **Design a real test for the few-shot branching question.** Current
   evidence is confounded (§3, Rule 0 status note; §5 methodology note)
   — post-few-shot runs show correct branching, but so do several
   pre-few-shot runs on the same case. Need to identify a branch the
   model was *reliably* missing before the few-shot addition, then
   re-check that specific case post-fix. Without this, "does few-shot
   help" remains genuinely unanswered, not answered-and-confirmed.
2. **Structured-output redesign (§5g)** — now the strongest-motivated
   item on this list, per Finding #16's direct demonstration of exactly
   the gap this redesign closes. Would change the report-generation
   contract, the verifier, and likely the prompt shape; not a patch.
3. **Finding #16 (ratio-vs-percentage) remains open** — deliberately not
   being patched at the prompt level again (§5f decision note); tracked
   here as a live, accepted-for-now limitation pending item 2, not
   forgotten.
4. **Section-scope check [2] in `report_verify.py`** to eliminate the
   persistent "Recommendations section" false-positive rate (technique
   names, model names). Still open, lower priority than (1)–(3).
5. **Dedicated test coverage for recent new code paths** —
   `invalid_citations`, the split `class_balance` check, `compute()`'s
   auto-repair, `BARE_STEP_MENTION_RE` — currently validated only by
   live-run evidence and manual reproduction, not by
   `tests/test_agent.py` or `tests/test_report_verifier.py` directly.
   43/43 existing tests passing confirms no regression, not that these
   specific paths have dedicated coverage.
6. **Module split (`eda_agent.py` → `config.py`/`tools.py`/
   `prompts.py`/`agent.py`) and CI** — still explicitly non-blocking,
   per the original SE-phase decision. No new forcing function has
   appeared.
7. **Add `qwen/qwen3-32b` as a second model** for comparison — namespace
   persistence (Finding #4) is stable, so this is unblocked and ready to
   schedule whenever prioritized above the items above it.
8. **Decide detect-vs-correct for Finding #9** (§4.5): leave
   `flag_stale_numeric_recall` as a manual post-hoc call, or wire it
   into `continue_conversation` automatically.
9. **`ground_truth_summary`'s blind spot on agent-cleaned columns** —
   now that the §2.7 mutation-leak fix keeps `df` pristine,
   `ground_truth_summary` structurally cannot independently verify any
   column the agent itself converted from object to numeric. Open
   decision: should it attempt its own independent coercion using a
   *different* method than the agent's, to stay a genuine independent
   check?

---

## 7. Related Approaches (Prior Art) and Suggested Next Steps

This project's core problem — can an agent's claims be trusted, and if not,
how do you catch it systematically — is not a novel problem. Worth
naming explicitly where this project sits relative to documented
approaches, rather than treating `report_verify.py` as an ad hoc
invention.

### 7.1 The landscape

1. **Grounding upstream, before generation** — force retrieval/a tool
   call for a fact *before* the model is allowed to state it, rather
   than checking after the fact. This is what `missingness_report`
   does for Finding #6, and what `tool_choice="required"` attempted (and
   failed) to do for Finding #9 (§4.3–4.4). Consensus in the field
   matches this project's own conclusion: upstream grounding reduces
   hallucination sharply *when it can be enforced* — but forcing
   compliance at the API level is exactly the brittle point, and this
   project independently reproduced that brittleness rather than being
   an outlier case.

2. **Post-hoc claim-level verification ("tool receipts")** — decompose
   output into atomic claims, check each against a structured record of
   what tools actually returned. This is `report_verify.py`,
   precisely. The published framing for this pattern models each
   verification case as a
   `(request, tool_outputs, response, ground_truth)` tuple and checks
   whether every claim in the response resolves back to a real tool
   receipt — functionally identical to what checks [1]–[4] do here.

3. **Constrained output schema + citation enforcement** — the sharper
   version of #2: require the model to emit structured output where
   every claim carries an explicit reference (a tool_call_id or
   audit_log index), checked by a direct registry lookup rather than
   free-text parsing. **Not yet attempted in this project — see §5g,
   now the top-priority open item per §6.**

4. **Self-consistency / semantic entropy sampling** — generate the same
   response multiple times, measure disagreement across samples as a
   hallucination signal. Not tried here, and of limited relevance to
   this project specifically: this method exists to substitute for
   ground truth when none is available, and this project has always had
   `ground_truth_summary` as a direct, stronger signal.

5. **Separate verifier/judge model** — a second LLM (or a small
   fine-tuned classifier) whose only job is fact-checking the first
   model's output. Deliberately **not used** in this project —
   `report_verify.py` is zero-LLM by design, since a judge model adds
   a second thing to trust rather than removing trust from the system.

6. **EDA/data-agent-specific benchmarks** — academic benchmarks exist
   for this exact problem class (e.g. multi-step data-analytics agents
   scored on both correctness and hallucination rate), including at
   least one proposing a baseline agent architecture for the same task
   this project is doing by hand. Not yet consulted for methodology
   ideas beyond correctness-checking.

**Assessment:** this project independently arrived at approaches #1 and
#2 — the two most established patterns for this problem — without
starting from the literature. That's worth stating in the methodology
section as validation of the project's instincts, not just luck. The
gap is #3: the current implementation of #2 is regex/keyword-based over
free text (`NUMBER_RE_STRICT`, tag-matching), which is why so much of
this project's effort has gone into fixing the checker's own parsing
bugs (§2.5) — thousands-separator handling, dtype-digit exclusion,
semantic-tag collisions — and, this round, why Finding #16 exists at
all: a structured-receipt version of #2 would not have this failure
class, because there would be no "was the arithmetic done in the tool
call or by hand afterward" question left to ask.

### 7.2 Suggested next steps arising from this comparison

*(Placeholder — to be filled in with specific proposals for each of the
six approaches above.)*

1. TBD
2. TBD
3. TBD
4. TBD
5. TBD
6. TBD

---

## 8. Repository Contents

- `eda_agent.py` — agent loop, tool definitions, `call_with_retry`,
  the coverage/citation completion gates (§2.10)
- `report_verify.py` — `flag_unverified_numbers`,
  `flag_unverified_tokens`, `flag_internal_contradictions`,
  `flag_stale_numeric_recall`, `flag_citation_mismatches`,
  `verify_report` (entry point)
- `continue_conversation.py` — `continue_conversation`
- `main.py` — driver script; produces `full_run_report.md` and
  `verification_details.md` (§2.8)
- `tests/test_agent.py`, `tests/test_report_verifier.py` — 43 tests
  total, all passing as of the most recent check-in (see §6 item 5 for
  what this does and doesn't cover)