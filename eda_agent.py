"""
Automated EDA Agent — Groq version.

Same loop as before (inspect -> think -> call tool -> observe -> repeat ->
report), but using Groq's OpenAI-compatible chat completions API instead
of Gemini. Groq uses plain dict-based `messages` and `tools`, not
Gemini's protos.Content/FunctionCall objects — the tool schema and the
loop's message-passing are both different from the Gemini version.
"""

import os
import io
import json
import re
import contextlib
import traceback
import time
import pandas as pd
import numpy as np
from scipy.stats import skew as scipy_skew
from groq import Groq
pd.np = np  # restores the old pd.np.X shortcut some training data still uses


def extract_code_from_tool_use_error(error_str: str):
    """Groq's tool_use_failed error includes the model's malformed call
    in failed_generation. Two shapes have been observed so far:

    1. A JSON-quoted "code" key exists but the outer structure is broken
       (the original, still-common shape): {"code": "print(1)\\n..."}
    2. No "code" key at all -- the model dumps raw, unquoted Python
       directly after "arguments": with no JSON string wrapping. Seen
       on gpt-oss-120b in a follow-up-turn context, not just the
       llama-3.3 text-formatted-call bug this was originally written for
       -- this model isn't fully immune to malformed tool calls, just
       lower-frequency, and may surface more in longer conversations.

    Tries shape 1 first (JSON-quoted, more reliable when it matches),
    falls back to shape 2 (greedy match to the LAST trailing '"}' in the
    string, not the first '}' encountered -- code that itself contains a
    dict literal has internal '}' characters that a naive first-match
    would stop at too early).
    """
    match = re.search(r'"code":\s*"((?:[^"\\]|\\.)*)"', error_str)
    if match:
        raw = match.group(1)
        try:
            return json.loads('"' + raw + '"')  # unescape \n, \", etc.
        except json.JSONDecodeError:
            pass

    match2 = re.search(r'"arguments":\s*(.*)\s*"\}\s*$', error_str, re.DOTALL)
    if match2:
        return match2.group(1).strip()

    return None

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
client = Groq(api_key=os.environ["GROQ_API_KEY"])

MODEL_NAME = "openai/gpt-oss-120b"  # good tool-calling reliability, 1000 req/day free
MAX_ITERATIONS = 15

# Groq's TPM cap observed in practice: 8000 tokens/minute. A confirmed live
# 413 showed a single request alone estimated at 8649 tokens -- already over
# the ENTIRE per-minute budget before any other usage in that window, so no
# amount of retrying fixes it (unlike a genuine 429, where Used+Requested
# exceeds the limit but Requested alone doesn't -- that case is transient
# and call_with_retry's existing sleep-and-retry already handles it). This
# budget is deliberately conservative (well under 8000) to leave headroom
# for completion tokens and other usage sharing the same window.
REQUEST_TOKEN_BUDGET = 5500


def _estimate_tokens(messages) -> int:
    """Rough token estimate (~4 characters per token for English text and
    JSON structure). Not exact tokenization -- good enough to catch 'this
    request is clearly oversized' before sending, not meant to match
    Groq's own tokenizer precisely."""
    return len(json.dumps(messages)) // 4


def _compact_messages_to_budget(messages, budget=REQUEST_TOKEN_BUDGET):
    """If `messages` would produce a request over budget, replace the
    CONTENT of the OLDEST tool-result messages with a short placeholder,
    one at a time, until the estimate is back under budget.

    Never touches the system message or the most recent messages -- and
    never REMOVES a message, only replaces its content. Removing a
    tool-role message would break the API's requirement that every
    tool_call gets a matching tool-role response, corrupting the
    conversation structure rather than just shrinking it.

    Mutates `messages` in place (dicts are mutable, passed by reference),
    so a compacted message stays compacted for the rest of the session --
    intentional, since the point is to stop repeatedly re-sending the same
    oversized history on every subsequent call, not just this one.

    Deliberately does NOT touch audit_log -- that's a separate structure
    report_verifier.py checks against, untouched by this function, so
    compacting `messages` can never make a previously-verified number look
    unverified. The one real side effect: the model can no longer re-read
    a compacted step's full output later, so a {{step:N}} citation to that
    step will land in report_verifier.py's miscited_verified_elsewhere or
    miscited_unverified bucket -- not a new failure mode, the existing
    citation-check machinery already handles it gracefully.
    """
    if _estimate_tokens(messages) <= budget:
        return messages

    PLACEHOLDER = "[Compacted for length -- original result still recorded in audit_log]"
    for msg in messages:
        if _estimate_tokens(messages) <= budget:
            break
        if msg.get("role") == "tool" and msg.get("content") != PLACEHOLDER:
            msg["content"] = PLACEHOLDER

    return messages


def call_with_retry(messages, tools, max_retries=3, tool_choice="auto"):
    """Groq (like Gemini) can hit transient errors or rate limits. Retry,
    and if the error message contains a suggested wait time, use it instead
    of guessing.

    Compacts `messages` toward REQUEST_TOKEN_BUDGET BEFORE ever sending a
    request -- proactive, not reactive. A request that's already over the
    entire per-minute cap (confirmed live: Requested=8649 > Limit=8000)
    can't be fixed by retrying; the identical oversized payload just fails
    identically every time. Checking budget first means that failure mode
    can't happen at all, rather than needing a second exception-handling
    path to detect and recover from it after the fact.

    tool_choice defaults to "auto" for normal agent steps. Pass "required"
    to force at least one tool call before the model is allowed to return
    a text-only answer -- used by continue_conversation's first follow-up
    step to structurally block Finding #9 (restating a number from memory
    instead of re-verifying), rather than relying on a prompt rule the
    model can silently ignore under soft phrasing.
    """
    _compact_messages_to_budget(messages)
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                temperature=0,
            )
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait = 5
            msg = str(e)
            if "try again in" in msg:
                try:
                    wait = float(msg.split("try again in")[1].split("s")[0].strip())
                except (IndexError, ValueError):
                    pass
            print(f"Retrying after error: {e}")
            time.sleep(wait)


# ---------------------------------------------------------------------------
# THE TOOLS — execute_python is unchanged. missingness_report is new: a
# dedicated tool that returns non-null count AND percentage, non-null and
# missing, all in one verified call, so the model has no reason to derive
# any of these by hand (Finding #6 and its recurrences).
# ---------------------------------------------------------------------------

def execute_python(code: str, namespace: dict) -> str:
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer):
            exec(code, {"__builtins__": __builtins__}, namespace)
        output = buffer.getvalue()
        return output if output.strip() else "(code ran, no printed output — add print() statements)"
    except Exception:
        return f"ERROR:\n{traceback.format_exc()}"


def missingness_report(df: pd.DataFrame, col: str = None, cols: list = None) -> str:
    """Verified missingness summary for one or more columns in a single
    tool call. Pass `col` for one column (backward compatible with the
    original single-column signature) or `cols` for a list -- batch mode
    exists specifically so a wide dataframe doesn't force one tool call
    (and one MAX_ITERATIONS step) per column. A 33-column dataframe
    previously hit MAX_ITERATIONS before finishing one-by-one column
    reports; batching removes that 1:1 scaling entirely.
    """
    target_cols = cols if cols else ([col] if col is not None else [])
    if not target_cols:
        return "ERROR: no column(s) specified -- pass 'col' or 'cols'."

    n = len(df)
    lines = []
    for c in target_cols:
        if c not in df.columns:
            lines.append(f"ERROR: column '{c}' not found in df. Available columns: {list(df.columns)}")
            continue
        missing = df[c].isnull().sum()
        non_null = n - missing
        pct_missing = round(missing / n * 100, 2)
        pct_non_null = round(non_null / n * 100, 2)
        lines.append(f"{c}: {non_null} non-null ({pct_non_null}%), "
                     f"{missing} missing ({pct_missing}%)")
    return "\n".join(lines)


def compute(expression: str = None, namespace: dict = None, expressions: list = None) -> str:
    """Evaluate one or more Python EXPRESSIONS (not statements --
    assignments, imports, loops, etc. are not valid here; use
    execute_python for those) against the same shared namespace
    execute_python uses, so it sees every variable the model has already
    defined this run.

    Exists specifically for Finding #11: derived ratios/percentages
    (e.g. "incident_id is 5050/5250 unique") were consistently computed
    by the model in its head and stated in the final report with no tool
    call behind them at all -- unlike missingness_report's non-null/
    missing figures, there was no tool the model could even reach for.
    This is that tool, generalized: any single arithmetic expression, not
    a fixed set of pre-named operations, so the model isn't limited to
    cases anticipated in advance.

    Deliberately NOT enforced via tool_choice="required" or any other
    blocking mechanism -- confirmed non-viable for this model under soft
    follow-up conditions (README §4.3-4.4: 100% noncompliance, tripped a
    TPD rate limit). This tool's presence is a cheap, obviously-correct
    option the model can reach for; whether it actually does is an
    empirical question to check against real transcripts, not something
    to assume by adding the tool.

    `expressions` (a list) exists for the same reason missingness_report
    has `cols=[...]` alongside `col`: a live transcript hit
    MAX_ITERATIONS after the model called compute() four separate times,
    one nunique() per categorical column, instead of batching -- the
    single-expression-only design (originally intentional, to keep each
    derived number unambiguous) was pushing directly against Finding #10
    (running out of iterations) to guard against Finding #11 (fabricated
    arithmetic). Batch mode removes that tradeoff: pass `expressions` for
    several derived numbers in one call, or `expression` for a single one
    (backward compatible with every existing call site and transcript).

    Returns one "<expression> = <result>" line per expression (not just
    the bare result) so each expression is preserved in audit_log next to
    its value -- useful for a human skimming the log, and harmless for
    the numeric-extraction checks in report_verifier.py, which only look
    for the number itself regardless of surrounding text.
    """
    target_exprs = expressions if expressions else ([expression] if expression else [])
    if not target_exprs:
        return "ERROR: no expression(s) specified -- pass 'expression' or 'expressions'."

    lines = []
    for expr in target_exprs:
        try:
            result = eval(expr, {"__builtins__": __builtins__}, namespace)
        except SyntaxError:
            # Common failure mode: the model writes a bare comprehension/
            # generator like "df[col].nunique() for col in df.columns"
            # with no enclosing brackets -- valid as an argument to a
            # function call, invalid as a standalone eval() expression.
            # Auto-repair by wrapping in [...] and retrying ONCE. This is
            # safe because bracket-wrapping only fixes syntax, it cannot
            # change what gets computed -- unlike auto-correcting the
            # actual logic, which would be a substance change and is not
            # something this function should ever do silently.
            if " for " in expr and not expr.strip().startswith(("[", "(", "{")):
                repaired = f"[{expr}]"
                try:
                    result = eval(repaired, {"__builtins__": __builtins__}, namespace)
                    lines.append(
                        f"(auto-repaired unparenthesized comprehension -> "
                        f"{repaired})\n{repaired} = {result!r}"
                    )
                    continue
                except Exception:
                    pass
            lines.append(
                f"ERROR evaluating '{expr}':\n{traceback.format_exc()}\n"
                f"HINT: if this was meant to be a list comprehension across "
                f"columns/values, wrap the whole thing in square brackets, "
                f"e.g. \"[df[c].nunique() for c in df.columns]\" -- a bare "
                f"'for ... in ...' is not valid as a standalone expression."
            )
            continue
        except Exception:
            lines.append(f"ERROR evaluating '{expr}':\n{traceback.format_exc()}")
            continue
        if isinstance(result, (pd.Series, pd.DataFrame, np.ndarray)):
            lines.append(f"{expr} = {result!r}\n"
                         f"(NOTE: this is not a single scalar value -- if you "
                         f"intended a single derived number, e.g. a ratio or "
                         f"percentage, refine the expression to reduce it to one.)")
        else:
            lines.append(f"{expr} = {result!r}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# COVERAGE GATE — module-level, defined once, not re-created every loop
# iteration. Deliberately crude pattern-matching on the `code` string logged
# for each audit_log entry, not semantic understanding of what the code
# does -- same philosophy as extract_code_from_tool_use_error's regex
# fallback above: cheap and mechanical, good enough to catch "this category
# of check never happened at all," not meant to verify the check was done
# well. Exists because prompt-only instructions (rule 7 asking the model to
# self-assess "enough evidence", rule 9 asking it not to state unverified
# numbers) have now failed under soft phrasing on this exact behavior more
# than once -- see README Finding #9 and the compute() docstring's note on
# tool_choice="required" being non-viable. This is the structural backstop,
# not a fourth rewording of the same prompt rule.
# ---------------------------------------------------------------------------

REQUIRED_CHECKS = {
    "missingness": lambda log: any("missingness_report" in e["code"] for e in log),
    "cardinality/skew": lambda log: any(
        ".skew(" in e["code"] or "nunique(" in e["code"] for e in log
    ),
    # Deliberately NOT "or compute(" here -- that used to accept ANY
    # compute() call as satisfying this check, including the unrelated
    # nunique-percentage calls already made for section 3. That loophole
    # meant a model that never touched value_counts on the target column
    # could still pass this gate, which is exactly what let two live runs
    # recite class-balance percentages from memory with no nudge to verify
    # them -- the gate reported zero gaps before the model ever ran a real
    # class-balance check. Only "value_counts" (whether it appears inside
    # an execute_python call or inside a compute() expression string --
    # this matches either) actually demonstrates the check happened.
    "class_balance": lambda log: any("value_counts" in e["code"] for e in log),
}


def missing_coverage(audit_log):
    """Return the list of REQUIRED_CHECKS names that have no matching
    audit_log entry yet. Empty list means full coverage."""
    return [name for name, check in REQUIRED_CHECKS.items() if not check(audit_log)]


_STEP_CITATION_RE = re.compile(r"\{\{step:(\d+)\}\}")


def invalid_citations(report_text, audit_log):
    """Return the sorted list of step numbers cited in report_text via
    {{step:N}} that do NOT correspond to any real audit_log entry.

    This is the structural counterpart to rule 10 in the system prompt
    ("only tag a number if you can actually see that step's result
    containing it"). That rule has now failed under soft phrasing on the
    same specific pattern across multiple separate runs (a high-
    cardinality/ID-like-column claim getting a second, invented step
    number appended) -- three confirmed recurrences of an identical
    failure against clear prose is a signal to gate this structurally,
    the same reasoning already applied to the class-balance coverage
    gate above, not to reword rule 10 a fourth time.

    Deliberately checks against the STEP NUMBER existing in audit_log at
    all, not whether the specific number-being-cited actually appears in
    that step's result -- report_verifier.py's existing citation checker
    already does the harder, more precise version of that check after
    the fact. This gate exists only to catch the cruder, cheaper-to-fix
    case: a citation pointing at a step that was never run in the first
    place.
    """
    valid_steps = {e["step"] for e in audit_log}
    cited_steps = {int(n) for n in _STEP_CITATION_RE.findall(report_text)}
    return sorted(cited_steps - valid_steps)


# ---------------------------------------------------------------------------
# TOOL SCHEMA — Groq/OpenAI format: a plain dict, not genai.protos objects.
# ---------------------------------------------------------------------------

tools = [
    {
        "type": "function",
        "function": {
            "name": "execute_python",
            "description": (
                "Execute Python code against the dataframe `df` (already loaded). "
                "The execution environment already contains:\n"
                "- df (pandas DataFrame)\n"
                "- pd (pandas)\n"
                "- np (numpy, if available)\n\n"
                "Always use print() to display results. "
                "Useful for dataframe inspection, EDA, statistics, plotting, "
                "feature engineering, and any custom Python analysis. "
                "For non-null counts and missing percentages specifically, "
                "prefer the missingness_report tool instead of computing "
                "these by hand here."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute."
                    }
                },
                "required": ["code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "missingness_report",
            "description": (
                "Return a verified, formatted missing-value summary: "
                "non-null count, non-null %, missing count, and missing %, "
                "all computed together in one call. Use this instead of "
                "deriving any of these four numbers by hand (e.g. "
                "subtracting a missing count from the total row count) — "
                "that kind of hand-derivation is exactly what this tool "
                "exists to replace. For a SINGLE column, pass 'col'. For "
                "MULTIPLE columns, pass 'cols' as a list and get all of "
                "their reports back in one call — always prefer 'cols' "
                "over repeated single-column calls when you already know "
                "you need several columns (e.g. from df.columns), since "
                "each call costs one step."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "col": {
                        "type": "string",
                        "description": "Name of a single dataframe column."
                    },
                    "cols": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "List of column names to check in one call. "
                            "Prefer this over multiple 'col' calls."
                        )
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compute",
            "description": (
                "Evaluate one or more Python expressions (not statements -- "
                "no assignments, imports, or loops; use execute_python for "
                "those) and return their values. Use this for any derived "
                "ratio, percentage, or difference you're about to state in "
                "the report -- e.g. a column's uniqueness percentage "
                "(df['id'].nunique() / len(df) * 100), a class-balance "
                "split, or any other arithmetic on values you already "
                "have. This exists so that kind of number has a real, "
                "logged, unquestionably-correct tool call behind it "
                "instead of being computed in your head and stated from "
                "memory. It shares the same variables as execute_python "
                "(df, pd, np, and anything you've already defined this "
                "session). If you need several similar derived numbers "
                "(e.g. nunique() for several columns), pass them all at "
                "once via expressions=[...] rather than calling this tool "
                "once per expression -- each call costs one step. "
                "IMPORTANT: if the number will be STATED as a percentage "
                "in your report, put the '* 100' inside the expression "
                "itself -- e.g. write "
                "\"df['id'].nunique() / len(df) * 100\", not just "
                "\"df['id'].nunique() / len(df)\" followed by multiplying "
                "the result by 100 yourself afterward. A ratio the tool "
                "returned is verified; a percentage you derive from that "
                "ratio by hand afterward is not."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A single Python expression to evaluate, e.g. \"df['incident_id'].nunique() / len(df) * 100\"."
                    },
                    "expressions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "List of Python expressions to evaluate in one call. "
                            "Prefer this over multiple single-'expression' calls "
                            "when you already need several derived numbers, e.g. "
                            "[\"df['crime_type'].nunique()\", \"df['district'].nunique()\", "
                            "\"df['weapon_used'].nunique()\"]."
                        )
                    }
                }
            }
        }
    }
]

SYSTEM_PROMPT = """You are an EDA (Exploratory Data Analysis) agent.
You have three tools:
- execute_python(code): runs Python code against a pandas dataframe called
  `df` (also available: `pd`, `np`). Use this for general inspection,
  statistics, and analysis.
- missingness_report(col) or missingness_report(cols=[...]): returns a
  verified non-null count, non-null %, missing count, and missing % for
  one column (col) or several columns at once (cols) in a single call.
  Always use this tool — not hand-derived arithmetic, and not a separate
  execute_python call that recomputes the same thing manually — whenever
  you need any of these four numbers for a column. If you need this for
  several columns (e.g. you already have the full column list from
  df.columns), pass them all at once via cols=[...] rather than calling
  this tool once per column.
- compute(expression) or compute(expressions=[...]): evaluates a single
  Python expression (not a statement) and returns its value, sharing the
  same variables as execute_python. Use this for any derived ratio,
  percentage, or difference you're about to state in the report — e.g.
  what fraction of a column's values are unique, or a class-balance
  split — instead of working it out in your head. missingness_report
  already covers non-null/missing figures specifically; compute() is for
  everything else you'd otherwise be tempted to calculate by hand. If you
  need several similar derived numbers (e.g. nunique() for several
  columns), pass them all at once via expressions=[...] rather than
  calling this tool once per number — each call costs one step, the same
  reason missingness_report supports cols=[...].

Your job:
0. Do not treat items 1-6 below as a fixed sequence to execute
   identically every time. After each tool call, decide whether the
   result warrants a follow-up before moving to the next item: e.g. an
   extreme skew value (>3 or <-3), an unexpected concentration of one
   category, or a count that contradicts an earlier result. If so,
   investigate that specific thing with another tool call before
   continuing. Two runs on different datasets should not produce the
   same tool-call sequence unless the datasets are genuinely
   equivalent in structure.
1. Inspect shape, dtypes, missing values, and basic stats.
2. Before interpreting skewness or applying any transformation advice,
   check df[col].nunique() for each numeric column. Columns with only 2
   unique values are binary/boolean flags, not continuous distributions —
   their skewness reflects class balance, not outliers, and
   transformations like log() do not apply to them. Interpret and report
   on these separately from truly continuous columns.
2b. Numeric columns are not just "binary" or "continuous" — check nunique()
    for every numeric column before making transform or distribution claims.
    Classify each numeric column by cardinality:
      - nunique() == 2: binary/boolean flag. Skewness reflects class
        balance, not shape. Never recommend log-transform.
      - nunique() <= 10 (and not already classified as binary): low-
        cardinality discrete/count column (e.g. number of bedrooms,
        bathrooms, stories, parking spots). Treat as ordinal/count data.
        Do not blanket-apply the same log-transform or skew-correction
        logic you'd use for continuous variables — state explicitly why
        you are including or excluding each such column from transform
        recommendations.
      - nunique() > 10: treat as genuinely continuous, and apply skew/
        transform reasoning normally.
    Do not lump low-cardinality discrete columns in with continuous
    columns like `price` or `area` without justifying it per-column.
3. Check skewness/distribution of numeric columns with meaningful nulls
   before recommending an imputation strategy.
4. Check class balance if there's an obvious target/label column.
5. Flag columns that look like IDs (high cardinality, no predictive value) —
   verify this with df[col].nunique(), do not infer it from row count alone.
6. Do not compute derived numbers by hand while writing the report — this
   includes non-null counts, percentages, ratios, differences, or any other
   arithmetic on values you already retrieved from earlier tool calls. For
   non-null counts and missing percentages specifically, call
   missingness_report(col) rather than deriving these yourself in any form.
   For any OTHER derived number — a uniqueness percentage, a class-balance
   split, or any other ratio/difference — call compute(expression) instead
   of working it out in your head. A number like "incident_id is ~96%
   unique" is exactly this case: it's not a missingness figure, so
   missingness_report doesn't cover it, but it's still arithmetic on data
   you already have, so it should go through compute(), not your own
   mental math.
6c. If the number you're about to state is a PERCENTAGE, the '* 100' must
    be part of the expression you pass to compute(), not something you do
    to the result afterward. Calling
    compute("df['id'].nunique() / len(df)") and then writing "96.19%" in
    your report by multiplying that ratio by 100 yourself is the SAME
    violation as rule 6 — the ratio was verified, but the percentage you
    stated was not, since the multiplication never went through the tool.
    Write compute("df['id'].nunique() / len(df) * 100") instead, so the
    exact number you state is exactly what the tool returned.
6b. This applies with extra force to non-numeric (object/categorical)
    columns specifically. describe() gives you a non-null count "for free"
    for numeric columns, but not for object columns — so for any
    object/categorical column, call missingness_report(col) rather than
    subtracting its missing count from the total row count by hand.
7. Do not stop and write a final report until you have made at least one
   tool call covering EACH of: missing values, numeric cardinality/skew,
   class balance or ID-check (if a target-like or high-cardinality column
   exists). "Enough evidence" means these categories are covered, not
   that you feel satisfied — the loop will reject an incomplete report
   and ask you to continue, so there is no benefit to guessing instead
   of calling the tool.
8. Before finalizing the report, check that every number is stated consistently everywhere it appears —
   a report that states a value correctly in one section and contradicts it in another is a failure this system prompt should catch.
9. Never state a specific count, percentage, or class-balance figure (e.g.
   how many rows have Stock == 'In Stock', or the split between two
   category values) unless you can point to a tool call in THIS
   conversation where you actually computed that exact number. If you
   believe you already know a value from an earlier step or from general
   knowledge of similar datasets, re-verify it with a fresh tool call
   before stating it — do not restate a number from memory, even if you
   are confident it is correct. A specific number that turns out to be
   wrong is a more serious error than an unverified-but-correct one, and
   this rule exists specifically to prevent that. This applies with extra
   force to well-known public datasets you may recognize (e.g. Adult/
   Census Income, Titanic, Iris) — recalling a published statistic about
   a dataset is the same violation as computing one from memory, and is
   more dangerous because it doesn't require looking at df at all.
10. When you state a number in the final report that came directly from
    a tool result (not restated from memory — see rule 9), tag it with
    the step that produced it, immediately after the number, like this:
    "skew = -0.60 {{step:4}}" where 4 is the step number shown before
    that tool's result earlier in this conversation (e.g. "[Step 4]").
    This is a citation, not a formatting decoration — only tag a number
    if you can actually see that step's result containing it. Do not
    tag a number "just in case"; an incorrect or fabricated citation is
    worse than no citation at all. If a number doesn't trace back to a
    single tool step this way (e.g. it's a rounded approximation or your
    own reasoning), leave it untagged rather than guess at a step number.

Do not state specific values, categories, labels, or examples you have not retrieved via a tool call.
If you don't know the actual category names, either query them or say the category values are unknown,
do not invent illustrative examples.

Always use the provided function-calling mechanism to call your tools.
Never write a tool call as text in your response (e.g. never write
something like <function=execute_python{...}>) — only use the structured
tool-calling interface.
"""

# ---------------------------------------------------------------------------
# THE LOOP — Groq/OpenAI style: messages is a growing list of dicts, and
# tool results go back in as role="tool" messages tagged with tool_call_id.
# ---------------------------------------------------------------------------

def run_eda_agent(df: pd.DataFrame, verbose: bool = True):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Here is a dataframe `df` with shape {df.shape} and columns: "
                f"{list(df.columns)}. Begin your investigation."
            ),
        },
    ]

    audit_log = []

    # .copy() is deliberate: without it, `df` here is the exact same object
    # as the caller's variable (e.g. notebook's df_c). Any in-place mutation
    # the agent makes (e.g. df['new_col'] = ...) would leak back and
    # contaminate any ground_truth_summary(df_c) call made afterward with
    # the SAME variable -- silently breaking the independence the entire
    # verification methodology depends on. Confirmed live: an agent-added
    # column appeared in the caller's own "independent" ground truth.
    namespace = {"df": df.copy(), "pd": pd, "np": np}

    for step in range(MAX_ITERATIONS):
        try:
            response = call_with_retry(messages, tools)
        except Exception as e:
            # Read the actual parsed error body, not str(e) — str(e) runs the
            # body through repr(), which escapes backslashes a SECOND time
            # and silently breaks the unescaping below (this was a real bug
            # in an earlier version of this function).
            body = getattr(e, "body", None)
            if body is None:
                try:
                    body = e.response.json()
                except Exception:
                    body = None

            error_code = (body or {}).get("error", {}).get("code", "")
            failed_generation = (body or {}).get("error", {}).get("failed_generation", "")

            if error_code == "tool_use_failed" or "tool_use_failed" in str(e):
                salvaged_code = extract_code_from_tool_use_error(failed_generation or str(e))
                if salvaged_code:
                    if verbose:
                        print(f"\n[Step {step}] Malformed tool call salvaged from error text:\n{salvaged_code}")
                    result = execute_python(salvaged_code, namespace)
                    audit_log.append({"step": step, "code": salvaged_code, "result": result})
                    if verbose:
                        print(f"[Step {step}] RESULT:\n{result}")
                    messages.append({
                        "role": "user",
                        "content": (
                            f"(Your previous tool call was malformed, but I salvaged "
                            f"and ran this code anyway: {salvaged_code}\nResult:\n{result}\n"
                            f"Continue your investigation, using the proper function-calling "
                            f"interface, not text-formatted tool calls.)"
                        ),
                    })
                    continue
                else:
                    if verbose:
                        print(f"\n[Step {step}] Malformed tool call, could not salvage code. Nudging model.")
                    messages.append({
                        "role": "user",
                        "content": (
                            "Your last tool call was malformed and could not be parsed. "
                            "Please call execute_python again using the proper function-calling "
                            "interface, not a text-formatted call."
                        ),
                    })
                    continue
            raise

        choice = response.choices[0]
        message = choice.message

        # Add the assistant's message (which may include tool_calls) to history
        messages.append(message.model_dump(exclude_none=True))

        if not message.tool_calls:
            # Model wants to stop. Before accepting this as the final report,
            # run TWO structural checks -- neither relies on prompt wording
            # alone, since both categories of prompt-only rule (rule 7's
            # "enough evidence" and rule 10's citation honesty) have now
            # failed under soft phrasing on repeat, confirmed occasions.
            gaps = missing_coverage(audit_log)
            bad_citations = invalid_citations(message.content or "", audit_log)

            if (gaps or bad_citations) and step < MAX_ITERATIONS - 1:
                problems = []
                if gaps:
                    problems.append(
                        f"you have not yet made a tool call covering: {', '.join(gaps)}"
                    )
                if bad_citations:
                    problems.append(
                        f"your draft cites step(s) {bad_citations} with {{{{step:N}}}} "
                        f"tags, but no tool call in this conversation has that step "
                        f"number -- that citation is fabricated and must be removed "
                        f"or corrected to the real step that produced the number"
                    )
                if verbose:
                    print(f"\n[Step {step}] Model tried to finalize with problems: {problems}. Forcing continuation.")
                messages.append({
                    "role": "user",
                    "content": (
                        "Before finalizing: " + "; and ".join(problems) + ". "
                        "Do not state any figures for uncovered categories unless "
                        "you verify them now with a tool call -- this includes "
                        "figures you may recognize from a well-known public "
                        "dataset, which must still be freshly verified against "
                        "THIS dataframe, not recalled. Fix the fabricated "
                        "citation(s) if any, and either continue your "
                        "investigation to cover missing categories or write the "
                        "report omitting unverified sections."
                    ),
                })
                continue

            if verbose:
                print(f"\n[Step {step}] Model produced final report (no more tool calls).")
            return message.content, audit_log, messages

        # Groq/OpenAI can return multiple tool_calls per turn — handle all of
        # them, and DISPATCH BY NAME. Previously this block always assumed
        # execute_python regardless of which tool was actually called, which
        # silently broke missingness_report entirely (it would always parse
        # a 'col' argument as a missing 'code' argument and return a bogus
        # "malformed tool call" error back to the model).
        for tool_call in message.tool_calls:
            fn_name = tool_call.function.name
            try:
                args = json.loads(tool_call.function.arguments)
            except (json.JSONDecodeError, AttributeError):
                args = {}

            if fn_name == "execute_python":
                code = args.get("code")
                if code is None:
                    result = "ERROR: malformed tool call, no valid 'code' argument received."
                    logged_code = "<malformed execute_python call>"
                else:
                    result = execute_python(code, namespace)
                    logged_code = code

            elif fn_name == "missingness_report":
                col = args.get("col")
                cols = args.get("cols")
                if col is None and not cols:
                    result = "ERROR: malformed tool call, no valid 'col' or 'cols' argument received."
                    logged_code = "<malformed missingness_report call>"
                else:
                    result = missingness_report(df, col=col, cols=cols)
                    logged_code = (f"missingness_report(cols={cols!r})" if cols
                                   else f"missingness_report(col={col!r})")

            elif fn_name == "compute":
                expression = args.get("expression")
                expressions = args.get("expressions")
                if not expression and not expressions:
                    result = "ERROR: malformed tool call, no valid 'expression' or 'expressions' argument received."
                    logged_code = "<malformed compute call>"
                else:
                    result = compute(expression=expression, namespace=namespace, expressions=expressions)
                    logged_code = (f"compute(expressions={expressions!r})" if expressions
                                   else f"compute({expression!r})")

            else:
                result = f"ERROR: unknown tool '{fn_name}'."
                logged_code = f"<unknown tool: {fn_name}>"

            # Log EVERY tool call's real code/result, regardless of which
            # tool it was -- report_verifier.py's checks only trust numbers
            # that appear here, so a missingness_report call that never
            # reaches audit_log would make the checker falsely flag its
            # (actually verified) numbers as unverified.
            audit_log.append({"step": step, "code": logged_code, "result": result})

            if verbose:
                print(f"\n[Step {step}] MODEL CALLED {fn_name}:\n{logged_code}")
                print(f"[Step {step}] RESULT:\n{result}")

            # Tool results go back as role="tool", tagged to the specific call
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })

    return "Hit MAX_ITERATIONS without a final report — inspect audit_log.", audit_log, messages


# ---------------------------------------------------------------------------
# GROUND TRUTH — unchanged from before, model-agnostic.
# ---------------------------------------------------------------------------

def ground_truth_summary(df: pd.DataFrame) -> None:
    print("=" * 60)
    print("GROUND TRUTH (computed independently, not by the agent)")
    print("=" * 60)

    print(f"\nShape: {df.shape}")

    print("\nNull counts:")
    print(df.isnull().sum())

    print("\nUnique value counts per column (flag anything close to row count as a likely ID):")
    for col in df.columns:
        n_unique = df[col].nunique()
        print(f"  {col}: {n_unique} unique / {len(df)} rows"
              + ("  <-- likely ID" if n_unique == len(df) else ""))

    print("\nSkewness (numeric columns only):")
    for col in df.select_dtypes(include="number").columns:
        print(f"  {col}: {scipy_skew(df[col].dropna()):.4f}")

    print("\nCompare these numbers by hand against the agent's audit_log entries "
          "and final report. A mismatch, or a claim with no matching tool call "
          "in audit_log, is exactly what you're here to catch.")


# ---------------------------------------------------------------------------
# USAGE (in a notebook cell, after df is loaded):
#
# report, audit_log = run_eda_agent(df)
# print(report)
# ground_truth_summary(df)
# ---------------------------------------------------------------------------