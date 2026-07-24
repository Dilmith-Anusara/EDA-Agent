"""
Automated verification for the EDA agent's final report.

Purpose: replace manual "read the report, cross-check the audit_log by eye"
with a deterministic, zero-LLM-call pass that flags:

  1. Numbers stated in the report with no matching value anywhere in
     audit_log's actual tool outputs (covers Finding #1's unverified-claim
     pattern and Finding #6's hand-computed-arithmetic pattern).
  2. Category/label-like tokens (quoted strings, capitalized words near
     column names) stated in the report with no matching token in
     audit_log's tool outputs (covers Finding #7's fabricated-content
     pattern).
  3. The same column mentioned with two different numeric values in
     different parts of the report (covers Finding #8's self-contradiction
     pattern).

This does NOT judge whether the agent's reasoning/recommendations are
sound (imputation choice, etc.) — that's still a human judgment call per
your own methodology note. This only catches "is every stated fact
actually backed by something the agent really ran."

Known limitations (read before trusting a clean bill of health):
  - Number matching uses rounding tolerance, not semantic understanding.
    A coincidentally-matching number that was never actually verified for
    THAT claim could slip through undetected. This is a safety net, not a
    proof of correctness.
  - The consistency check (#3) is heuristic: it looks for the nearest
    column name before a percentage/number, which can misattribute in
    unusual report structures. Spot-check anything it does NOT flag on
    reports with dense cross-references, at least until you trust it.
  - Category/token matching (#2) will have false positives on generic
    English words. Tune STOPWORDS / add domain terms as needed.
"""

import re
from difflib import SequenceMatcher


# ---------------------------------------------------------------------------
# STEP 1: pull every number that actually appeared in a real tool call
# ---------------------------------------------------------------------------

NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")

# Numbers immediately preceded by a letter (float64, int64) are dtype names,
# not data claims -- exclude via lookbehind.
NUMBER_RE_STRICT = re.compile(r"(?<![A-Za-z])-?\d[\d,]*\.?\d*")

# Standard ML ratio-jargon like "80/20 train-test split" or "70/30 split"
# is methodology vocabulary, not a claim about THIS dataset -- it's the
# numeric-token counterpart to the "Recommendations section" false-positive
# gap already known for check [2]'s text tokens. Matches an N/M ratio
# immediately followed by train/test/validation/split wording, so genuine
# data numbers that merely happen to look like a fraction (e.g. "80/300
# rows") are left alone -- only the specific train-test-split idiom is
# excluded.
JARGON_NUMBER_CONTEXT_RE = re.compile(
    r"\b\d{1,3}\s*/\s*\d{1,3}\b"
    r"(?:\s*(?:train[\s\-]*test|train|test|validation|holdout))?"
    r"\s*split\b",
    re.IGNORECASE,
)


def _jargon_spans(line: str):
    """Character spans in `line` covered by recognized ML/stats jargon
    idioms -- numbers inside these spans are excluded from
    flag_unverified_numbers rather than treated as unverified data claims."""
    return [m.span() for m in JARGON_NUMBER_CONTEXT_RE.finditer(line)]

# --- Consolidated unicode-glyph normalization -----------------------------
#
# Three separate bugs so far were the same root cause wearing different
# clothes: the model writes "nicer" unicode punctuation (typographic
# spaces, dashes, comparison operators) where these regexes expected
# plain ASCII, and each was fixed by adding exactly the one character that
# broke (U+202F thousands space, then U+2264/U+2265 thresholds, then
# U+2011 minus sign). That's the same failure recurring three times, which
# is this project's own stated bar for "stop patching one instance, fix
# the class." These three character classes are deliberately widened past
# only the specific glyphs already observed, to the full common set for
# each role, so the next merely-plausible variant doesn't need its own
# incident to get added.

# Unicode whitespace characters plausible as a thousands-grouping
# separator: NBSP, the U+2000-U+200A space family (en quad through hair
# space), narrow no-break space, medium mathematical space, ideographic
# space, plus the already-confirmed figure space and thin space.
UNICODE_THOUSANDS_SPACE_CHARS = (
    "\u00a0\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009"
    "\u200a\u202f\u205f\u3000"
)
THOUSANDS_SPACE_RE = re.compile(
    r"(?<=\d)[" + UNICODE_THOUSANDS_SPACE_CHARS + r"](?=\d{3}\b)"
)

# Unicode dash/minus glyphs plausible as a negative sign in place of ASCII
# hyphen-minus: hyphen, non-breaking hyphen, figure dash, en dash, em dash,
# horizontal bar, minus sign, and their small/fullwidth presentation forms.
# Only converted when NOT preceded by a digit -- a dash directly after a
# digit is a range separator ("10-20%"), not a sign; that ambiguity
# predates this fix and is deliberately left alone.
UNICODE_DASH_CHARS = "\u2010\u2011\u2012\u2013\u2014\u2015\u2212\ufe58\ufe63\uff0d"
UNICODE_MINUS_RE = re.compile(
    r"(?<!\d)[" + UNICODE_DASH_CHARS + r"](?=\d)"
)

# Unicode comparison-operator glyphs plausible in a threshold reference
# ("cardinality \u2264 10"): ASCII already handled by THRESHOLD_CONTEXT_RE
# itself; this adds \u2264/\u2265 (already-confirmed) plus their less-common
# stroke-through variants \u2266/\u2267 and double-line variants \u2a7d/\u2a7e.
UNICODE_COMPARISON_CHARS = "\u2264\u2265\u2266\u2267\u2a7d\u2a7e"

# Markdown headers ("### 6. Next steps") and ordered-list markers at the
# start of a line ("6. Do X") put a bare number right before a period --
# these are structural, not data claims. Strip them before parsing a line.
HEADING_NUMBER_RE = re.compile(r"^(#+\s*|)\d+[.\)]\s")

# A number preceded (ignoring whitespace) by a comparison operator is a
# threshold reference from the classification rules ("cardinality > 10"),
# not a claim about this specific dataset -- exclude those too. Covers
# ASCII </>/<=/>= plus the unicode comparison glyphs above.
THRESHOLD_CONTEXT_RE = re.compile(
    r"[<>" + UNICODE_COMPARISON_CHARS + r"]=?\s*$"
)

# Bare, un-braced "step N" / "Step N" / "(step 7)" mentions in prose --
# the model's system prompt specifies the {{step:N}} tag convention, but
# it doesn't reliably hold to that exact syntax; it sometimes writes the
# citation as plain prose instead ("...66.48% unique, step 7)"). Confirmed
# on a live run: strip_citation_tags' brace-only regex left the bare "7"
# in "step 7" untouched, so it survived into flag_internal_contradictions
# as ordinary text, got tagged "nunique" (the word "unique" sits right
# next to it), and got attributed to the nearest preceding column
# (fnlwgt) -- fabricating a false fnlwgt=66.48-vs-7 contradiction. Worse,
# flag_unverified_numbers didn't catch the bare 7 either, because 7
# coincidentally IS a real verified value elsewhere (marital_status has 7
# uniques) -- exactly the "coincidentally-matching number" limitation this
# module's own docstring already warns about. Recognizing bare step-
# mentions as citation metadata (same status as a {{step:N}} tag, not a
# second data claim) closes both gaps in one place, since both checks
# already consume the same stripped text.
BARE_STEP_MENTION_RE = re.compile(r"\(?\bstep\s*#?\s*\d+(?:\s*,\s*\d+)*\)?", re.IGNORECASE)


def _normalize_number_spacing(text: str) -> str:
    """Collapse space-grouped thousands (using any recognized Unicode
    space glyph) into a single digit run, and normalize any recognized
    unicode dash/minus glyph acting as a negative sign to ASCII '-', so
    downstream regexes see one correctly-signed number, not a magnitude
    stripped of its sign or split into fragments. Both normalizations
    funnel through this single function, which every check already calls,
    so a widened character class here propagates everywhere at once
    instead of needing to be re-applied per check."""
    text = UNICODE_MINUS_RE.sub("-", text)
    prev = None
    while prev != text:
        prev = text
        text = THOUSANDS_SPACE_RE.sub("", text)
    return text


def _parse_numbers(text: str):
    """Extract numeric tokens from a string, normalizing thousands-space
    grouping and excluding digits glued to letters (dtype names like
    float64/int64). Used everywhere numbers are pulled from text, so this
    one fix propagates to the verified-number pool AND the contradiction
    checker -- previously only flag_unverified_numbers had this exclusion,
    which let dtype-digit noise leak into check [3].
    """
    text = _normalize_number_spacing(text)
    out = []
    for match in NUMBER_RE_STRICT.findall(text):
        cleaned = match.replace(",", "")
        try:
            out.append(float(cleaned))
        except ValueError:
            continue
    return out


def build_verified_number_pool(audit_log):
    """Collect every number that appeared in any tool call's code OR result.

    Numbers in `code` count too (e.g. a literal threshold the model wrote,
    like `df[df.x > 10]`) since those aren't "measured" but also aren't
    fabricated — excluding them would create false positives. The real
    target is numbers in the REPORT with no origin anywhere in the log.
    """
    pool = set()
    for entry in audit_log:
        for field in ("code", "result"):
            text = entry.get(field, "") or ""
            for num in _parse_numbers(text):
                pool.add(round(num, 4))
    return pool


def _in_pool(num, pool, rel_tol=0.005, abs_tol=0.5, small_abs_tol=0.02,
             small_threshold=5.0):
    """Fuzzy membership: exact rounding differences and reasonable
    round-off (e.g. 63.00 vs 62.998624) should count as verified.

    abs_tol=0.5 is deliberately tuned for large-number rounding drift, not
    small-magnitude values -- applying it uniformly meant a claim like a
    fabricated `0.12` skew value would falsely "match" almost any small
    pool number (including a completely unrelated `0` from some other
    column's missing-count), since 0.5 is enormous relative to numbers
    near zero. For |num| below `small_threshold`, use the much tighter
    `small_abs_tol` instead -- confirmed via direct test that this closes
    the false-verification gap without breaking genuine small-number
    matches (e.g. a real skew of 0.12 vs an actual tool output of
    0.1198 still matches within small_abs_tol).
    """
    effective_abs_tol = small_abs_tol if abs(num) < small_threshold else abs_tol
    for p in pool:
        if abs(num - p) <= effective_abs_tol:
            return True
        if p != 0 and abs(num - p) / abs(p) <= rel_tol:
            return True
    return False


# ---------------------------------------------------------------------------
# STEP 2: flag report numbers with no match in the verified pool
# ---------------------------------------------------------------------------

def flag_unverified_numbers(report: str, audit_log, ignore_below=1.0):
    """Return a list of (number, surrounding_context) for every number in
    the report that doesn't match anything actually produced by a tool call.

    ignore_below: skip tiny bare INTEGERS (0, 1, 2, ...) that are almost
    always structural (list indices, markdown table decoration, rank
    numbers) rather than claims worth verifying. Deliberately does NOT
    apply to decimals -- a value like `0.12` is never a list index or
    rank marker, and small-magnitude decimals are exactly the shape a
    fabricated statistic (e.g. a skew value near zero) takes. An earlier
    version of this cutoff applied to ALL numbers regardless of decimal
    point, which let a fabricated `0.12` skewness claim (real value:
    -0.5988, never computed by any tool call) pass through completely
    undetected -- confirmed on a live transcript before this fix. Tune
    down if you want stricter checking of small integers too.
    """
    pool = build_verified_number_pool(audit_log)
    flags = []

    # Walk the report line by line so we can attach context to each flag.
    for raw_line in report.splitlines():
        line = _normalize_number_spacing(raw_line)
        stripped = line.strip()
        # skip markdown headers and ordered-list markers -- "### 6. Title"
        # or "6. Do X" put a bare structural number right before a period,
        # which isn't a data claim.
        if HEADING_NUMBER_RE.match(stripped):
            continue
        jargon_spans = _jargon_spans(line)
        for m in NUMBER_RE_STRICT.finditer(line):
            raw = m.group()
            cleaned = raw.replace(",", "")
            try:
                num = float(cleaned)
            except ValueError:
                continue
            if abs(num) < ignore_below and "." not in raw:
                continue
            # skip threshold references like "cardinality > 10"
            if THRESHOLD_CONTEXT_RE.search(line[:m.start()]):
                continue
            # skip ML ratio-split jargon like "80/20 train-test split"
            if any(start <= m.start() < end for start, end in jargon_spans):
                continue
            if not _in_pool(num, pool):
                snippet = raw_line.strip()
                flags.append((raw, snippet))

    return flags


# ---------------------------------------------------------------------------
# STEP 3: flag fabricated non-numeric content (category names, labels)
# ---------------------------------------------------------------------------

STOPWORDS = {
    "the", "a", "an", "this", "that", "these", "those", "is", "are", "was",
    "were", "no", "not", "and", "or", "for", "with", "of", "in", "on", "to",
    "column", "columns", "value", "values", "row", "rows", "unique", "e.g",
    "eg", "none", "unknown", "missing", "category", "categories", "flag",
    "binary", "continuous", "numeric", "object", "float64", "int64", "nan",
}


def _quoted_tokens(text: str):
    """Pull anything in quotes or backticks — the usual place a model
    states a specific observed value ('Electronics', `In Stock`, etc.)."""
    tokens = re.findall(r"['\"`]([A-Za-z][A-Za-z0-9_\- ]{1,30})['\"`]", text)
    return {t.strip() for t in tokens if t.strip().lower() not in STOPWORDS}


EG_PATTERN_RE = re.compile(
    r"(?:e\.?g\.?,?|such as)\s+([^)\n.;]{2,60})", re.IGNORECASE
)


def _illustrative_example_phrases(text: str):
    """Catch invented examples introduced via 'e.g., X' or 'such as X' even
    when X isn't quoted -- this is exactly the shape Finding #7's
    fabrications took ('e.g., product groups', 'e.g., Electronics,
    Clothing'). Returns the raw phrase after e.g./such as for checking."""
    return [m.group(1).strip() for m in EG_PATTERN_RE.finditer(text)]


def flag_unverified_tokens(report: str, audit_log):
    """Return quoted/backticked tokens in the report that never appeared
    in any tool call's printed result. Column names themselves will
    naturally appear (they're in df.columns, which almost every run
    prints) so this mainly catches invented VALUES within columns, e.g.
    invented category labels, invented stock-status strings, etc.
    """
    audit_text = " ".join(
        (e.get("code", "") or "") + " " + (e.get("result", "") or "")
        for e in audit_log
    )
    audit_tokens = _quoted_tokens(audit_text)
    # also grab bare words that appear anywhere in audit output, to catch
    # tokens the model wrote without quotes in the audit log
    audit_words = set(re.findall(r"[A-Za-z][A-Za-z0-9_]{2,20}", audit_text))

    report_tokens = set(_quoted_tokens(report))
    report_tokens.update(_illustrative_example_phrases(report))

    flags = []
    for tok in report_tokens:
        tok_clean = tok.strip().strip(",")
        if not tok_clean or tok_clean.lower() in STOPWORDS:
            continue
        if tok_clean in audit_tokens:
            continue
        # allow partial credit if every substantive word in the phrase
        # appears somewhere in the audit log
        words = [w for w in re.findall(r"[A-Za-z0-9_]+", tok_clean)
                 if w.lower() not in STOPWORDS]
        if words and all(w in audit_words for w in words):
            continue
        flags.append(tok_clean)

    return flags


# ---------------------------------------------------------------------------
# STEP 4: flag the same column stated with contradictory numbers
# ---------------------------------------------------------------------------

def _split_into_units(report: str):
    """Break the report into comparison units that respect structure:
    - a markdown table row (line starting with '|') is split into its
      individual pipe-delimited CELLS, so numbers in one column's cell
      are never compared against a different column's cell on the same row.
    - prose lines are split into SENTENCES, so a window doesn't bleed
      across unrelated sentences either.
    Returns a list of (unit_text, is_table_cell) strings to search within.
    """
    units = []
    for line in report.splitlines():
        stripped = line.strip()
        if stripped.startswith("|"):
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            units.extend(cells)
        else:
            # crude sentence split; good enough for this purpose
            for sent in re.split(r"(?<=[.!?])\s+", stripped):
                if sent:
                    units.append(sent)
    return units


def _resolve_owner_column(col_positions, num_start):
    """Decide which column a number belongs to, given (position, column)
    pairs already found in the same unit (sentence/cell).

    Nearest-PRECEDING mention wins when one exists -- unchanged from the
    original logic. The fallback for "no preceding mention" used to pick
    the nearest column by raw distance regardless of how many columns
    followed. That silently mis-attributed range statements like
    "Moderate missingness (10-20%): victim_age, victim_gender,
    weapon_used, case_status" -- a single bucket describing FOUR columns
    together, not two separate claims about victim_age specifically. Both
    10 and 20 have no preceding column, so both got assigned to
    victim_age (the nearest one), fabricating a false 10-vs-20
    contradiction for a column the sentence never actually singled out.

    Fix: when there's no preceding column, only resolve automatically if
    exactly ONE column follows (unambiguous -- "10% missing in
    victim_age" with nothing else nearby). If MULTIPLE columns follow
    with none preceding, the number is describing a shared bucket, not a
    specific column -- return None rather than guessing which one.
    """
    preceding = [cp for cp in col_positions if cp[0] <= num_start]
    if preceding:
        return preceding[-1][1]
    following = [cp for cp in col_positions if cp[0] > num_start]
    if len(following) == 1:
        return following[0][1]
    return None


TYPE_KEYWORDS = {
    "missing_pct": [r"missing"],
    "non_null_pct": [r"non[\s\u2011\u2010\-]?null"],
    "skew": [r"skew"],
    "nunique": [r"unique"],
    "mean": [r"\bmean\b"],
    "std": [r"\bstd\b", r"standard deviation"],
    "median": [r"\bmedian\b"],
}


def _classify_number_type(context: str, num_start_in_context: int, window=40):
    """Tag a number by the nearest recognized keyword within `window`
    characters on either side. Returns None if no keyword is found nearby
    -- untagged numbers are excluded from contradiction-checking rather
    than guessed at, since a wrong tag is worse than no tag.

    This exists because two DIFFERENT, non-conflicting metrics for the
    same column (e.g. non-null% and missing%, which necessarily differ
    since they sum to ~100) were being flagged as "contradictions" purely
    for being more than 5 points apart and near the same column name.
    missingness_report's own output format ("X non-null (Y%), Z missing
    (W%)") guarantees this fires on every column, every run, unless
    numbers are compared only within their own claim-type.
    """
    lo = max(0, num_start_in_context - window)
    hi = min(len(context), num_start_in_context + window)
    window_text = context[lo:hi]
    best_tag, best_dist = None, None
    for tag, patterns in TYPE_KEYWORDS.items():
        for pat in patterns:
            for m in re.finditer(pat, window_text, re.IGNORECASE):
                dist = abs((lo + m.start()) - num_start_in_context)
                if best_dist is None or dist < best_dist:
                    best_dist, best_tag = dist, tag
    return best_tag


def _classify_with_line_fallback(line, num_start):
    """Same as _classify_number_type, but if the local window finds
    nothing, fall back to checking whether exactly ONE claim-type keyword
    appears anywhere in the whole line. Condensed summary sentences often
    state the keyword once for an entire comma-separated list ("Category
    63%, Rating 47%, ... Discount 9%, ... missing.") -- the numbers late
    in the list can be 40+ characters from the only occurrence of
    "missing" in the sentence. If the line is unambiguous (only one
    keyword type present at all), it's reasonable to apply it line-wide;
    if multiple different types appear, stay silent rather than guess.
    """
    tag = _classify_number_type(line, num_start)
    if tag is not None:
        return tag
    present_tags = set()
    for t, patterns in TYPE_KEYWORDS.items():
        for pat in patterns:
            if re.search(pat, line, re.IGNORECASE):
                present_tags.add(t)
                break
    if len(present_tags) == 1:
        return next(iter(present_tags))
    return None


def flag_internal_contradictions(report: str, column_names):
    """For each column name, look at numbers that are actually NEAR that
    column's mention (nearest-PRECEDING-mention within the same table cell
    or sentence), tag each by its claim-type (missing%, non-null%, skew,
    etc.) via nearby keywords, and flag only when two numbers SHARE a tag
    for the same column but disagree by more than rounding tolerance.

    Untagged numbers (no recognized keyword nearby) are excluded from
    comparison entirely -- this trades some recall (a contradiction with
    no nearby keyword won't be caught) for a much lower false-positive
    rate, which matters more now that bundled multi-metric report formats
    make untyped proximity-only comparison fire on nearly every column.
    """
    units = _split_into_units(report)
    per_column_typed_numbers = {}  # (col, tag) -> [(num, context), ...]

    for unit in units:
        normalized = _normalize_number_spacing(unit)

        col_positions = []
        for col in column_names:
            for m in re.finditer(re.escape(col), normalized, re.IGNORECASE):
                col_positions.append((m.start(), col))
        if not col_positions:
            continue
        col_positions.sort()

        for m in NUMBER_RE_STRICT.finditer(normalized):
            raw = m.group()
            try:
                num = float(raw.replace(",", ""))
            except ValueError:
                continue
            if not (0 < num <= 100):
                continue
            if THRESHOLD_CONTEXT_RE.search(normalized[:m.start()]):
                continue

            owner_col = _resolve_owner_column(col_positions, m.start())
            if owner_col is None:
                continue  # ambiguous shared-bucket statement -- don't guess

            tag = _classify_number_type(normalized, m.start())
            if tag is None:
                continue  # can't classify -- exclude rather than guess

            key = (owner_col, tag)
            per_column_typed_numbers.setdefault(key, []).append((num, unit.strip()))

    flags_by_col = {}
    for (col, tag), found in per_column_typed_numbers.items():
        if len(found) < 2:
            continue
        values = [n for n, _ in found]
        if max(values) - min(values) > 5:
            flags_by_col.setdefault(col, []).extend(found)

    flags = [(col, found) for col, found in flags_by_col.items()]

    return flags


# ---------------------------------------------------------------------------
# STEP 5: flag numbers restated from memory instead of freshly re-verified
# ---------------------------------------------------------------------------
#
# This is the backstop for Finding #9. tool_choice="required" (see
# continue_conversation_snippet.py) forces the model to call SOME tool
# before answering a follow-up, but doesn't force it to call a tool that
# actually re-verifies every number it goes on to state -- a model could
# satisfy the requirement with an unrelated call (e.g. df.shape) and still
# write earlier-turn numbers from memory in its prose. This check catches
# that case: a number is "stale recall" if it matches something already in
# audit_log BEFORE this turn's boundary index, but does NOT match anything
# logged AFTER that boundary (i.e. nothing freshly re-verified it here).
#
# Being correct doesn't exempt a number from this flag -- Rule 9 is about
# the ACT of re-verifying, not just about landing on the right value by
# recall. A number that's stale-but-accidentally-correct is exactly the
# case that slipped through undetected in initial soft-phrasing testing.

_MISSINGNESS_CALL_RE = re.compile(
    r"missingness_report\(\s*(?:cols\s*=\s*\[(?P<cols>[^\]]*)\]"
    r"|col\s*=\s*['\"](?P<col_kw>[^'\"]+)['\"]"
    r"|['\"](?P<col_pos>[^'\"]+)['\"])",
)


def _extract_known_columns(audit_log):
    """Recover the set of real column names the agent has actually queried,
    by parsing missingness_report(...) calls out of audit_log['code'].
    Lets flag_stale_numeric_recall scope matches to (column, tag) pairs
    without requiring callers to pass column_names explicitly -- existing
    notebook calls (`flag_stale_numeric_recall(report2, audit_log, boundary)`)
    keep working unchanged.
    """
    cols = set()
    for entry in audit_log:
        code = entry.get("code", "") or ""
        for m in _MISSINGNESS_CALL_RE.finditer(code):
            if m.group("cols"):
                for piece in m.group("cols").split(","):
                    name = piece.strip().strip("'\"")
                    if name:
                        cols.add(name)
            elif m.group("col_kw"):
                cols.add(m.group("col_kw"))
            elif m.group("col_pos"):
                cols.add(m.group("col_pos"))
    return cols


def _tagged_pool_from_audit_log(audit_log, index_range, known_columns):
    """Build a pool of (column, tag, rounded_value) triples instead of bare
    numbers. Column is attributed per-line (audit_log results are one
    column per line, even in batched missingness_report output); tag comes
    from the same nearby-keyword classifier check [3] already uses. Numbers
    with no resolvable column AND no resolvable tag go in a separate
    'untagged' bucket matched with a much tighter tolerance, since there's
    no context left to scope them by.
    """
    tagged = set()
    untagged = set()
    for i in index_range:
        entry = audit_log[i]
        for field in ("code", "result"):
            text = entry.get(field, "") or ""
            for raw_line in text.splitlines():
                line = _normalize_number_spacing(raw_line)
                col_positions = []
                for c in known_columns:
                    for cm in re.finditer(re.escape(c), line, re.IGNORECASE):
                        col_positions.append((cm.start(), c))
                col_positions.sort()
                for m in NUMBER_RE_STRICT.finditer(line):
                    try:
                        num = round(float(m.group().replace(",", "")), 4)
                    except ValueError:
                        continue
                    tag = _classify_with_line_fallback(line, m.start())
                    owner_col = _resolve_owner_column(col_positions, m.start()) \
                        if col_positions else None
                    if owner_col and tag:
                        tagged.add((owner_col.lower(), tag, num))
                    else:
                        untagged.add(num)
    return tagged, untagged


def _tagged_match(num, owner_col, tag, tagged_pool, untagged_pool,
                   rel_tol=0.005, abs_tol=0.5,
                   untagged_rel_tol=0.0005, untagged_abs_tol=0.02):
    """Match scoped to (column, tag) when both are known -- eliminates
    cross-matches between unrelated facts that merely happen to be
    numerically close (e.g. a 10.95% missingness figure vs an unrelated
    num_arrests unique-count of 11, previously a confirmed false match).
    Falls back to a much tighter tolerance for untagged numbers, since
    there's no context left to scope by and a loose tolerance there is
    exactly what caused the original bug.
    """
    if owner_col and tag:
        key = (owner_col.lower(), tag)
        for (c, t, p) in tagged_pool:
            if (c, t) != key:
                continue
            if abs(num - p) <= abs_tol:
                return True
            if p != 0 and abs(num - p) / abs(p) <= rel_tol:
                return True
        return False
    for p in untagged_pool:
        if abs(num - p) <= untagged_abs_tol:
            return True
        if p != 0 and abs(num - p) / abs(p) <= untagged_rel_tol:
            return True
    return False


def flag_stale_numeric_recall(new_report_text: str, audit_log, boundary_index,
                                ignore_below=1.0):
    """Return a list of (number, context) for every number in
    `new_report_text` (e.g. a follow-up turn's `report2`, NOT the full
    conversation's report) that matches a pre-boundary audit_log entry but
    has no support from any audit_log entry at or after `boundary_index`.

    boundary_index: pass len(audit_log) captured BEFORE calling
    continue_conversation(), so entries [0:boundary_index] are "old"
    (available in the model's context from memory) and entries
    [boundary_index:] are "fresh" (actually re-verified this turn).

    Matching is scoped to (column, claim-type) pairs, not bare numeric
    proximity -- a flat pool of numbers previously let an unrelated fact
    (e.g. num_arrests having 11 unique values) falsely "verify" an
    unrelated stated percentage (10.95% missingness) purely because 11 and
    10.95 are numerically close. Confirmed via direct test before this fix.
    """
    known_columns = _extract_known_columns(audit_log)
    old_tagged, old_untagged = _tagged_pool_from_audit_log(
        audit_log, range(0, boundary_index), known_columns)
    fresh_tagged, fresh_untagged = _tagged_pool_from_audit_log(
        audit_log, range(boundary_index, len(audit_log)), known_columns)

    flags = []
    for raw_line in new_report_text.splitlines():
        line = _normalize_number_spacing(raw_line)
        stripped = line.strip()
        if HEADING_NUMBER_RE.match(stripped):
            continue
        col_positions = []
        for c in known_columns:
            for cm in re.finditer(re.escape(c), line, re.IGNORECASE):
                col_positions.append((cm.start(), c))
        col_positions.sort()

        for m in NUMBER_RE_STRICT.finditer(line):
            raw = m.group()
            cleaned = raw.replace(",", "")
            try:
                num = float(cleaned)
            except ValueError:
                continue
            if abs(num) < ignore_below and "." not in raw:
                continue
            owner_col = _resolve_owner_column(col_positions, m.start()) \
                if col_positions else None
            tag = _classify_with_line_fallback(line, m.start())
            is_old = _tagged_match(num, owner_col, tag, old_tagged, old_untagged)
            is_fresh = _tagged_match(num, owner_col, tag, fresh_tagged, fresh_untagged)
            # Only a concern if it matches something OLD. A number with no
            # match anywhere is already caught by flag_unverified_numbers;
            # this check is specifically about old-but-not-refreshed recall.
            if is_old and not is_fresh:
                flags.append((raw, raw_line.strip()))

    return flags


# ---------------------------------------------------------------------------
# STEP 6: citation-tag verification (the {{step:N}} convention)
# ---------------------------------------------------------------------------
#
# Rationale: every check above has to GUESS whether a report number traces
# back to a real tool call, via fuzzy pool matching -- which is exactly
# where the recurring glyph/regex bugs keep coming from (unicode minus
# signs, thousands-separators, threshold operators, magnitude-blind
# tolerances). tool_choice="required" was tried to force verification at
# generation time and confirmed non-viable for this model under soft
# follow-up conditions (100% noncompliance, triggered a TPD rate limit --
# see the Finding #9 case study). This does NOT repeat that mistake: the
# citation tag is a prompt-level convention, not an enforced constraint --
# the model is never blocked from answering without one. When a tag IS
# present, verification stops being fuzzy: it's an exact lookup against
# the cited step's own logged output, which is strictly stronger than
# whole-log fuzzy matching and immune to the glyph-normalization arms race
# (no need to fuzzy-parse a number's magnitude/sign when you're checking
# it against one specific, already-known step's result). When a tag is
# ABSENT, coverage doesn't regress -- the existing whole-log checks still
# run on tag-stripped text exactly as before.
#
# A CITED-but-wrong claim (real step, doesn't contain this number; or a
# step number that doesn't exist at all) is treated as a STRONGER signal
# than an ordinary unverified number -- the model asserted a specific,
# checkable provenance and that provenance was false, which is a more
# deliberate-looking failure than simply not citing anything.

CITATION_TAG_RE = re.compile(r"\{\{step:\s*(\d+(?:\s*,\s*\d+)*)\s*\}\}")


def strip_citation_tags(text: str) -> str:
    """Remove {{step:N}} tags AND bare 'step N' prose mentions before
    handing text to the existing whole-log checks, so neither form's
    digits are ever mistaken for a bare unverified number (both are
    metadata about a claim's provenance, not a second claim of their own).

    The bare-mention removal was added after a confirmed live failure:
    the model wrote "step 7" as plain prose instead of the {{step:N}}
    convention its own system prompt specifies. The brace-only regex left
    that "7" untouched, and it went on to fabricate a false contradiction
    in flag_internal_contradictions (tagged as a second "nunique" value
    for the nearest preceding column) and slipped past
    flag_unverified_numbers entirely (7 coincidentally matched an
    unrelated column's real unique-count). See BARE_STEP_MENTION_RE's own
    comment for the full trace.
    """
    text = CITATION_TAG_RE.sub("", text)
    text = BARE_STEP_MENTION_RE.sub("", text)
    return text


def _preceding_number_match(line: str, tag_start: int):
    """The citation tag documents the nearest number stated before it on
    the same line -- find that number's regex match object (or None if
    the tag has nothing to attach to, e.g. it got separated from its
    number by unrelated text)."""
    best = None
    for m in NUMBER_RE_STRICT.finditer(line):
        if m.end() <= tag_start:
            best = m
        else:
            break
    return best


def _step_result_pool(audit_log, step_idx):
    """Numbers that actually appeared in the code OR result of the
    audit_log entry recorded under this exact step index (matched by the
    entry's own 'step' field, not list position -- continue_conversation
    entries use string steps like 'followup-0', so position-based
    indexing would silently misalign)."""
    pool = set()
    found_entry = False
    for entry in audit_log:
        if entry.get("step") == step_idx:
            found_entry = True
            for field in ("code", "result"):
                text = entry.get(field, "") or ""
                for num in _parse_numbers(text):
                    pool.add(round(num, 4))
    return pool, found_entry


def _whole_log_pool(audit_log):
    """Numbers from every step's code/result, regardless of which step --
    the same fallback flag_unverified_numbers already uses, reused here
    to tell an indexing slip apart from a genuine fabrication."""
    pool = set()
    for entry in audit_log:
        for field in ("code", "result"):
            text = entry.get(field, "") or ""
            for num in _parse_numbers(text):
                pool.add(round(num, 4))
    return pool


def flag_citation_mismatches(report: str, audit_log):
    """Verify every {{step:N}} tag in the report against the step it
    cites. Returns (invalid_step_refs, miscited_verified_elsewhere,
    miscited_unverified):

      - invalid_step_refs: the tag cites a step index that doesn't exist
        anywhere in audit_log at all (a fabricated citation).
      - miscited_verified_elsewhere: the cited step's own output doesn't
        contain the number, but SOME other step's output does. Confirmed
        on three live transcripts to be the dominant real-world case --
        the model consistently cites the step one after the one that
        actually produced the value (nunique computed at Step 2 cited as
        {{step:3}}, etc.). This is an indexing slip, not a data problem:
        the number itself IS genuinely backed by a real tool call, just
        filed under the wrong step. Low severity.
      - miscited_unverified: the cited step is real and wrong, AND the
        number doesn't appear in any other step's output either. This is
        the actually alarming case -- a specific, checkable provenance
        was asserted and nothing in the whole session backs it. Treat
        this the same severity as invalid_step_refs.

    A number with NO tag is not flagged here at all -- that's what
    flag_unverified_numbers (on tag-stripped text) is still for. This
    check only judges claims the model chose to cite via the {{step:N}}
    convention specifically -- bare "step N" prose mentions are handled
    upstream (see strip_citation_tags / BARE_STEP_MENTION_RE): they're
    removed before flag_unverified_numbers/flag_internal_contradictions
    ever see them, but they're NOT verified against audit_log here either,
    since a bare mention doesn't commit to the same precise, checkable
    provenance claim a real {{step:N}} tag does.

    Threshold references ("per rule >10") are excluded the same way
    flag_unverified_numbers already excludes them -- confirmed missing
    here by a live case: "(per rule >10) {{step:2}}" was flagged as a
    fabricated citation, when the number is boilerplate classification-
    rule text the citation just happened to land near, not a claim about
    a specific value the tag was actually meant to source.
    """
    invalid_step_refs = []
    miscited_verified_elsewhere = []
    miscited_unverified = []

    for raw_line in report.splitlines():
        line = _normalize_number_spacing(raw_line)
        for tag_m in CITATION_TAG_RE.finditer(line):
            step_indices = [int(s.strip()) for s in tag_m.group(1).split(",")]
            num_m = _preceding_number_match(line, tag_m.start())
            if num_m is None:
                continue  # tag with no attached number -- nothing to check
            if THRESHOLD_CONTEXT_RE.search(line[:num_m.start()]):
                continue  # threshold reference, not a data claim
            try:
                num = float(num_m.group().replace(",", ""))
            except ValueError:
                continue

            for step_idx in step_indices:
                pool, found_entry = _step_result_pool(audit_log, step_idx)
                if not found_entry:
                    invalid_step_refs.append(
                        (step_idx, num_m.group(), raw_line.strip()))
                    continue
                if _in_pool(num, pool):
                    continue  # correctly cited
                if _in_pool(num, _whole_log_pool(audit_log)):
                    miscited_verified_elsewhere.append(
                        (step_idx, num_m.group(), raw_line.strip()))
                else:
                    miscited_unverified.append(
                        (step_idx, num_m.group(), raw_line.strip()))

    return invalid_step_refs, miscited_verified_elsewhere, miscited_unverified


# ---------------------------------------------------------------------------
# TOP-LEVEL ENTRY POINT
# ---------------------------------------------------------------------------

def verify_report(report: str, audit_log, df_columns=None):
    """Run all checks and print a summary. Call this right after
    `report, audit_log = run_eda_agent(df)` in your notebook.
    """
    print("=" * 60)
    print("AUTOMATED REPORT VERIFICATION")
    print("=" * 60)

    # Strip {{step:N}} tags AND bare "step N" prose mentions before the
    # existing whole-log checks -- both are metadata about a claim's
    # provenance, not a second claim of their own.
    untagged_report = strip_citation_tags(report)

    unverified_numbers = flag_unverified_numbers(untagged_report, audit_log)
    print(f"\n[1] Unverified numbers (no matching tool-call output): "
          f"{len(unverified_numbers)}")
    for num, ctx in unverified_numbers:
        print(f"    ⚠ '{num}' in line: {ctx}")

    unverified_tokens = flag_unverified_tokens(untagged_report, audit_log)
    print(f"\n[2] Unverified quoted values/labels: {len(unverified_tokens)}")
    for tok in unverified_tokens:
        print(f"    ⚠ '{tok}' — not found in any tool output")

    if df_columns is not None:
        contradictions = flag_internal_contradictions(untagged_report, df_columns)
        print(f"\n[3] Possible internal contradictions: {len(contradictions)}")
        for col, nums in contradictions:
            print(f"    ⚠ Column '{col}' stated with conflicting values:")
            for n, ctx in nums:
                print(f"        {n} — near: \"{ctx}\"")
    else:
        print("\n[3] Skipped internal-contradiction check "
              "(pass df_columns=list(df.columns) to enable)")

    invalid_step_refs, miscited_verified_elsewhere, miscited_unverified = \
        flag_citation_mismatches(report, audit_log)
    serious_citation_problems = len(invalid_step_refs) + len(miscited_unverified)
    print(f"\n[4] Citation-tag problems: {serious_citation_problems} serious"
          f" (+ {len(miscited_verified_elsewhere)} minor indexing slips)")
    for step_idx, num, ctx in invalid_step_refs:
        print(f"    ⚠ '{num}' cites step {step_idx}, which doesn't exist "
              f"in audit_log — fabricated citation: {ctx}")
    for step_idx, num, ctx in miscited_unverified:
        print(f"    ⚠ '{num}' cites step {step_idx}, and this value "
              f"appears NOWHERE in audit_log — likely fabricated: {ctx}")
    for step_idx, num, ctx in miscited_verified_elsewhere:
        print(f"    · '{num}' cites step {step_idx}, but is actually "
              f"verified under a different step — indexing slip, not a "
              f"data problem: {ctx}")

    total_flags = (len(unverified_numbers) + len(unverified_tokens)
                   + (len(contradictions) if df_columns is not None else 0)
                   + serious_citation_problems)
    print(f"\n{'PASS — nothing serious flagged' if total_flags == 0 else f'{total_flags} serious issue(s) flagged — review above'}"
          f" ({len(miscited_verified_elsewhere)} minor indexing slip(s) not counted toward this total)")
    print("=" * 60)

    return {
        "unverified_numbers": unverified_numbers,
        "unverified_tokens": unverified_tokens,
        "invalid_step_refs": invalid_step_refs,
        "miscited_verified_elsewhere": miscited_verified_elsewhere,
        "miscited_unverified": miscited_unverified,
        "contradictions": contradictions if df_columns is not None else None,
    }


# ---------------------------------------------------------------------------
# USAGE (in your notebook, right after a run):
#
# from report_verifier import verify_report
# report, audit_log = run_eda_agent(df)
# print(report)
# ground_truth_summary(df)
# verify_report(report, audit_log, df_columns=list(df.columns))
# ---------------------------------------------------------------------------