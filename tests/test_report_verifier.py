"""
Test suite for report_verify.py.

Every test here is built from a REAL bug found and fixed during this
project's development, using reconstructed transcript data -- not
generic/invented examples. Each test class covers one check function;
each test method's docstring says which finding/fix it comes from, so
this file also serves as a regression record.

Run with:
    pytest tests/test_report_verify.py -v
"""

import pytest

from report_verify import (
    flag_unverified_numbers,
    flag_unverified_tokens,
    flag_internal_contradictions,
    flag_citation_mismatches,
    flag_stale_numeric_recall,
    strip_citation_tags,
    verify_report,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def missingness_audit_log():
    """A minimal but realistic audit_log entry, matching the shape
    missingness_report() actually produces."""
    return [
        {
            "step": 1,
            "code": "missingness_report(cols=['suspect_age'])",
            "result": "suspect_age: 4153 non-null (79.1%), 1097 missing (20.9%)",
        }
    ]


@pytest.fixture
def skew_audit_log():
    """audit_log matching a real skew computation, including the negative
    value that triggered the unicode-minus bug."""
    return [
        {
            "step": 5,
            "code": "df['num_arrests'].skew()",
            "result": "num_arrests skew: -0.5989551285875486",
        }
    ]


# ---------------------------------------------------------------------------
# flag_unverified_numbers
# ---------------------------------------------------------------------------

class TestFlagUnverifiedNumbers:

    def test_genuine_unverified_number_is_flagged(self):
        """Baseline: a number with no matching tool call anywhere should
        be flagged. Without this passing, nothing else in this class
        means anything."""
        report = "The dataset has 4217 unique customers."
        flags = flag_unverified_numbers(report, [])
        assert len(flags) == 1
        assert flags[0][0] == "4217"

    def test_unicode_threshold_operator_not_flagged(self):
        """Fix: THRESHOLD_CONTEXT_RE originally only matched ASCII </>,
        not the unicode <=/>= glyphs the model actually writes. A
        'cardinality <= 10' rule description was a false positive
        before this fix."""
        report = "Low-cardinality columns (cardinality \u2264 10) were treated as categorical."
        assert flag_unverified_numbers(report, []) == []

    def test_unicode_minus_sign_on_verified_skew_value(self, skew_audit_log):
        """Fix: a genuinely-correct negative skew value written with
        U+2011 (non-breaking hyphen) instead of ASCII '-' as the minus
        sign was parsing as its POSITIVE magnitude, failing to match the
        pool's negative value, and getting falsely flagged as
        unverified. Confirmed on a live transcript."""
        report = "num_arrests skew is **\u20110.60** (slight left-skew)"
        assert flag_unverified_numbers(report, skew_audit_log) == []

    def test_fabricated_small_decimal_skew_is_still_caught(self, skew_audit_log):
        """Fix (magnitude blind spot): ignore_below=1.0 was silently
        discarding ANY number under 1.0 regardless of shape, including
        small-magnitude fabricated statistics. A report claiming
        num_arrests skew = 0.12 (never computed; real value -0.5990)
        passed check [1] completely undetected before this fix."""
        report = "num_arrests skewness is **0.12** (approximately symmetric)"
        flags = flag_unverified_numbers(report, skew_audit_log)
        assert len(flags) == 1
        assert flags[0][0] == "0.12"

    def test_train_test_split_jargon_excluded(self):
        """Fix: '80/20 train-test split' is ML methodology vocabulary,
        not a data claim -- was being flagged as two unverified numbers
        (80 and 20)."""
        report = "We recommend an 80/20 train-test split for model validation."
        assert flag_unverified_numbers(report, []) == []

    def test_genuine_fraction_shaped_number_still_checked(self):
        """Regression guard for the jargon fix above: a genuine data
        claim that merely LOOKS like a ratio ('80/300 rows had missing
        values') must not be swept into the same exclusion."""
        report = "We observed 80/300 rows with missing values."
        flags = flag_unverified_numbers(report, [])
        assert len(flags) == 2

    def test_thousands_space_normalized(self):
        """Fix: numbers using a Unicode space (narrow no-break space,
        etc.) as a thousands separator were splitting into two bogus
        fragments instead of being read as one number."""
        audit_log = [{"step": 0, "code": "", "result": "Shape: (5250, 12)"}]
        report = "The dataset has 5\u202f250 rows."
        assert flag_unverified_numbers(report, audit_log) == []

    def test_markdown_heading_number_excluded(self):
        """Ordered-list/heading markers ('### 6. Next steps') are
        structural, not data claims."""
        assert flag_unverified_numbers("### 6. Next steps", []) == []


# ---------------------------------------------------------------------------
# flag_internal_contradictions
# ---------------------------------------------------------------------------

class TestFlagInternalContradictions:

    def test_shared_bucket_range_not_a_false_contradiction(self):
        """Fix: 'Moderate missingness (10-20%): victim_age, victim_gender,
        weapon_used, case_status' -- a single range describing FOUR
        columns together -- was fabricating a false 10-vs-20
        contradiction for victim_age specifically, because neither
        number had a preceding column and the fallback picked the
        nearest one by raw distance."""
        report = (
            "Moderate missingness (10-20%): victim_age, victim_gender, "
            "weapon_used, case_status"
        )
        cols = ["victim_age", "victim_gender", "weapon_used", "case_status"]
        assert flag_internal_contradictions(report, cols) == []

    def test_genuine_same_column_contradiction_still_detected(self):
        """Regression guard: the fix above must not blind the checker to
        an actual contradiction (same column, same claim-type, two
        different values)."""
        report = "victim_age missing 12%. Later: victim_age missing 25%."
        flags = flag_internal_contradictions(report, ["victim_age"])
        assert len(flags) == 1
        assert flags[0][0] == "victim_age"

    def test_bundled_non_null_and_missing_pct_not_confused(self):
        """missingness_report's own bundled output format ('X non-null
        (Y%), Z missing (W%)') would fire a false contradiction on every
        column, every run, unless non-null% and missing% are compared
        only within their own claim-type."""
        report = "Category: 3312 non-null (63.09%), 1938 missing (36.91%)"
        assert flag_internal_contradictions(report, ["Category"]) == []


# ---------------------------------------------------------------------------
# flag_citation_mismatches / strip_citation_tags
# ---------------------------------------------------------------------------

class TestCitationChecks:

    def test_correct_citation_passes_cleanly(self, skew_audit_log):
        report = "num_arrests skew = -0.60 {{step:5}}"
        invalid, verified_elsewhere, unverified = flag_citation_mismatches(
            report, skew_audit_log
        )
        assert invalid == [] and verified_elsewhere == [] and unverified == []

    def test_citation_to_nonexistent_step_is_fabricated(self, skew_audit_log):
        """A step index that never ran at all is a fabricated citation --
        the most severe tier."""
        report = "num_arrests skew = -0.60 {{step:99}}"
        invalid, _, _ = flag_citation_mismatches(report, skew_audit_log)
        assert len(invalid) == 1
        assert invalid[0][0] == 99

    def test_off_by_one_indexing_slip_is_downgraded_not_fabrication(self):
        """Fix: confirmed on live transcripts, the model consistently
        cites the step AFTER the one that actually produced a value
        (nunique computed at Step 2, cited as {{step:3}}). Step 3 is a
        REAL step in this scenario (it exists, it just computed
        something else) -- that's what makes this an indexing slip
        rather than a fabricated step reference. The number is genuinely
        verified elsewhere -- this must NOT be treated as fabrication,
        or the false-positive rate explodes (17-28 issues per run before
        this fix, almost all noise)."""
        audit_log = [
            {"step": 2, "code": "", "result": "suspect_age nunique = 218"},
            {"step": 3, "code": "", "result": "property_loss_usd_num converted"},
        ]
        report = "suspect_age: 218 unique {{step:3}}"
        invalid, verified_elsewhere, unverified = flag_citation_mismatches(
            report, audit_log
        )
        assert invalid == [] and unverified == []
        assert len(verified_elsewhere) == 1

    def test_citation_to_real_step_with_wrong_value_is_unverified(self, skew_audit_log):
        """Cites a real step, but that step's actual output doesn't
        contain the claimed number -- this is the genuinely serious
        case, distinct from an indexing slip."""
        report = "incident_id has 4200 unique values {{step:5}}"
        _, _, unverified = flag_citation_mismatches(report, skew_audit_log)
        assert len(unverified) == 1

    def test_threshold_boilerplate_near_citation_not_fabricated(self):
        """Fix: '(per rule >10) {{step:2}}' -- boilerplate classification
        rule text the citation tag happened to land near -- was flagged
        as a fabricated citation. Same THRESHOLD_CONTEXT_RE exclusion
        flag_unverified_numbers already had, ported to the citation
        checker."""
        audit_log = [{"step": 2, "code": "", "result": "num_arrests nunique = 11"}]
        report = "num_arrests (11 unique) -- continuous (per rule >10) {{step:2}}"
        invalid, verified_elsewhere, unverified = flag_citation_mismatches(
            report, audit_log
        )
        assert invalid == [] and unverified == []

    def test_strip_citation_tags_removes_tag_only(self):
        """Tag digits must not be mistaken for a second, bare unverified
        number claim once stripped."""
        text = "num_arrests skew = -0.60 {{step:5}}"
        stripped = strip_citation_tags(text)
        assert "{{step:5}}" not in stripped
        assert "-0.60" in stripped

    def test_stripped_correctly_cited_number_does_not_double_flag(self, skew_audit_log):
        report = "num_arrests skew = -0.60 {{step:5}}"
        stripped = strip_citation_tags(report)
        assert flag_unverified_numbers(stripped, skew_audit_log) == []


# ---------------------------------------------------------------------------
# flag_stale_numeric_recall (Finding #9)
# ---------------------------------------------------------------------------

class TestFlagStaleNumericRecall:

    def test_restated_old_number_with_no_fresh_verification_is_flagged(self):
        """Core Finding #9 case: a follow-up turn with ZERO tool calls
        restates a number from an earlier missingness_report call. This
        must be flagged as stale recall."""
        audit_log = [
            {
                "step": 1,
                "code": "missingness_report(col='suspect_gender')",
                "result": "suspect_gender: 3837 non-null (73.09%), 1413 missing (26.91%)",
            }
        ]
        boundary = len(audit_log)  # nothing added after this -- no fresh verification
        follow_up = "suspect_gender is missing in 26.91% of rows."
        flags = flag_stale_numeric_recall(follow_up, audit_log, boundary)
        assert len(flags) == 1

    def test_freshly_reverified_number_is_not_flagged(self):
        """The same number, but WITH a fresh tool call after the
        boundary re-confirming it, must not be flagged -- that's the
        whole point of the check."""
        audit_log = [
            {
                "step": 1,
                "code": "missingness_report(col='suspect_gender')",
                "result": "suspect_gender: 3837 non-null (73.09%), 1413 missing (26.91%)",
            }
        ]
        boundary = len(audit_log)
        audit_log.append(
            {
                "step": "followup-0",
                "code": "missingness_report(col='suspect_gender')",
                "result": "suspect_gender: 3837 non-null (73.09%), 1413 missing (26.91%)",
            }
        )
        follow_up = "suspect_gender is missing in 26.91% of rows."
        flags = flag_stale_numeric_recall(follow_up, audit_log, boundary)
        assert flags == []


# ---------------------------------------------------------------------------
# verify_report (top-level integration)
# ---------------------------------------------------------------------------

class TestVerifyReportIntegration:

    def test_clean_report_produces_no_flags(self, skew_audit_log):
        report = "num_arrests skew = -0.60 {{step:5}}"
        result = verify_report(report, skew_audit_log, df_columns=["num_arrests"])
        assert result["unverified_numbers"] == []
        assert result["unverified_tokens"] == []
        assert result["invalid_step_refs"] == []
        assert result["miscited_unverified"] == []

    def test_fabricated_citation_surfaces_in_result_dict(self, skew_audit_log):
        report = "num_arrests skew = -0.60 {{step:99}}"
        result = verify_report(report, skew_audit_log, df_columns=["num_arrests"])
        assert len(result["invalid_step_refs"]) == 1