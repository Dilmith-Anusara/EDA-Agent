"""
Test suite for eda_agent.py.

Covers the pure/testable units: the tool functions (execute_python,
missingness_report, compute), the token-budget compaction, and
call_with_retry's compaction trigger via a mock (no real API call).
The full run_eda_agent() loop itself isn't tested here -- it's an
integration of the Groq API, which belongs in a live-transcript check,
not a mocked unit test that would just be testing the mock.

Requires GROQ_API_KEY to be set to ANYTHING (even a dummy value) for
import to succeed, since eda_agent.py constructs a Groq client at
import time. No real key or network access is needed to run these tests.

Run with:
    pytest tests/test_agent.py -v
"""

import os
os.environ.setdefault("GROQ_API_KEY", "dummy-key-for-tests")

import json
from unittest.mock import patch
import pandas as pd
import numpy as np
import pytest

import eda_agent as agent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "incident_id": [f"INC{i}" for i in range(100)],
        "crime_type": ["a", "b", "a", "c"] * 25,
        "suspect_age": [25.0, 30.0, None, 40.0] * 25,
    })


@pytest.fixture
def namespace(sample_df):
    return {"df": sample_df, "pd": pd, "np": np}


# ---------------------------------------------------------------------------
# execute_python
# ---------------------------------------------------------------------------

class TestExecutePython:

    def test_prints_are_captured(self, namespace):
        result = agent.execute_python("print(df.shape)", namespace)
        assert "(100, 3)" in result

    def test_no_output_gives_a_helpful_message_not_a_blank_string(self, namespace):
        """Silent code (no print) should say so, not return empty --
        an empty string is indistinguishable from a real error."""
        result = agent.execute_python("x = 5", namespace)
        assert "no printed output" in result.lower()

    def test_error_is_caught_not_raised(self, namespace):
        """A bad expression must return an error STRING back to the
        model, not raise and crash the whole agent loop."""
        result = agent.execute_python("1/0", namespace)
        assert "ERROR" in result

    def test_namespace_mutations_persist_across_calls(self, namespace):
        """Finding #4: the namespace must persist across calls within a
        run -- a variable defined in one call must be visible in the
        next, not recreated fresh each time."""
        agent.execute_python("df['new_col'] = df['suspect_age'] * 2", namespace)
        result = agent.execute_python("print('new_col' in df.columns)", namespace)
        assert "True" in result


# ---------------------------------------------------------------------------
# missingness_report
# ---------------------------------------------------------------------------

class TestMissingnessReport:

    def test_single_column_via_col(self, sample_df):
        result = agent.missingness_report(sample_df, col="suspect_age")
        assert "suspect_age" in result
        assert "non-null" in result
        assert "missing" in result

    def test_batch_via_cols(self, sample_df):
        """cols=[...] must return one line per column in a single call,
        not require repeated single-column calls."""
        result = agent.missingness_report(sample_df, cols=["incident_id", "suspect_age"])
        assert "incident_id" in result and "suspect_age" in result
        assert result.count("non-null") == 2

    def test_no_columns_specified_returns_error_not_crash(self, sample_df):
        result = agent.missingness_report(sample_df)
        assert "ERROR" in result

    def test_nonexistent_column_returns_error_not_crash(self, sample_df):
        result = agent.missingness_report(sample_df, col="does_not_exist")
        assert "ERROR" in result
        assert "not found" in result


# ---------------------------------------------------------------------------
# compute
# ---------------------------------------------------------------------------

class TestCompute:

    def test_single_expression(self, namespace):
        result = agent.compute(expression="df['crime_type'].nunique()", namespace=namespace)
        assert "= 3" in result

    def test_batch_expressions(self, namespace):
        """Fix: a live run burned 4 separate steps computing nunique()
        one column at a time before batch mode existed. Confirms the
        exact scenario that motivated the fix."""
        result = agent.compute(
            expressions=[
                "df['crime_type'].nunique()",
                "df['incident_id'].nunique()",
            ],
            namespace=namespace,
        )
        assert result.count("=") == 2
        assert "crime_type'].nunique() = 3" in result
        assert "incident_id'].nunique() = 100" in result

    def test_ratio_times_100_folded_into_expression(self, namespace):
        """Fix: the model must be ABLE to fold '* 100' into the
        expression itself (this is what the tool/prompt fix nudges
        toward) -- confirm the tool supports it correctly, since that's
        the actual mechanism the fix depends on."""
        result = agent.compute(
            expression="df['incident_id'].nunique() / len(df) * 100",
            namespace=namespace,
        )
        assert "= 100.0" in result

    def test_statement_not_expression_errors_cleanly(self, namespace):
        """compute() must reject statements (assignments, etc.) with a
        clean error, not crash -- it's expression-only by design,
        assignments belong in execute_python."""
        result = agent.compute(expression="x = 5", namespace=namespace)
        assert "ERROR" in result

    def test_non_scalar_result_warns_instead_of_silently_returning_a_series(self, namespace):
        result = agent.compute(expression="df['crime_type']", namespace=namespace)
        assert "NOTE" in result

    def test_partial_batch_failure_does_not_lose_the_good_results(self, namespace):
        """One bad expression in a batch must not prevent the other,
        valid expressions in the same call from returning results."""
        result = agent.compute(
            expressions=["df['crime_type'].nunique()", "df['nonexistent'].nunique()"],
            namespace=namespace,
        )
        assert "crime_type'].nunique() = 3" in result
        assert "ERROR" in result

    def test_no_arguments_returns_error_not_crash(self, namespace):
        result = agent.compute(namespace=namespace)
        assert "ERROR" in result


# ---------------------------------------------------------------------------
# Token-budget compaction
# ---------------------------------------------------------------------------

class TestTokenBudgetCompaction:

    def _build_oversized_messages(self, n_tool_results=20, result_size=300):
        # result_size tuned down from 2000 -- REQUEST_TOKEN_BUDGET was
        # tightened (5500 -> 4000) and _estimate_tokens' divisor tightened
        # (4 -> 2) after a live crash showed the old settings were ~2x too
        # generous for this JSON-heavy payload shape (see eda_agent.py).
        # 2000 chars/result under the new settings compacts ALL 20 tool
        # messages including the most recent one, which would make
        # test_system_message_and_most_recent_tool_result_preserved's
        # invariant untestable in this fixture, not actually false in
        # general -- 300 keeps this test exercising real compaction while
        # leaving enough headroom for the newest tool result to survive.
        messages = [{"role": "system", "content": "You are an EDA agent." * 50}]
        for i in range(n_tool_results):
            messages.append({
                "role": "assistant", "content": None,
                "tool_calls": [{"id": f"call_{i}", "type": "function",
                                 "function": {"name": "execute_python", "arguments": "{}"}}],
            })
            messages.append({"role": "tool", "tool_call_id": f"call_{i}",
                              "content": "x" * result_size})
        return messages

    def test_oversized_messages_get_compacted_under_budget(self):
        """Compares against the EFFECTIVE budget (REQUEST_TOKEN_BUDGET
        minus TOOLS_SCHEMA_TOKEN_ESTIMATE), not the raw budget -- that's
        the actual quantity _compact_messages_to_budget targets, since
        the `tools` schema's fixed overhead is now accounted for (see
        TestTokenEstimateDivisorFix.test_tools_schema_overhead_is_now_counted)."""
        effective_budget = agent.REQUEST_TOKEN_BUDGET - agent.TOOLS_SCHEMA_TOKEN_ESTIMATE
        messages = self._build_oversized_messages()
        before = agent._estimate_tokens(messages)
        assert before > effective_budget

        agent._compact_messages_to_budget(messages)
        after = agent._estimate_tokens(messages)
        assert after <= effective_budget

    def test_system_message_and_most_recent_tool_result_preserved(self):
        """Compaction must go oldest-first and must never touch the
        system message -- both are load-bearing for the model's
        behavior on the NEXT call."""
        messages = self._build_oversized_messages()
        original_system = messages[0]["content"]
        last_content = messages[-1]["content"]

        agent._compact_messages_to_budget(messages)

        assert messages[0]["content"] == original_system
        assert messages[-1]["content"] == last_content

    def test_already_under_budget_messages_are_untouched(self):
        """Compaction must be a true no-op when nothing needs shrinking --
        it should not touch messages just because it ran."""
        small = [{"role": "system", "content": "short"}, {"role": "user", "content": "hi"}]
        original = [dict(m) for m in small]
        agent._compact_messages_to_budget(small)
        assert small == original

    def test_call_with_retry_invokes_compaction_before_every_request(self):
        """The whole point of putting compaction inside call_with_retry
        (rather than duplicating it into continue_conversation
        separately) is that every caller gets it automatically. Confirm
        it's actually wired in, not just present as a dead function."""
        oversized = self._build_oversized_messages()
        with patch.object(agent, "_compact_messages_to_budget",
                           wraps=agent._compact_messages_to_budget) as spy:
            with patch.object(agent.client.chat.completions, "create",
                               return_value="FAKE_RESPONSE"):
                result = agent.call_with_retry(oversized, tools=[])
                assert spy.called
                assert result == "FAKE_RESPONSE"


# ---------------------------------------------------------------------------
# Malformed tool-call salvage (Groq tool_use_failed recovery)
# ---------------------------------------------------------------------------

class TestExtractCodeFromToolUseError:

    def test_json_quoted_code_shape(self):
        """The common shape: a JSON-quoted 'code' key exists but the
        outer structure is broken."""
        error_str = '{"code": "print(df.shape)\\nprint(1)"}'
        result = agent.extract_code_from_tool_use_error(error_str)
        assert result == "print(df.shape)\nprint(1)"

    def test_unparseable_error_returns_none_not_crash(self):
        """If neither salvage shape matches, must return None cleanly so
        the caller can fall back to nudging the model, not raise."""
        result = agent.extract_code_from_tool_use_error("completely unrelated text")
        assert result is None


# ---------------------------------------------------------------------------
# missing_coverage / REQUIRED_CHECKS (Session fix: class_balance loophole)
# ---------------------------------------------------------------------------

class TestMissingCoverage:

    def test_empty_log_flags_all_three_checks(self):
        gaps = agent.missing_coverage([])
        assert set(gaps) == {"missingness", "cardinality/skew", "class_balance"}

    def test_unrelated_compute_call_does_not_satisfy_class_balance(self):
        """Regression test for the fixed loophole: originally ANY
        compute() call satisfied this check, including an unrelated
        nunique-percentage call for a totally different column. That let
        a model finalize a report having never actually checked class
        balance. A compute() call with no 'value_counts' in it must NOT
        clear the class_balance gap."""
        log = [{"step": 0, "code": "compute(\"df['incident_id'].nunique() / len(df) * 100\")",
                "result": "... = 96.19"}]
        gaps = agent.missing_coverage(log)
        assert "class_balance" in gaps

    def test_value_counts_inside_compute_expression_satisfies_class_balance(self):
        """The fixed check must still recognize the legitimate case: a
        genuine class-balance check run via compute()."""
        log = [{"step": 0,
                "code": "compute(\"df['income'].value_counts(normalize=True)*100\")",
                "result": "... "}]
        gaps = agent.missing_coverage(log)
        assert "class_balance" not in gaps

    def test_value_counts_inside_execute_python_also_satisfies_class_balance(self):
        """value_counts run via execute_python (not compute()) must count
        too -- the check matches on the logged code string regardless of
        which tool produced it."""
        log = [{"step": 0, "code": "df['income'].value_counts()", "result": "..."}]
        gaps = agent.missing_coverage(log)
        assert "class_balance" not in gaps

    def test_missingness_check_satisfied_by_missingness_report_call(self):
        log = [{"step": 0, "code": "missingness_report(col='age')", "result": "..."}]
        gaps = agent.missing_coverage(log)
        assert "missingness" not in gaps

    def test_cardinality_check_satisfied_by_skew_or_nunique(self):
        log_skew = [{"step": 0, "code": "df['age'].skew()", "result": "..."}]
        log_nunique = [{"step": 0, "code": "df['age'].nunique()", "result": "..."}]
        assert "cardinality/skew" not in agent.missing_coverage(log_skew)
        assert "cardinality/skew" not in agent.missing_coverage(log_nunique)

    def test_all_checks_satisfied_returns_empty_list(self):
        log = [
            {"step": 0, "code": "missingness_report(col='age')", "result": "..."},
            {"step": 1, "code": "df['age'].skew()", "result": "..."},
            {"step": 2, "code": "df['income'].value_counts()", "result": "..."},
        ]
        assert agent.missing_coverage(log) == []


# ---------------------------------------------------------------------------
# invalid_citations (Session fix: structural citation-validity gate)
# ---------------------------------------------------------------------------

class TestInvalidCitations:

    def test_citation_to_real_step_is_valid(self):
        log = [{"step": 4, "code": "df['x'].skew()", "result": "..."}]
        report = "x skew = -0.60 {{step:4}}"
        assert agent.invalid_citations(report, log) == []

    def test_citation_to_nonexistent_step_is_invalid(self):
        """Core case this gate was added for: a fabricated second
        {{step:N}} tag on an ID-column claim, where that step number
        never ran."""
        log = [{"step": 7, "code": "df['id'].nunique()", "result": "..."}]
        report = "id column is high-cardinality {{step:7}} {{step:8}}"
        assert agent.invalid_citations(report, log) == [8]

    def test_multiple_invalid_citations_all_reported_sorted(self):
        log = [{"step": 1, "code": "", "result": ""}]
        report = "a {{step:9}} b {{step:3}} c {{step:1}}"
        assert agent.invalid_citations(report, log) == [3, 9]

    def test_no_citations_returns_empty_list(self):
        log = [{"step": 1, "code": "", "result": ""}]
        assert agent.invalid_citations("No citations in this text at all.", log) == []

    def test_empty_audit_log_any_citation_is_invalid(self):
        """No tool calls at all yet -- any {{step:N}} tag is necessarily
        fabricated, since nothing has run."""
        assert agent.invalid_citations("value {{step:0}}", []) == [0]


# ---------------------------------------------------------------------------
# compute() auto-repair for bare comprehensions
# ---------------------------------------------------------------------------

class TestComputeAutoRepair:

    def test_bare_comprehension_gets_wrapped_and_evaluated(self, namespace):
        """Core fix: the model kept passing a bare, unbracketed
        comprehension to compute(), which is invalid as a standalone
        eval() expression. This must be auto-repaired (wrapped in [...])
        and retried once rather than just erroring out."""
        result = agent.compute(
            expression="df[col].nunique() for col in ['crime_type']",
            namespace=namespace,
        )
        assert "ERROR" not in result
        assert "auto-repaired" in result
        assert "= [3]" in result

    def test_repair_only_triggers_for_bare_for_clauses(self, namespace):
        """Regression guard: an expression that's already validly
        bracketed must evaluate normally on the first try, not get a
        second '[...]' wrapped around it (which would silently change
        the result shape from a list to a list-of-one-list)."""
        result = agent.compute(
            expression="[df['crime_type'].nunique()]",
            namespace=namespace,
        )
        assert "auto-repaired" not in result
        assert "= [3]" in result

    def test_batch_mode_auto_repairs_individual_bad_expressions(self, namespace):
        """The auto-repair must apply per-expression inside a batch, not
        require the whole batch to be well-formed."""
        result = agent.compute(
            expressions=[
                "df['crime_type'].nunique()",
                "df[c].nunique() for c in ['incident_id']",
            ],
            namespace=namespace,
        )
        assert "crime_type'].nunique() = 3" in result
        assert "auto-repaired" in result
        assert "ERROR" not in result

    def test_non_comprehension_syntax_error_is_not_auto_repaired(self, namespace):
        """A genuine syntax error with no 'for' clause must fall through
        to the normal error path -- auto-repair is narrowly scoped to the
        bare-comprehension shape, not a general syntax-error fixer."""
        result = agent.compute(expression="1 +", namespace=namespace)
        assert "ERROR" in result
        assert "auto-repaired" not in result

    def test_repair_that_still_fails_falls_back_to_normal_error(self, namespace):
        """If wrapping in brackets still doesn't evaluate (e.g. the
        comprehension itself references something invalid), the function
        must fall back to the standard error message rather than raising
        or returning something misleading."""
        result = agent.compute(
            expression="df[c].nunique() for c in ['does_not_exist_col']",
            namespace=namespace,
        )
        # Bracket-wrapping succeeds syntactically, but the comprehension
        # itself raises a runtime KeyError (bad column name) -- that must
        # fall through to the standard ERROR message, not raise or hang.
        assert "ERROR" in result
        assert "auto-repaired" not in result


# ---------------------------------------------------------------------------
# run_eda_agent: coverage-gate / citation-gate wiring (dead-code bug fix)
# ---------------------------------------------------------------------------
#
# Session fix: missing_coverage()/invalid_citations() were defined
# correctly but never actually CALLED inside the `if not message.tool_calls`
# branch -- the old unconditional `return` was still there underneath.
# These tests mock the Groq client entirely and drive run_eda_agent()
# through a finalize attempt with incomplete coverage, to confirm the gate
# actually blocks the first attempt (forces continuation) rather than
# silently accepting it -- the exact case that would have passed before
# this fix and must fail now if the wiring regresses.

class _FakeFunction:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class _FakeToolCall:
    def __init__(self, call_id, name, arguments):
        self.id = call_id
        self.type = "function"
        self.function = _FakeFunction(name, arguments)


class _FakeMessage:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls

    def model_dump(self, exclude_none=True):
        d = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            d["tool_calls"] = [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in self.tool_calls
            ]
        elif exclude_none:
            pass
        return d


class _FakeChoice:
    def __init__(self, message):
        self.message = message


class _FakeResponse:
    def __init__(self, message):
        self.choices = [_FakeChoice(message)]


class TestRunEdaAgentCoverageGateWiring:

    def test_incomplete_coverage_finalize_attempt_is_rejected_not_returned(self, sample_df):
        """Step 0: model tries to finalize immediately with zero tool
        calls -- coverage is maximally incomplete (all three checks
        missing). If the gate is wired correctly, run_eda_agent must NOT
        return this as the final report; it must push a corrective user
        message and call the model again. If the dead-code bug regressed
        (gate computed but the old unconditional return still fires),
        this response WOULD be accepted as final, which is exactly what
        this test catches."""
        premature_report = _FakeMessage(content="Everything looks fine, no issues found.")
        # Step 1: a single turn that makes all three required tool calls
        # at once, so coverage becomes complete before the next finalize
        # attempt.
        covering_calls = _FakeMessage(tool_calls=[
            _FakeToolCall("c1", "missingness_report", '{"col": "suspect_age"}'),
            _FakeToolCall("c2", "execute_python", '{"code": "df[\\"suspect_age\\"].skew()"}'),
            _FakeToolCall("c3", "execute_python", '{"code": "df[\\"crime_type\\"].value_counts()"}'),
        ])
        final_report = _FakeMessage(content="Final report, coverage complete.")
        responses = [
            _FakeResponse(premature_report),
            _FakeResponse(covering_calls),
            _FakeResponse(final_report),
        ]

        with patch.object(agent, "call_with_retry", side_effect=responses) as mock_call:
            report, audit_log, messages = agent.run_eda_agent(sample_df, verbose=False)

        # The premature (uncorrected) report text must NOT be what was
        # returned -- proof the first finalize attempt was rejected.
        assert report != "Everything looks fine, no issues found."
        assert report == "Final report, coverage complete."
        # The gate must have forced the model to be called again instead
        # of accepting the first finalize attempt outright.
        assert mock_call.call_count == 3
        # A corrective message pointing out the coverage gap must have
        # been injected into the conversation before the next call.
        corrective = [m for m in messages if m.get("role") == "user"
                      and "you have not yet made a tool call covering" in m.get("content", "")]
        assert len(corrective) == 1
        assert "missingness" in corrective[0]["content"]
        assert "class_balance" in corrective[0]["content"]

    def test_fabricated_citation_on_finalize_is_rejected(self, sample_df):
        """Step 0: model finalizes with full coverage (so missing_coverage
        is clean) but cites a step number that was never run. The
        citation gate must independently catch and reject this too --
        confirms invalid_citations() is wired in as its own trigger, not
        only reachable via the coverage gap path."""
        covered_log_calls = [
            _FakeMessage(tool_calls=[_FakeToolCall(
                "c1", "missingness_report", '{"col": "suspect_age"}')]),
            _FakeMessage(tool_calls=[_FakeToolCall(
                "c2", "execute_python", '{"code": "df[\\"suspect_age\\"].skew()"}')]),
            _FakeMessage(tool_calls=[_FakeToolCall(
                "c3", "execute_python", '{"code": "df[\\"crime_type\\"].value_counts()"}')]),
            # Full coverage achieved. Now finalize with a fabricated citation.
            _FakeMessage(content="suspect_age mean is 30 {{step:99}}"),
            _FakeMessage(content="Corrected final report, no citations."),
        ]
        responses = [_FakeResponse(m) for m in covered_log_calls]

        with patch.object(agent, "call_with_retry", side_effect=responses):
            report, audit_log, messages = agent.run_eda_agent(sample_df, verbose=False)

        assert report == "Corrected final report, no citations."
        corrective = [m for m in messages if m.get("role") == "user"
                      and "fabricated" in m.get("content", "")]
        assert len(corrective) == 1
        assert "[99]" in corrective[0]["content"]


# ---------------------------------------------------------------------------
# Token-estimate divisor fix + graceful degradation on unrecoverable
# rate-limit/size errors (live crash, 2026-07)
# ---------------------------------------------------------------------------

class TestTokenEstimateDivisorFix:

    def test_fallback_estimate_uses_divisor_two_not_four(self):
        """Regression guard for the crash: a live request compacted to an
        estimated <=5500 tokens under the OLD divide-by-4 fallback was
        actually 8546-10678 real tokens per Groq -- roughly 2.0-2.6 real
        chars/token, not 4. Confirm the FALLBACK heuristic uses the
        tighter divisor. Forces the fallback path explicitly (rather than
        relying on the test environment having no network access to
        tiktoken's vocab) so this stays deterministic everywhere."""
        with patch.object(agent, "_get_tokenizer", return_value=None):
            text = "x" * 1000
            messages = [{"role": "user", "content": text}]
            estimate = agent._estimate_tokens(messages)
            raw_len = len(json.dumps(messages))
            assert estimate == raw_len // 2

    def test_real_tokenizer_used_when_available(self):
        """If a tokenizer loads, _estimate_tokens must use its actual
        .encode() length, not silently fall back to the heuristic anyway.
        Uses a fake tokenizer rather than requiring real network access
        to tiktoken's vocab, so this is deterministic in any environment."""
        class _FakeTokenizer:
            def encode(self, text, disallowed_special=()):
                return list(range(42))  # arbitrary fixed count

        with patch.object(agent, "_get_tokenizer", return_value=_FakeTokenizer()):
            assert agent._estimate_tokens([{"role": "user", "content": "anything"}]) == 42

    def test_tokenizer_load_failure_falls_back_cleanly(self):
        """_get_tokenizer must return None (not raise) when tiktoken is
        missing or its vocab can't be fetched (e.g. no network) -- this
        is the exact situation confirmed in this project's own sandbox
        (tiktoken installed, vocab download blocked by network policy),
        and _estimate_tokens must still work via the fallback."""
        with patch.object(agent, "_get_tokenizer", return_value=None):
            # Must not raise, and must return a positive int via the
            # fallback divisor.
            result = agent._estimate_tokens([{"role": "user", "content": "x" * 100}])
            assert isinstance(result, int) and result > 0

    def test_tools_schema_overhead_is_now_counted(self):
        """Fix: the `tools` schema is sent on every real request but was
        previously never counted by _estimate_tokens/
        _compact_messages_to_budget at all (both only ever looked at
        `messages`). TOOLS_SCHEMA_TOKEN_ESTIMATE must be a real, positive
        estimate of that fixed overhead, not zero or unset."""
        assert agent.TOOLS_SCHEMA_TOKEN_ESTIMATE > 0
        assert agent.TOOLS_SCHEMA_TOKEN_ESTIMATE == agent._estimate_tokens(agent.tools)

    def test_compaction_budget_accounts_for_tools_overhead(self):
        """_compact_messages_to_budget must compare `messages` against
        (budget - TOOLS_SCHEMA_TOKEN_ESTIMATE), not the raw budget --
        otherwise a request could look comfortably under budget on
        `messages` alone while the REAL request (messages + tools) was
        already much closer to Groq's actual cap. Forces the fallback
        tokenizer path for a deterministic, exact assertion."""
        with patch.object(agent, "_get_tokenizer", return_value=None):
            # Build messages sized to land exactly between the raw budget
            # and the tools-adjusted effective budget, so this test only
            # passes if the overhead is actually being subtracted.
            filler_chars = (agent.REQUEST_TOKEN_BUDGET * 2) - 200
            messages = [{"role": "system", "content": "s"},
                        {"role": "tool", "tool_call_id": "c0", "content": "x" * filler_chars}]
            est = agent._estimate_tokens(messages)
            effective = agent.REQUEST_TOKEN_BUDGET - agent.TOOLS_SCHEMA_TOKEN_ESTIMATE
            assert est > effective, "test fixture must actually exceed the effective budget"

            agent._compact_messages_to_budget(list(messages))  # should not raise
            compacted = [dict(m) for m in messages]
            agent._compact_messages_to_budget(compacted)
            assert agent._estimate_tokens(compacted) <= effective


class TestGracefulDegradationOnRateLimitError:

    def test_persistent_rate_limit_error_returns_partial_results_not_crash(self, sample_df):
        """Core fix for the live crash: a rate_limit_exceeded/'Request
        too large' error that survives all of call_with_retry's retries
        must not propagate as a bare exception and kill the whole run --
        it must return whatever audit_log/messages already exist, the
        same graceful-degradation contract MAX_ITERATIONS already has."""

        class FakeRateLimitError(Exception):
            body = {"error": {"code": "rate_limit_exceeded",
                               "message": "Request too large for model ..."}}

        # First call succeeds (a real tool call), second call raises the
        # unrecoverable error -- confirms partial progress is preserved.
        good_message = _FakeMessage(tool_calls=[
            _FakeToolCall("c1", "missingness_report", '{"col": "suspect_age"}'),
        ])
        responses = [_FakeResponse(good_message)]

        call_count = {"n": 0}

        def side_effect(*a, **kw):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return responses[0]
            raise FakeRateLimitError()

        with patch.object(agent, "call_with_retry", side_effect=side_effect):
            report, audit_log, messages = agent.run_eda_agent(sample_df, verbose=False)

        assert "rate-limit" in report.lower() or "request-size" in report.lower()
        # The one real tool call made before the crash must be preserved,
        # not lost.
        assert len(audit_log) == 1
        assert audit_log[0]["code"] == "missingness_report(col='suspect_age')"

    def test_tool_use_failed_errors_are_unaffected_by_this_change(self, sample_df):
        """Regression guard: the new rate-limit branch must not swallow
        or short-circuit the existing tool_use_failed salvage path --
        that path must still take priority and still `continue` the loop
        rather than returning early."""

        class FakeToolUseFailedError(Exception):
            body = {"error": {"code": "tool_use_failed",
                               "failed_generation": '{"code": "print(1)"}'}}

        final_report = _FakeMessage(content="Done.")

        call_count = {"n": 0}

        def side_effect(*a, **kw):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise FakeToolUseFailedError()
            return _FakeResponse(final_report)

        with patch.object(agent, "call_with_retry", side_effect=side_effect):
            report, audit_log, messages = agent.run_eda_agent(sample_df, verbose=False)

        # Salvaged code must have actually run and been logged.
        assert any(e["code"] == "print(1)" for e in audit_log)


# ---------------------------------------------------------------------------
# execute_python: structural correction for tool-name-as-Python-name
# confusion (prompt rule 0c alone confirmed insufficient on a live run)
# ---------------------------------------------------------------------------

class TestToolNameConfusionCorrection:

    def test_direct_call_shape_gets_pointed_correction(self, namespace):
        """Live-confirmed shape 1: print(missingness_report)."""
        result = agent.execute_python("print(missingness_report)", namespace)
        assert "ERROR" in result
        assert "CORRECTION" in result
        assert "TOOL, not a Python name" in result

    def test_assignment_shape_gets_pointed_correction(self, namespace):
        """Live-confirmed shape 2: result = missingness_report; print(result)."""
        result = agent.execute_python(
            "result = missingness_report\nprint(result)", namespace)
        assert "CORRECTION" in result

    def test_assignment_plus_method_call_shape_gets_pointed_correction(self, namespace):
        """Live-confirmed shape 3: res = missingness_report; res.to_dict() --
        confirms the correction fires even when the NameError happens on
        a later line, not just the first."""
        result = agent.execute_python(
            "res = missingness_report\nprint(res.to_dict())", namespace)
        assert "CORRECTION" in result

    def test_compute_tool_name_also_covered(self, namespace):
        """The same confusion could equally happen with 'compute' --
        confirm it's not hardcoded to missingness_report only."""
        result = agent.execute_python("print(compute)", namespace)
        assert "CORRECTION" in result
        assert "'compute' is a TOOL" in result

    def test_unrelated_nameerror_gets_no_false_correction(self, namespace):
        """Regression guard: an ordinary NameError unrelated to a tool
        name must NOT get the CORRECTION text -- that would be confusing,
        incorrect guidance for a genuine typo/undefined-variable bug."""
        result = agent.execute_python("print(some_undefined_variable)", namespace)
        assert "ERROR" in result
        assert "CORRECTION" not in result

    def test_non_nameerror_exceptions_unaffected(self, namespace):
        """Regression guard: this fix only special-cases NameError -- a
        different exception type (e.g. ZeroDivisionError) must behave
        exactly as before, no CORRECTION text attached."""
        result = agent.execute_python("1/0", namespace)
        assert "ERROR" in result
        assert "CORRECTION" not in result


# ---------------------------------------------------------------------------
# run_eda_agent: escalation on REPEATED tool-name confusion within a run
# ---------------------------------------------------------------------------

class TestToolNameConfusionEscalation:

    def test_second_occurrence_triggers_escalated_stop_message(self, sample_df):
        """Live-confirmed pattern: the model hit the tool-name-confusion
        CORRECTION at step 2, moved on, then made the identical mistake
        independently at step 6 in the same run -- one reactive message
        was not durable. The SECOND occurrence must inject a separate,
        blunt escalation message, not just rely on the tool result text
        landing a second time."""
        responses = [
            _FakeResponse(_FakeMessage(tool_calls=[
                _FakeToolCall("c1", "execute_python", '{"code": "print(missingness_report)"}')])),
            _FakeResponse(_FakeMessage(tool_calls=[
                _FakeToolCall("c2", "execute_python", '{"code": "print(1)"}')])),
            _FakeResponse(_FakeMessage(tool_calls=[
                _FakeToolCall("c3", "execute_python", '{"code": "print(missingness_report)"}')])),
            _FakeResponse(_FakeMessage(tool_calls=[
                _FakeToolCall("c4", "missingness_report", '{"col": "suspect_age"}'),
                _FakeToolCall("c5", "execute_python", '{"code": "df[\\"suspect_age\\"].skew()"}'),
                _FakeToolCall("c6", "execute_python", '{"code": "df[\\"crime_type\\"].value_counts()"}'),
            ])),
            _FakeResponse(_FakeMessage(content="Final report, done.")),
        ]

        with patch.object(agent, "call_with_retry", side_effect=responses):
            report, audit_log, messages = agent.run_eda_agent(sample_df, verbose=False)

        escalations = [m for m in messages if m.get("role") == "user"
                       and m.get("content", "").startswith("STOP:")]
        assert len(escalations) == 1
        assert "2nd time" in escalations[0]["content"]

    def test_single_occurrence_does_not_escalate(self, sample_df):
        """A single tool-name-confusion mistake must NOT trigger the
        escalation -- it's specifically for repeats. One occurrence
        should get only the existing CORRECTION text in the tool result,
        no separate STOP message."""
        responses = [
            _FakeResponse(_FakeMessage(tool_calls=[
                _FakeToolCall("c1", "execute_python", '{"code": "print(missingness_report)"}')])),
            _FakeResponse(_FakeMessage(tool_calls=[
                _FakeToolCall("c2", "missingness_report", '{"col": "suspect_age"}'),
                _FakeToolCall("c3", "execute_python", '{"code": "df[\\"suspect_age\\"].skew()"}'),
                _FakeToolCall("c4", "execute_python", '{"code": "df[\\"crime_type\\"].value_counts()"}'),
            ])),
            _FakeResponse(_FakeMessage(content="Final report, done.")),
        ]

        with patch.object(agent, "call_with_retry", side_effect=responses):
            report, audit_log, messages = agent.run_eda_agent(sample_df, verbose=False)

        escalations = [m for m in messages if m.get("role") == "user"
                       and m.get("content", "").startswith("STOP:")]
        assert escalations == []

    def test_unrelated_execute_python_errors_never_escalate(self, sample_df):
        """Regression guard: an ordinary execute_python error (no
        CORRECTION text) must never contribute to the escalation
        counter, no matter how many times it happens."""
        responses = [
            _FakeResponse(_FakeMessage(tool_calls=[
                _FakeToolCall("c1", "execute_python", '{"code": "1/0"}')])),
            _FakeResponse(_FakeMessage(tool_calls=[
                _FakeToolCall("c2", "execute_python", '{"code": "1/0"}')])),
            _FakeResponse(_FakeMessage(tool_calls=[
                _FakeToolCall("c3", "missingness_report", '{"col": "suspect_age"}'),
                _FakeToolCall("c4", "execute_python", '{"code": "df[\\"suspect_age\\"].skew()"}'),
                _FakeToolCall("c5", "execute_python", '{"code": "df[\\"crime_type\\"].value_counts()"}'),
            ])),
            _FakeResponse(_FakeMessage(content="Final report, done.")),
        ]

        with patch.object(agent, "call_with_retry", side_effect=responses):
            report, audit_log, messages = agent.run_eda_agent(sample_df, verbose=False)

        escalations = [m for m in messages if m.get("role") == "user"
                       and m.get("content", "").startswith("STOP:")]
        assert escalations == []


# ---------------------------------------------------------------------------
# run_eda_agent: content-aware citation check (real step, wrong content)
# ---------------------------------------------------------------------------

class TestFabricatedCitationToRealStep:

    def test_finalize_rejected_when_cited_step_content_does_not_match(self, sample_df):
        """Live-confirmed severe case: invalid_citations() alone only
        checks that a cited step NUMBER exists -- it waved through a
        report citing {{step:0}} for a skew value where step 0 was a
        real audit_log entry (missingness_report + skew + value_counts,
        all logged under the same loop step since they came from one
        model turn), just not one containing that number at all.
        Confirms the finalize gate now also rejects this via
        flag_citation_mismatches's miscited_unverified bucket, not just
        nonexistent step numbers."""
        covering_calls = _FakeMessage(tool_calls=[
            _FakeToolCall("c1", "missingness_report", '{"col": "suspect_age"}'),
            _FakeToolCall("c2", "execute_python", '{"code": "df[\\"suspect_age\\"].skew()"}'),
            _FakeToolCall("c3", "execute_python", '{"code": "df[\\"crime_type\\"].value_counts()"}'),
        ])
        # Cites step 0 (real, but none of ITS results contain -0.09, and
        # -0.09 appears nowhere else in the session -- fabricated).
        fabricated_report = _FakeMessage(
            content="suspect_age skew = -0.09 {{step:0}}")
        final_report = _FakeMessage(content="Corrected final report.")
        responses = [
            _FakeResponse(covering_calls),
            _FakeResponse(fabricated_report),
            _FakeResponse(final_report),
        ]

        with patch.object(agent, "call_with_retry", side_effect=responses) as mock_call:
            report, audit_log, messages = agent.run_eda_agent(sample_df, verbose=False)

        assert report == "Corrected final report."
        assert mock_call.call_count == 3
        corrective = [m for m in messages if m.get("role") == "user"
                      and "fabricated, not just miscited" in m.get("content", "")]
        assert len(corrective) == 1

    def test_indexing_slip_alone_does_not_block_finalization(self, sample_df):
        """Regression guard: a citation to the WRONG but still-correct
        step (the value IS verified, just filed under an adjacent step
        number from a separate turn -- an indexing slip) must NOT be
        treated as fabrication and must NOT block finalization, per this
        project's own established low-severity handling of that case.
        Uses two separate turns so the real nunique value and the
        misattributed citation land under genuinely different step
        indices."""
        turn0 = _FakeMessage(tool_calls=[
            _FakeToolCall("c1", "missingness_report", '{"col": "suspect_age"}'),
            _FakeToolCall("c2", "execute_python", '{"code": "df[\\"suspect_age\\"].skew()"}'),
        ])
        turn1 = _FakeMessage(tool_calls=[
            _FakeToolCall("c3", "execute_python", '{"code": "df[\\"crime_type\\"].value_counts()"}'),
            _FakeToolCall("c4", "compute", '{"expression": "df[\\"suspect_age\\"].nunique()"}'),
        ])
        # Real nunique value (3, for this fixture's suspect_age column)
        # is logged under step 1 (turn1); citing it to step 0 (a real
        # step, different content) is an indexing slip, not fabrication.
        report_with_slip = _FakeMessage(
            content="suspect_age nunique = 3 {{step:0}}")
        responses = [_FakeResponse(turn0), _FakeResponse(turn1), _FakeResponse(report_with_slip)]

        with patch.object(agent, "call_with_retry", side_effect=responses) as mock_call:
            report, audit_log, messages = agent.run_eda_agent(sample_df, verbose=False)

        assert report == "suspect_age nunique = 3 {{step:0}}"
        assert mock_call.call_count == 3