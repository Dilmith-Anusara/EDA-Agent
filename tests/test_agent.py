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

    def _build_oversized_messages(self, n_tool_results=20, result_size=2000):
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
        messages = self._build_oversized_messages()
        before = agent._estimate_tokens(messages)
        assert before > agent.REQUEST_TOKEN_BUDGET

        agent._compact_messages_to_budget(messages)
        after = agent._estimate_tokens(messages)
        assert after <= agent.REQUEST_TOKEN_BUDGET

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