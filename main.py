"""
Driver script: runs the full EDA-agent pipeline end to end.

Produces TWO files, deliberately split by audience:
  - full_run_report.md      -- what an analyst actually reads: the agent's
                                report, plus a short trust banner. No raw
                                checker debug output.
  - verification_details.md -- ground truth + the full raw verification
                                output (citation slips, unverified-token
                                lists, etc.). Read only if the banner
                                flags something, or to double-check the
                                checker itself.

Run with:
    python main.py

Requires GROQ_API_KEY to already be set as an environment variable
(see setup steps -- `echo %GROQ_API_KEY%` should print your key).
"""

import io
import contextlib
from datetime import datetime

import pandas as pd
from rich.console import Console
from rich.markdown import Markdown

from eda_agent import run_eda_agent, ground_truth_summary
from continue_conversation import continue_conversation
from report_verify import verify_report, flag_stale_numeric_recall

console = Console()


def capture_stdout(func, *args, **kwargs):
    """Run a function that prints its own output (ground_truth_summary,
    verify_report both print internally rather than returning a string)
    and capture that printed text instead of letting it go straight to
    the terminal, uncaptured and unsaved. Returns (captured_text,
    return_value)."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        return_value = func(*args, **kwargs)
    return buffer.getvalue(), return_value


def build_trust_banner(verify_result: dict, stale_flags: list = None) -> str:
    """One short, analyst-facing line (or a few) summarizing verification
    results -- NOT the raw checker output. Uses the same 'serious vs
    minor' distinction report_verifier.py's own verify_report() applies
    internally, so this number always matches what's in the detailed
    printout, just without making the analyst read the printout to get it.

    'Serious' = unverified numbers + unverified quoted tokens + internal
    contradictions + citation problems that are fabricated or unverified
    anywhere in the log. Citation issues that are merely mis-indexed but
    verified elsewhere are NOT counted here -- they're a checker-internal
    detail, not something that should make an analyst distrust the report.
    """
    unverified_numbers = verify_result.get("unverified_numbers", [])
    unverified_tokens = verify_result.get("unverified_tokens", [])
    contradictions = verify_result.get("contradictions") or []
    invalid_step_refs = verify_result.get("invalid_step_refs", [])
    miscited_unverified = verify_result.get("miscited_unverified", [])

    serious_count = (
        len(unverified_numbers)
        + len(unverified_tokens)
        + len(contradictions)
        + len(invalid_step_refs)
        + len(miscited_unverified)
    )
    stale_count = len(stale_flags) if stale_flags else 0
    total = serious_count + stale_count

    if total == 0:
        return "> ✅ **Verification: all stated figures in this report were checked against the agent's actual tool calls. No issues found.**"

    parts = []
    if unverified_numbers:
        parts.append(f"{len(unverified_numbers)} unverified number(s)")
    if unverified_tokens:
        parts.append(f"{len(unverified_tokens)} unverified quoted value(s)")
    if contradictions:
        parts.append(f"{len(contradictions)} internal contradiction(s)")
    if invalid_step_refs or miscited_unverified:
        parts.append(f"{len(invalid_step_refs) + len(miscited_unverified)} unsupported citation(s)")
    if stale_count:
        parts.append(f"{stale_count} restated-without-reverifying number(s) in the follow-up")

    breakdown = ", ".join(parts)
    return (
        f"> ⚠️ **Verification: {total} issue(s) found** — {breakdown}. "
        f"See `verification_details.md` for exactly which statements."
    )


def display_section(title: str, content: str, as_markdown: bool = True) -> None:
    """Render one section readably in the terminal via rich."""
    console.rule(f"[bold]{title}[/bold]")
    if as_markdown:
        console.print(Markdown(content))
    else:
        console.print(content)


def build_document(title: str, sections: list) -> str:
    """Combine (section_title, content) pairs into one markdown document."""
    parts = [f"# {title}", f"*Generated: {datetime.now().isoformat(timespec='seconds')}*"]
    for section_title, content in sections:
        parts.append(f"\n---\n\n## {section_title}\n\n{content}")
    return "\n".join(parts)


def main():
    report_sections = []   # goes into full_run_report.md -- analyst-facing
    detail_sections = []   # goes into verification_details.md -- audit trail

    # --- 1. Load your dataset -----------------------------------------
    df = pd.read_csv("your_dataset.csv")

    # --- 2. Run the agent -----------------------------------------------
    report, audit_log, messages = run_eda_agent(df)

    # --- 3. Ground truth -- captured, goes to the detail file, not the
    # analyst-facing one (it's a developer/validation artifact, not
    # something an analyst needs to read).
    ground_truth_text, _ = capture_stdout(ground_truth_summary, df)
    detail_sections.append(("Ground Truth (independent computation)", f"```\n{ground_truth_text}```"))

    # --- 4. Automated verification -- captured. Full raw text goes to the
    # detail file; the analyst-facing file only gets the short banner.
    verify_text, verify_result = capture_stdout(
        verify_report, report, audit_log, df_columns=list(df.columns)
    )
    detail_sections.append(("Automated Verification (full detail)", f"```\n{verify_text}```"))

    banner = build_trust_banner(verify_result)

    # --- 5. Optional: send a follow-up question into the same conversation
    # Uncomment to try it.
    #
    # boundary_index = len(audit_log)  # capture BEFORE calling continue_conversation
    # follow_up_text, audit_log, messages = continue_conversation(
    #     messages, audit_log, df,
    #     follow_up_message="Summarize the missingness pattern in one sentence.",
    # )
    # stale_text, stale_flags = capture_stdout(
    #     flag_stale_numeric_recall, follow_up_text, audit_log, boundary_index
    # )
    # detail_sections.append(("Follow-up Stale-Recall Check", f"```\n{stale_text}```"))
    # follow_up_banner = build_trust_banner(verify_result, stale_flags)
    # report_sections.append((
    #     "Follow-up",
    #     f"{follow_up_banner}\n\n{follow_up_text}",
    # ))

    # --- 6. Assemble the analyst-facing report: banner + report, nothing else
    report_sections.insert(0, ("Agent Report", f"{banner}\n\n{report}"))

    full_report_doc = build_document("EDA Agent Report", report_sections)
    with open("full_run_report.md", "w", encoding="utf-8") as f:
        f.write(full_report_doc)

    detail_doc = build_document("Verification Details (audit trail)", detail_sections)
    with open("verification_details.md", "w", encoding="utf-8") as f:
        f.write(detail_doc)

    # Terminal: show the analyst-facing view by default.
    display_section("Agent Report", f"{banner}\n\n{report}")
    console.print(
        "\n[dim]Full report saved to full_run_report.md — "
        "raw verification detail saved to verification_details.md[/dim]"
    )


if __name__ == "__main__":
    main()