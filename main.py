"""
Driver script: runs the full EDA-agent pipeline end to end.

This replaces what used to be separate notebook cells. Run with:
    python main.py

Requires GROQ_API_KEY to already be set as an environment variable
(see setup steps -- `echo %GROQ_API_KEY%` should print your key).
"""

import pandas as pd
from rich.console import Console
from rich.markdown import Markdown

from agent import run_eda_agent, ground_truth_summary
from continue_conversation import continue_conversation
from report_verify import verify_report

console = Console()


def display_and_save_report(report: str, filename: str = "report.md") -> None:
    """Render the report readably in the terminal (via rich) AND save it
    to a .md file for a permanent, properly-rendered copy (open in VS Code
    with Ctrl+Shift+V, or view on GitHub). No model tokens involved --
    `report` is already a complete markdown string returned by
    run_eda_agent; this only changes how it's displayed/stored, not its
    content."""
    console.rule("[bold]FINAL REPORT[/bold]")
    console.print(Markdown(report))

    with open(filename, "w", encoding="utf-8") as f:
        f.write(report)
    console.print(f"\n[dim]Report also saved to {filename}[/dim]")


def main():
    # --- 1. Load your dataset -----------------------------------------
    # Replace this with your actual dataset path.
    df = pd.read_csv("your_dataset.csv")

    # --- 2. Run the agent -----------------------------------------------
    report, audit_log, messages = run_eda_agent(df)
    display_and_save_report(report)

    # --- 3. Independent ground truth, to compare by eye -----------------
    ground_truth_summary(df)

    # --- 4. Automated verification ---------------------------------------
    verify_report(report, audit_log, df_columns=list(df.columns))

    # --- 5. Optional: send a follow-up question into the same conversation
    # Uncomment to try it:
    #
    # boundary_index = len(audit_log)  # capture BEFORE calling continue_conversation
    # follow_up_text, audit_log, messages = continue_conversation(
    #     messages, audit_log, df,
    #     follow_up_message="Summarize the missingness pattern in one sentence.",
    # )
    # display_and_save_report(follow_up_text, filename="report_followup.md")


if __name__ == "__main__":
    main()