from agent import call_with_retry, tools, execute_python, missingness_report, compute, MAX_ITERATIONS, extract_code_from_tool_use_error
import numpy as np
import pandas as pd
import json

def continue_conversation(messages, audit_log, df, follow_up_message, verbose=True):
    """Send one more user message into an already-completed run's
    conversation, and let the model respond (with tool access, same as
    before). Reuses `messages`/`audit_log` from a prior run_eda_agent()
    call, so this is much cheaper than starting a fresh run.

    Includes the SAME malformed-tool-call salvage/recovery path as
    run_eda_agent's main loop -- an earlier version of this helper
    omitted it and crashed uncaught the first time the model produced a
    malformed call in a follow-up turn. Malformed calls are not just a
    fresh-run phenomenon; don't assume a helper that reuses call_with_retry
    is safe without this handling.
    """
    messages.append({"role": "user", "content": follow_up_message})
    # Same reasoning as run_eda_agent's namespace fix: copy so this turn's
    # mutations never leak back into the caller's own dataframe variable.
    namespace = {"df": df.copy(), "pd": pd, "np": np}

    # NOTE on Finding #9: tool_choice="required" was tried here and
    # abandoned. Live testing showed gpt-oss-120b (via Groq) fails to
    # comply with "required" 100% of the time in a continuation context
    # (long history + soft follow-up prompt) -- it consistently generates
    # a full prose answer instead of a tool call, which Groq rejects as
    # tool_use_failed. Retrying against this only burns tokens for zero
    # gain; it is not intermittent noncompliance, it is a hard wall for
    # this model/context combination. Verification is enforced entirely
    # post-hoc instead: run flag_stale_numeric_recall() on report2 after
    # every call to this function, using the boundary index captured
    # before the call. See report_verifier.py.

    for step in range(MAX_ITERATIONS):
        try:
            response = call_with_retry(messages, tools)
        except Exception as e:
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
                        print(f"\n[Follow-up step {step}] Malformed tool call salvaged from error text:\n{salvaged_code}")
                    result = execute_python(salvaged_code, namespace)
                    audit_log.append({"step": f"followup-{step}", "code": salvaged_code, "result": result})
                    if verbose:
                        print(f"[Follow-up step {step}] RESULT:\n{result}")
                    messages.append({
                        "role": "user",
                        "content": (
                            f"(Your previous tool call was malformed, but I salvaged "
                            f"and ran this code anyway: {salvaged_code}\nResult:\n{result}\n"
                            f"Continue, using the proper function-calling interface, "
                            f"not text-formatted tool calls.)"
                        ),
                    })
                    continue
                else:
                    if verbose:
                        print(f"\n[Follow-up step {step}] Malformed tool call, could not salvage. Nudging model.")
                    messages.append({
                        "role": "user",
                        "content": (
                            "Your last tool call was malformed and could not be parsed. "
                            "Please retry using the proper function-calling interface, "
                            "not a text-formatted call."
                        ),
                    })
                    continue
            raise

        choice = response.choices[0]
        message = choice.message
        messages.append(message.model_dump(exclude_none=True))

        if not message.tool_calls:
            if verbose:
                print(f"\n[Follow-up step {step}] Model responded (no tool calls). "
                      f"Run flag_stale_numeric_recall() on this response to check "
                      f"for unverified recalled numbers.")
            return message.content, audit_log, messages

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

            audit_log.append({"step": f"followup-{step}", "code": logged_code, "result": result})
            if verbose:
                print(f"\n[Follow-up step {step}] MODEL CALLED {fn_name}:\n{logged_code}")
                print(f"[Follow-up step {step}] RESULT:\n{result}")

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })

    return "Hit MAX_ITERATIONS in follow-up without a final report.", audit_log, messages