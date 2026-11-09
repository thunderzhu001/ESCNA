"""Small, independently logged v3 transport; does not change frozen v2 behavior."""
from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import requests


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def safe_text(text, secret=""):
    text = str(text)
    if secret:
        text = text.replace(secret, "[REDACTED]")
    return re.sub(r"sk-[A-Za-z0-9._-]{12,}", "[REDACTED]", text)


def schema_for(kind, options=(), fact_ids=(), *, strict_disclosure=False):
    if kind == "facilitator":
        properties = {"message": {"type": "string"}}
    elif kind == "participant":
        properties = {
            "message": {"type": "string"},
            "disclose_fact_ids": {"type": "array", "items": {"type": "string"}},
            "recommendation": {"type": "string", "enum": [*options, "ABSTAIN"]},
        }
    elif kind == "ballot":
        properties = {
            "recommendation": {"type": "string", "enum": [*options, "ABSTAIN"]},
            "acceptable_options": {"type": "array", "items": {"type": "string", "enum": list(options)}},
        }
    else:
        raise ValueError("unknown response contract")
    if kind == "participant" and strict_disclosure:
        field = properties["disclose_fact_ids"]
        field["maxItems"] = len(fact_ids)
        if fact_ids:
            field["items"]["enum"] = list(fact_ids)
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


def parse_contract(text, kind, options=(), fact_ids=()):
    """Validate the three concrete contracts, not an incomplete JSON Schema engine."""
    def pairs(values):
        obj = {}
        for key, value in values:
            if key in obj:
                raise ValueError("duplicate JSON key")
            obj[key] = value
        return obj
    def nonfinite(_):
        raise ValueError("non-finite JSON value")
    value = json.loads(text, object_pairs_hook=pairs, parse_constant=nonfinite)
    schema = schema_for(kind, options, fact_ids)
    if not isinstance(value, dict) or set(value) != set(schema["required"]):
        raise ValueError("wrong response fields")
    if "message" in value and (not isinstance(value["message"], str) or not value["message"].strip()):
        raise ValueError("empty or non-string message")
    if "recommendation" in value and value["recommendation"] not in [*options, "ABSTAIN"]:
        raise ValueError("invalid recommendation")
    for field, allowed in (("disclose_fact_ids", fact_ids), ("acceptable_options", options)):
        if field in value:
            items = value[field]
            if not isinstance(items, list) or any(not isinstance(x, str) or x not in allowed for x in items):
                raise ValueError(f"invalid {field}")
            if len(items) != len(set(items)):
                raise ValueError(f"duplicate {field}")
    if kind == "ballot" and value["recommendation"] != "ABSTAIN" and value["recommendation"] not in value["acceptable_options"]:
        raise ValueError("recommended option must be acceptable")
    if kind == "ballot" and value["recommendation"] == "ABSTAIN" and value["acceptable_options"]:
        raise ValueError("an abstaining ballot must have no acceptable options")
    return value


def request_payload(profile, system, user, schema):
    # Sending the same textual contract to every backend avoids privileged
    # schema-only instructions for the backends with native structured output.
    system = system + "\nReturn exactly one JSON object, with no Markdown. JSON schema: " + json.dumps(schema)
    if profile["api"] == "gemini_generate_content":
        return {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {
                "maxOutputTokens": 512,
                "thinkingConfig": {"thinkingLevel": "low"},
                "responseMimeType": "application/json",
                "responseJsonSchema": schema,
            },
        }
    payload = {
        "model": profile["model"],
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": 0.0, "max_tokens": 512, "stream": False,
    }
    if profile["provider"] == "qwen":
        payload.update(enable_thinking=False, response_format={
            "type": "json_schema", "json_schema": {"name": "escna", "strict": True, "schema": schema},
        })
    elif profile["provider"] == "deepseek":
        payload.update(thinking={"type": "disabled"}, response_format={"type": "json_object"})
    else:
        raise ValueError("unregistered provider")
    return payload


def response_fields(profile, data):
    if not isinstance(data, dict):
        raise ValueError("response envelope must be an object")
    if profile["api"] == "gemini_generate_content":
        candidate = (data.get("candidates") or [{}])[0]
        parts = candidate.get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts if not p.get("thought", False))
        usage = data.get("usageMetadata", {})
        output = usage.get("candidatesTokenCount")
        reasoning = usage.get("thoughtsTokenCount")
        fields = {
            "text": text, "observed_model": data.get("modelVersion"),
            "response_id": data.get("responseId") or data.get("id"),
            "finish_reason": candidate.get("finishReason"),
            "input_tokens": usage.get("promptTokenCount"),
            "output_tokens": output, "reasoning_tokens": reasoning,
            "cached_input_tokens": usage.get("cachedContentTokenCount"),
            "total_tokens": usage.get("totalTokenCount"),
            "completion_including_reasoning": (
                output + reasoning if isinstance(output, int) and isinstance(reasoning, int)
                else usage["totalTokenCount"] - usage["promptTokenCount"]
                if isinstance(usage.get("totalTokenCount"), int) and isinstance(usage.get("promptTokenCount"), int)
                else None
            ),
            "reasoning_usage_reported": "thoughtsTokenCount" in usage,
            "reasoning_text_present": any(p.get("thought", False) for p in parts),
        }
    else:
        choice = (data.get("choices") or [{}])[0]
        usage = data.get("usage", {})
        details = usage.get("completion_tokens_details", {}) or {}
        fields = {
            "text": choice.get("message", {}).get("content") or "",
            "observed_model": data.get("model"), "response_id": data.get("id"),
            "finish_reason": choice.get("finish_reason"),
            "input_tokens": usage.get("prompt_tokens"),
            "output_tokens": usage.get("completion_tokens"),
            "reasoning_tokens": details.get("reasoning_tokens"),
            "reasoning_usage_reported": "reasoning_tokens" in details,
            "reasoning_text_present": bool(choice.get("message", {}).get("reasoning_content")),
            "completion_including_reasoning": usage.get("completion_tokens"),
            "cached_input_tokens": usage.get("prompt_cache_hit_tokens", (usage.get("prompt_tokens_details") or {}).get("cached_tokens")),
            "total_tokens": usage.get("total_tokens"),
        }
    fields["raw_usage"] = usage
    if profile["provider"] == "gemini" and fields["reasoning_tokens"] is None:
        total_generated = fields["completion_including_reasoning"]
        visible = fields["output_tokens"]
        fields["unitemized_generated_tokens"] = (total_generated-visible
            if isinstance(total_generated, int) and isinstance(visible, int) else None)
        fields["thinking_accounting"] = "not_itemized; total generation retained; residual is not independently reported thinking"
    fields["reported_monetary_cost"] = None
    fields["reported_cost_fields"] = {
        source_name + "." + name: source[name]
        for source_name, source in (("usage", usage), ("response", data))
        for name in ("cost", "total_cost", "currency") if name in source
    }
    # Preserve missing usage as null. Do not call unknown costs zero.
    return fields


def validate_envelope(profile, fields):
    if fields["observed_model"] not in profile["accepted_response_models"]:
        raise ValueError("unexpected or missing returned model ID")
    if fields["finish_reason"] not in {"stop", "STOP"}:
        raise ValueError("response did not finish normally")
    for key in ("input_tokens", "output_tokens", "completion_including_reasoning"):
        value = fields[key]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError("missing or invalid token usage")
    budget_basis = profile.get("completion_budget_basis", "total_generated")
    if budget_basis not in {"visible_output", "total_generated"}:
        raise ValueError("unknown completion budget basis")
    budget_field = "output_tokens" if budget_basis == "visible_output" else "completion_including_reasoning"
    if fields[budget_field] > 512:
        raise ValueError("completion exceeds shared 512-token cap")
    if budget_basis == "visible_output" and profile["provider"] == "gemini":
        thought = fields["reasoning_tokens"]
        if thought is not None and (not isinstance(thought, int) or isinstance(thought, bool) or thought < 0):
            raise ValueError("invalid Gemini thinking usage")
        if fields["completion_including_reasoning"] < fields["output_tokens"]:
            raise ValueError("total generation cannot be below visible output")
        if thought is not None and fields["completion_including_reasoning"] != fields["output_tokens"] + thought:
            raise ValueError("inconsistent Gemini generation usage")
    if profile["provider"] in {"qwen", "deepseek"} and (fields["reasoning_text_present"] or (fields["reasoning_tokens"] or 0) > 0):
        raise ValueError("reasoning observed despite disabled thinking")


class LoggedClient:
    """One append-only JSONL, one worker per file, at most one service retry.

    Interrupted started requests are left unresolved, never silently resubmitted.
    Failed content is terminal. Headers/keys are never serialized.
    """
    def __init__(self, profile, ledger, *, retries=0, session=None):
        if retries not in (0, 1):
            raise ValueError("only zero or one service retry")
        self.profile, self.ledger, self.retries = profile, Path(ledger), retries
        self.session = session or requests.Session()
        self.ledger.parent.mkdir(parents=True, exist_ok=True)
        self.known = {}
        if self.ledger.exists():
            for line in self.ledger.read_text().splitlines():
                row = json.loads(line)
                self.known[row["request_id"]] = row

    def append(self, row):
        with self.ledger.open("a", encoding="utf8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        self.known[row["request_id"]] = row

    def generate(self, request_id, system, user, *, kind, options=(), fact_ids=(), metadata=None):
        schema = schema_for(kind, options, fact_ids,
                            strict_disclosure=self.profile.get("strict_disclosure", False))
        payload = request_payload(self.profile, system, user, schema)
        context = {"kind": kind, "options": list(options), "allowed_fact_ids": list(fact_ids), "metadata": metadata or {}}
        if request_id in self.known:
            previous = self.known[request_id]
            if previous.get("request") != payload or previous.get("profile") != self.profile:
                raise ValueError("request ID reused with a changed request/profile")
            if previous.get("validation_context", context) != context:
                raise ValueError("request ID reused with a changed validation context")
            if previous["status"] == "started":
                raise RuntimeError("unresolved request: inspect before any reissue")
            return previous
        key = os.environ.get(self.profile["key_env"])
        if not key:
            raise RuntimeError(f"missing {self.profile['key_env']}; no request sent")
        url = self.profile["url"]
        if not url.startswith("https://"):
            raise ValueError("HTTPS endpoint required")
        headers = {"Content-Type": "application/json", "User-Agent": "ESCNA-Research/v3"}
        if self.profile["api"] == "gemini_generate_content":
            headers["x-goog-api-key"] = key
        else:
            headers["Authorization"] = f"Bearer {key}"
        base = {"request_id": request_id, "profile": self.profile, "request": payload,
                "contract": kind, "metadata": metadata or {}, "validation_context": context}
        for attempt in range(1, self.retries + 2):
            self.append({**base, "status": "started", "attempt": attempt, "started_at": utc_now()})
            row = {**base, "status": "failed", "attempt": attempt, "started_at": utc_now()}
            started = time.perf_counter()
            retryable = False
            try:
                response = self.session.post(url, json=payload, headers=headers, timeout=(20, 120), allow_redirects=False)
                row["http_status"] = response.status_code
                row["raw_response"] = safe_text(response.text, key)
                if response.status_code != 200:
                    retryable = response.status_code == 429 or 500 <= response.status_code < 600
                    row["failure_kind"] = "rate_limit" if response.status_code == 429 else ("server" if response.status_code >= 500 else "http")
                else:
                    row["failure_kind"] = "response_contract"
                    data = json.loads(response.text)
                    fields = response_fields(self.profile, data)
                    row["response"] = fields
                    validate_envelope(self.profile, fields)
                    row["parsed"] = parse_contract(fields["text"], kind, options, fact_ids)
                    row["status"], row["failure_kind"] = "completed", None
            except (requests.Timeout, requests.ConnectionError) as exc:
                row["failure_kind"] = "network_or_timeout"
                row["detail"] = safe_text(exc, key)
                retryable = True
            except (ValueError, KeyError, TypeError, IndexError, AttributeError, requests.RequestException) as exc:
                row["failure_kind"] = row.get("failure_kind", "response_contract")
                row["detail"] = safe_text(exc, key)
            row.update(latency_seconds=time.perf_counter() - started, finished_at=utc_now())
            # A provider must not be able to reflect the credential into our logs.
            row = json.loads(safe_text(json.dumps(row, ensure_ascii=False), key))
            self.append(row)
            if not retryable or attempt > self.retries:
                return row
            time.sleep(1)
        raise AssertionError("unreachable")
