"""Isolated one-call worker. Importing this module never performs inference."""

from __future__ import annotations

import logging
import os
import sys
from threading import Timer
from time import perf_counter

from deploy.render.muse_diagnostic import (
    MODEL, PROMPT_SHA256, TOTAL_TIMEOUT_SECONDS, ParsedFields, SemanticChecks, TokenUsage, WorkerReport,
    canonical, digest, pinned_inputs, strict_json,
)

MAX_BODY_BYTES = 131072
URL = "https://integrate.api.nvidia.com/v1/chat/completions"


def infer_once(api_key: str, *, transport=None) -> WorkerReport:
    """Only tests inject MockTransport. Real transport has no retries/proxies."""
    import httpx
    from alpha_engine.ai.prompts.loader import load_recommendation_prompt
    from alpha_engine.ai.providers.nvidia import NvidiaProvider
    from alpha_engine.ai.schemas import LLMRecommendation, parse_llm_recommendation
    from alpha_engine.config.settings import Settings

    report = WorkerReport()
    started = perf_counter()
    try:
        context, pinned_prompt = pinned_inputs()
        prompt = load_recommendation_prompt()
        if (prompt != pinned_prompt or digest(prompt.encode()) != PROMPT_SHA256
                or not api_key.strip()):
            return report
        # Explicit diagnostic-only settings. Do not call get_settings/read .env.
        settings = Settings(_env_file=None, _env_prefix="MUSE_DIAGNOSTIC_UNUSED_",
                            app_env="diagnostic", prescriptive_news_enabled=False,
                            llm_provider="nvidia", nvidia_api_key=api_key,
                            nvidia_base_url="https://integrate.api.nvidia.com/v1",
                            nvidia_model=MODEL, nvidia_reasoning_effort="low", llm_temperature=0.95)

        class LimitedStream(httpx.SyncByteStream):
            def __init__(self, stream):
                self.stream = stream

            def __iter__(self):
                received = 0
                for chunk in self.stream:
                    received += len(chunk)
                    if received > MAX_BODY_BYTES:
                        raise ValueError("Response limit")
                    yield chunk

            def close(self):
                self.stream.close()

        class OneAttempt(httpx.BaseTransport):
            def __init__(self, inner):
                self.inner = inner

            def handle_request(self, request):
                if report.network_attempts != 0 or str(request.url) != URL or request.method != "POST":
                    raise ValueError("Diagnostic request rejected")
                report.network_attempts = 1
                response = self.inner.handle_request(request)
                report.http_status = response.status_code
                response.stream = LimitedStream(response.stream)
                return response

            def close(self):
                self.inner.close()

        def observe(response):
            response.read()
            try:
                envelope = strict_json(response.content)
                if not isinstance(envelope, dict):
                    raise ValueError("Envelope must be an object")
                report.envelope_json_valid = True
                model = envelope.get("model")
                report.reported_model = MODEL if model == MODEL else ("OTHER" if model is not None else None)
                report.model_matches = model == MODEL
                choices = envelope.get("choices")
                if isinstance(choices, list) and choices and isinstance(choices[0], dict):
                    reason = choices[0].get("finish_reason")
                    report.finish_reason = reason if reason in ("stop", "length") else ("OTHER" if reason is not None else None)
                usage = envelope.get("usage")
                if isinstance(usage, dict):
                    safe = {key: value for key, value in usage.items()
                            if key in TokenUsage.model_fields and type(value) is int
                            and 0 <= value <= 1_000_000_000}
                    report.usage = TokenUsage(**safe)
            except (ValueError, TypeError):
                report.envelope_json_valid = False

        report.stage = "provider"
        inner = transport if transport is not None else httpx.HTTPTransport(retries=0, trust_env=False)
        with httpx.Client(base_url="https://integrate.api.nvidia.com/v1/",
                          transport=OneAttempt(inner), trust_env=False, follow_redirects=False,
                          event_hooks={"response": [observe]}) as client:
            provider = NvidiaProvider(settings, prompt, client=client)
            provider_started = perf_counter()
            try:
                raw = provider.generate_recommendation(context, timeout_seconds=30)
            finally:
                report.provider_duration_ms = round((perf_counter() - provider_started) * 1000, 3)
                provider.close()
        report.final_content_sha256 = digest(raw.encode())
        report.stage = "parser"
        original = None
        try:
            original = strict_json(raw)
            report.final_json_valid = isinstance(original, dict)
        except (ValueError, TypeError):
            report.final_json_valid = False
        try:
            LLMRecommendation.model_validate(original)
            report.raw_schema_valid = True
        except ValueError:
            report.raw_schema_valid = False
        report.parser_executed = True
        try:
            parsed = parse_llm_recommendation(raw)
        except ValueError:
            report.parser_accepted = False
            return report
        report.parser_accepted = True
        parsed_json = parsed.model_dump(mode="json")
        report.parser_modified_content = canonical(original) != canonical(parsed_json)
        report.parsed_fields = ParsedFields(signal=parsed.signal, risk_level=parsed.risk_level,
                                            confidence=float(parsed.confidence))
        # No generated narrative is emitted or logged. These are limited checks,
        # not a claim of comprehensive semantic validation.
        texts = [parsed.summary, *parsed.key_factors, *parsed.risks, parsed.warning]
        narrative = " ".join(texts).casefold()
        original_texts = [original.get("summary", ""), *original.get("key_factors", []),
                          *original.get("risks", []), original.get("warning", "")]
        expected_signal = context["decision_policy"]["action"]
        if expected_signal in ("WAIT", "HOLD"):
            expected_signal = "HOLD"
        checks = SemanticChecks(
            signal_matches_action=parsed.signal == expected_signal,
            risk_matches_context=parsed.risk_level == context["risk"]["level"],
            historical_non_operational_context=(context["validation_context"]["operational_recommendation"] is False
                                                and context["validation_context"]["expired_at_evaluation"] is True),
            narrative_without_digits=not any(c.isdigit() for text in original_texts for c in text),
            mentions_expiry=any(word in narrative for word in ("vencid", "expirad", "no vigente", "expiraci")),
            mentions_news_absence=(any(word in narrative for word in ("noticia", "sentimiento"))
                                   and any(word in narrative for word in ("no hay", "sin noticia", "no dispon", "ausen", "falta"))),
        )
        report.semantic_checks = checks
        report.semantic_contract_pass = all(checks.model_dump().values())
        report.technical_pass = (report.http_status == 200 and report.model_matches is True
                                 and report.envelope_json_valid is True and report.finish_reason == "stop"
                                 and report.final_json_valid is True and report.parser_accepted is True)
        report.status = ("PASS" if report.technical_pass and report.semantic_contract_pass
                         and report.parser_modified_content is False else
                         "PARTIAL" if report.technical_pass else "FAILED")
        report.stage = "complete"
        return report
    except Exception:
        # Never serialize exception messages, response bodies or provider headers.
        return report
    finally:
        if report.provider_duration_ms is None and report.stage == "provider":
            report.provider_duration_ms = round((perf_counter() - started) * 1000, 3)


def main() -> None:
    logging.disable(logging.CRITICAL)
    if sys.argv[1:] != ["--execute"] or os.environ.get("MUSE_DIAGNOSTIC_CHILD") != "1":
        report = WorkerReport()
    else:
        # A second, child-owned deadline also closes sockets if the HTTP parent
        # disappears. No replacement process or retry is ever started.
        watchdog = Timer(TOTAL_TIMEOUT_SECONDS, lambda: os._exit(124))
        watchdog.daemon = True
        watchdog.start()
        try:
            report = infer_once(os.environ.get("NVIDIA_API_KEY", ""))
        finally:
            watchdog.cancel()
    sys.stdout.write(report.model_dump_json())


if __name__ == "__main__":
    main()
