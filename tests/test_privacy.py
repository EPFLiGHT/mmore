"""Tests for the privacy pipeline (package mmore.privacy).

* unit tests   - they test individual components
* E2E tests    - the whole compiled graph on CPU, with the answer model faked
                 and heavy detectors mocked so only pipeline wiring is tested
* GPU tests    - marked ``gpu`` (run with ``pytest --gpu``): these load real
                 detection and language models to check the wiring against
                 actual backends
"""

from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from mmore.privacy.config import (
    AnalyzerConfig,
    AttackVector,
    CloudLLMConfig,
    DetectionConfig,
    DetectionEngineType,
    LeakageAdversaryConfig,
    PrivacyConfig,
    SanitizationConfig,
    SanitizationStrategyType,
    VerifierCheck,
    VerifierConfig,
    as_privacy_config,
)
from mmore.privacy.detection.base import PIISpan
from mmore.privacy.schemas.policy import PrivacyPolicy
from mmore.rag.llm import LLMConfig
from mmore.utils import load_config

# ==========================================================================
# Shared fixtures and helpers
# ==========================================================================


@pytest.fixture(autouse=True)
def reset_model_cache():
    """Keep the process-wide model cache from leaking models between tests."""
    from mmore.privacy.model_cache import MODEL_REGISTRY

    MODEL_REGISTRY.clear()
    yield
    MODEL_REGISTRY.clear()


@pytest.fixture
def isolated_tool_registry():
    """Restore the global tool registry after a test that mutates it."""
    from mmore.privacy.agents.registry import tool_registry

    snapshot = dict(tool_registry)
    yield tool_registry
    tool_registry.clear()
    tool_registry.update(snapshot)


_BASE_POLICY = PrivacyPolicy(
    domain="healthcare",
    sensitive_entities=["PERSON", "MRN"],
    detection_engine="presidio",
    sanitization_strategy="token_masking",
    consistency=True,
    domain_prompt="Protect patient identity.",
)


def make_policy(**overrides: str | bool | list[str] | dict) -> PrivacyPolicy:
    """A minimal, valid policy; override any field a test cares about."""
    return replace(_BASE_POLICY, **overrides)


def span(start: int, end: int, label: str = "PERSON", score: float = 0.9) -> PIISpan:
    return PIISpan(start=start, end=end, label=label, score=score)


def fake_prediction(**fields: str | bool | float | list[str]) -> MagicMock:
    """A stand-in DSPy prediction exposing the given output fields as attributes."""
    prediction = MagicMock()
    for name, value in fields.items():
        setattr(prediction, name, value)
    return prediction


def spy_on_registered_tool(registry: dict[str, Callable], name: str) -> MagicMock:
    """Wrap a registered tool with a spy that still runs it, to assert it was used."""
    spy = MagicMock(side_effect=registry[name])
    registry[name] = spy
    return spy


# ==========================================================================
# Schemas: policy hardening and verdicts
# ==========================================================================


def test_harden_merges_new_entities_without_duplicating():
    from mmore.privacy.schemas.policy import harden

    policy = make_policy(sensitive_entities=["PERSON", "MRN"])

    hardened = harden(policy, extra_entities=["MRN", "GPS_COORDINATES"])

    assert hardened.sensitive_entities == ["PERSON", "MRN", "GPS_COORDINATES"]


def test_harden_appends_escalation_note_once():
    from mmore.privacy.schemas.policy import harden

    once = harden(make_policy(domain_prompt="Base."))
    twice = harden(once)

    assert once.domain_prompt.count("Escalation:") == 1
    assert twice.domain_prompt.count("Escalation:") == 1


def test_harden_pins_reviewer_guidance_into_prompt():
    from mmore.privacy.schemas.policy import harden

    hardened = harden(
        make_policy(), guidance="mask job titles too", guidance_label="Human guidance"
    )

    assert "Human guidance: mask job titles too" in hardened.domain_prompt


def test_harden_leaves_original_policy_untouched():
    from mmore.privacy.schemas.policy import harden

    policy = make_policy(sensitive_entities=["PERSON"])

    harden(policy, extra_entities=["EMAIL_ADDRESS"], guidance="more")

    assert policy.sensitive_entities == ["PERSON"]
    assert "Escalation:" not in policy.domain_prompt


def test_verifier_verdict_is_clean_without_warnings():
    from mmore.privacy.schemas.verification import CLEAN_VERDICT

    assert CLEAN_VERDICT.clean
    assert CLEAN_VERDICT.summary == "clean"


def test_verifier_verdict_summary_counts_warnings_by_kind():
    from mmore.privacy.schemas.verification import VerifierVerdict, VerifierWarning

    verdict = VerifierVerdict(
        warnings=[
            VerifierWarning(VerifierCheck.RESIDUAL_LEAKAGE, "PERSON", "e", 0.8),
            VerifierWarning(VerifierCheck.RESIDUAL_LEAKAGE, "MRN", "e", 0.9),
            VerifierWarning(VerifierCheck.FAITHFULNESS, "claim", "e", 0.7),
        ]
    )

    assert not verdict.clean
    assert verdict.summary == "3 warning(s) (faithfulness: 1, residual_leakage: 2)"


# ==========================================================================
# Config loading
# ==========================================================================


def test_as_privacy_config_passes_through_existing_instance():
    config = PrivacyConfig(domain="global")

    assert as_privacy_config(config) is config


def test_as_privacy_config_builds_from_mapping():
    config = as_privacy_config({"domain": "healthcare", "interactive": True})

    assert isinstance(config, PrivacyConfig)
    assert config.domain == "healthcare"
    assert config.interactive is True


def test_detection_config_casts_engine_string_to_enum():
    config = load_config({"engine": "presidio"}, DetectionConfig)

    assert config.engine is DetectionEngineType.PRESIDIO


def test_detection_config_rejects_unknown_engine():
    with pytest.raises(ValueError):
        load_config({"engine": "regex"}, DetectionConfig)


def test_privacy_config_defaults_are_self_consistent():
    config = PrivacyConfig()

    assert config.leakage_adversary.enabled is True
    assert set(config.leakage_adversary.strategies) == set(AttackVector)
    assert set(config.verifier.checks) == set(VerifierCheck)


# ==========================================================================
# Domain profiles
# ==========================================================================


def test_get_domain_profile_returns_registered_profile():
    from mmore.privacy.domains import get_domain_profile

    profile = get_domain_profile("healthcare")

    assert profile.name == "healthcare"
    assert "MRN" in profile.sensitive_entities


def test_get_domain_profile_rejects_unknown_domain():
    from mmore.privacy.domains import UnknownDomainError, get_domain_profile

    with pytest.raises(UnknownDomainError, match="finance"):
        get_domain_profile("finance")


# ==========================================================================
# Tool registry
# ==========================================================================


def test_register_tool_as_decorator_and_resolve(isolated_tool_registry):
    from mmore.privacy.agents.registry import register_tool, resolve_tool

    @register_tool("shout")
    def shout(text: str) -> str:
        return text.upper()

    assert resolve_tool("shout") is shout


def test_register_tool_by_direct_call(isolated_tool_registry):
    from mmore.privacy.agents.registry import register_tool

    register_tool("noop", lambda: None)

    assert "noop" in isolated_tool_registry


def test_resolve_unknown_tool_raises_with_available_names(isolated_tool_registry):
    from mmore.privacy.agents.registry import ToolNotRegisteredError, resolve_tool

    with pytest.raises(ToolNotRegisteredError, match="ghost"):
        resolve_tool("ghost")


def test_detection_and_sanitization_tools_self_register():
    import importlib

    from mmore.privacy.agents.registry import tool_registry

    # Importing a package runs its modules, which register their tools
    importlib.import_module("mmore.privacy.detection")
    importlib.import_module("mmore.privacy.sanitization")

    assert {"detect_pii_presidio", "detect_pii_gliner", "detect_pii_llm"} <= set(
        tool_registry
    )
    assert {"sanitize_token_masking", "sanitize_presidio"} <= set(tool_registry)


# ==========================================================================
# Model cache (shared LRU registry)
# ==========================================================================


_MB = 1024 * 1024


def counting_loader():
    """A cheap cached-model loader (Faker is a cached type) plus its call counter."""
    from faker import Faker

    calls = 0

    def load() -> Faker:
        nonlocal calls
        calls += 1
        return Faker()

    return load, lambda: calls


def test_model_cache_loads_once_and_reuses():
    from mmore.privacy.model_cache import ModelRegistry

    registry = ModelRegistry(budget_mb=0)
    load, call_count = counting_loader()

    first = registry.get_or_load("k", load)
    again = registry.get_or_load("k", load)

    assert first is again
    assert call_count() == 1


def test_model_cache_evicts_least_recently_used_over_budget():
    from faker import Faker

    from mmore.privacy import model_cache
    from mmore.privacy.model_cache import ModelRegistry

    used = 0

    def sized_loader(size: int):
        def load() -> Faker:
            nonlocal used
            used += size
            return Faker()

        return load

    with (
        patch.object(model_cache, "_device_mem_bytes", lambda: used),
        patch.object(model_cache, "_empty_device_cache", lambda: None),
    ):
        registry = ModelRegistry(budget_mb=25)
        registry.get_or_load("a", sized_loader(10 * _MB))
        registry.get_or_load("b", sized_loader(10 * _MB))
        registry.get_or_load("c", sized_loader(10 * _MB))  # 30MB > 25MB, drops "a"

        reload_a, a_calls = counting_loader()
        registry.get_or_load("a", reload_a)

    assert a_calls() == 1  # "a" was evicted and had to be rebuilt


def test_model_cache_clear_is_scoped_by_prefix():
    from mmore.privacy.model_cache import ModelRegistry

    registry = ModelRegistry(budget_mb=0)
    load_gliner, _ = counting_loader()
    load_presidio, presidio_calls = counting_loader()
    registry.get_or_load("gliner:x", load_gliner)
    registry.get_or_load("presidio:y", load_presidio)

    registry.clear(prefix="gliner:")
    registry.get_or_load("presidio:y", load_presidio)

    assert presidio_calls() == 1  # untouched by a gliner-scoped clear


def test_disabled_model_cache_reloads_every_call():
    from mmore.privacy.model_cache import ModelRegistry

    registry = ModelRegistry(enabled=False)
    load, call_count = counting_loader()

    registry.get_or_load("k", load)
    registry.get_or_load("k", load)

    assert call_count() == 2


# ==========================================================================
# DSPy backend selection and helpers
# ==========================================================================


def test_build_dspy_lm_routes_local_hf_model_to_local_backend():
    from mmore.privacy.dspy_llm import LocalHFLM, build_dspy_lm

    config = LLMConfig(llm_name="some-org/tiny-model", max_new_tokens=16)
    assert config.provider == "HF" and config.base_url is None

    assert isinstance(build_dspy_lm(config), LocalHFLM)


def test_build_dspy_lm_routes_openai_model_to_litellm():
    import dspy

    from mmore.privacy.dspy_llm import LocalHFLM, build_dspy_lm

    config = LLMConfig(llm_name="gpt-4o-mini", max_new_tokens=16)
    with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
        lm = build_dspy_lm(config)

    assert isinstance(lm, dspy.LM) and not isinstance(lm, LocalHFLM)


@pytest.mark.parametrize(
    "raw, expected",
    [(0.5, 0.5), (1.7, 1.0), (-0.3, 0.0), ("0.8", 0.8), ("bad", 0.0), (None, 0.0)],
    ids=["in-range", "above-one", "below-zero", "numeric-string", "garbage", "none"],
)
def test_clamp_confidence_coerces_into_unit_interval(raw, expected):
    from mmore.privacy.dspy_llm import clamp_confidence

    assert clamp_confidence(raw) == expected


def test_format_guidance_renders_one_bullet_per_option():
    from mmore.privacy.dspy_llm import format_guidance

    rendered = format_guidance({"presidio": "rule based", "llm": "prompted"})

    assert rendered == "- presidio: rule based\n- llm: prompted"


def test_tolerant_json_adapter_recovers_single_output_field():
    import dspy

    from mmore.privacy.dspy_llm import TolerantJSONAdapter

    # Only the field names matter; the parser reads ``list(signature.output_fields)``.
    signature = SimpleNamespace(output_fields={"additional_entities": None})

    # The strict parent parser fails; the tolerant adapter recovers the lone field
    with patch("dspy.JSONAdapter.parse", side_effect=ValueError("not an object")):
        recovered = TolerantJSONAdapter().parse(
            cast(type[dspy.Signature], signature), '```json\n["PASSPORT_NUMBER"]\n```'
        )

    assert recovered == {"additional_entities": ["PASSPORT_NUMBER"]}


# ==========================================================================
# Detection engines (model loaders mocked)
# ==========================================================================


def _presidio_result(start: int, end: int, entity_type: str, score: float) -> MagicMock:
    result = MagicMock()
    result.start, result.end = start, end
    result.entity_type, result.score = entity_type, score
    return result


def test_presidio_engine_maps_analyzer_results_to_spans():
    from mmore.privacy.detection.presidio_engine import PresidioEngine

    analyzer = MagicMock()
    analyzer.analyze.return_value = [
        _presidio_result(0, 10, "PERSON", 0.95),
        _presidio_result(11, 31, "EMAIL_ADDRESS", 0.88),
    ]
    with patch(
        "mmore.privacy.detection.presidio_engine._load_presidio_analyzer",
        return_value=analyzer,
    ):
        spans = PresidioEngine(confidence_threshold=0.5).detect("note")

    assert [(s.label, s.score) for s in spans] == [
        ("PERSON", 0.95),
        ("EMAIL_ADDRESS", 0.88),
    ]


def test_presidio_engine_forwards_threshold_and_entities():
    from mmore.privacy.detection.presidio_engine import PresidioEngine

    analyzer = MagicMock()
    analyzer.analyze.return_value = []
    with patch(
        "mmore.privacy.detection.presidio_engine._load_presidio_analyzer",
        return_value=analyzer,
    ):
        PresidioEngine(
            sensitive_entities=["PERSON", "MRN"], confidence_threshold=0.55
        ).detect("note")

    kwargs = analyzer.analyze.call_args.kwargs
    assert kwargs["score_threshold"] == 0.55
    assert kwargs["entities"] == ["PERSON", "MRN"]


def test_engine_loads_model_lazily_and_shares_it_across_instances():
    from mmore.privacy.detection.presidio_engine import PresidioEngine

    analyzer = MagicMock()
    analyzer.analyze.return_value = []
    with patch(
        "mmore.privacy.detection.presidio_engine._load_presidio_analyzer",
        return_value=analyzer,
    ) as load:
        first, second = PresidioEngine(), PresidioEngine(confidence_threshold=0.9)
        assert load.call_count == 0  # nothing loaded before first detect
        first.detect("x")
        second.detect("y")
        assert load.call_count == 1  # one shared analyzer for both instances


def test_gliner_engine_filters_labels_and_threshold_through_to_model():
    from mmore.privacy.detection.gliner_engine import GLiNEREngine

    model = MagicMock()
    model.predict_entities.return_value = []
    with patch(
        "mmore.privacy.detection.gliner_engine._load_gliner_model", return_value=model
    ):
        GLiNEREngine(sensitive_entities=["PERSON"], confidence_threshold=0.7).detect(
            "note"
        )

    kwargs = model.predict_entities.call_args.kwargs
    assert kwargs["labels"] == ["PERSON"]
    assert kwargs["threshold"] == 0.7


def test_openai_filter_engine_drops_spans_below_threshold():
    from mmore.privacy.detection.openai_filter_engine import OpenAIFilterEngine

    pipeline = MagicMock()
    pipeline.return_value = [
        {"start": 0, "end": 4, "entity_group": "PERSON", "score": 0.95},
        {"start": 5, "end": 9, "entity_group": "EMAIL", "score": 0.30},
    ]
    with patch(
        "mmore.privacy.detection.openai_filter_engine._load_openai_filter_pipeline",
        return_value=pipeline,
    ):
        spans = OpenAIFilterEngine(confidence_threshold=0.7).detect("note")

    assert [s.label for s in spans] == ["PERSON"]


@contextmanager
def stub_llm_detection(predicted_spans: list[MagicMock]) -> Generator[MagicMock]:
    """Run an ``LLMDetectionEngine`` against a canned DSPy prediction."""
    prediction = MagicMock()
    prediction.spans = predicted_spans
    predictor = MagicMock(return_value=prediction)
    with patch.multiple(
        "mmore.privacy.detection.llm_engine",
        build_dspy_lm=MagicMock(return_value=MagicMock()),
        _build_predictor=MagicMock(return_value=predictor),
    ):
        yield predictor


def _llm_span(text: str, label: str, score: float) -> MagicMock:
    predicted = MagicMock()
    predicted.text, predicted.label, predicted.score = text, label, score
    return predicted


def test_llm_engine_locates_predicted_fragments_in_source_text():
    from mmore.privacy.detection.llm_engine import LLMDetectionEngine

    note = "John Smith emailed jane@x.org."
    with stub_llm_detection(
        [_llm_span("John Smith", "PERSON", 0.95), _llm_span("jane@x.org", "EMAIL", 0.9)]
    ):
        spans = LLMDetectionEngine(
            LLMConfig(llm_name="local/tiny"), confidence_threshold=0.5
        ).detect(note)

    assert [note[s.start : s.end] for s in spans] == ["John Smith", "jane@x.org"]


def test_llm_engine_skips_fragments_absent_from_source():
    from mmore.privacy.detection.llm_engine import LLMDetectionEngine

    with stub_llm_detection(
        [_llm_span("John Smith", "PERSON", 0.95), _llm_span("Jane Doe", "PERSON", 0.95)]
    ):
        spans = LLMDetectionEngine(
            LLMConfig(llm_name="local/tiny"), confidence_threshold=0.5
        ).detect("Patient John Smith called.")

    assert len(spans) == 1
    assert spans[0].label == "PERSON"


def test_llm_engine_returns_empty_when_prediction_fails():
    from mmore.privacy.detection.llm_engine import LLMDetectionEngine

    predictor = MagicMock(side_effect=ValueError("malformed output"))
    with patch.multiple(
        "mmore.privacy.detection.llm_engine",
        build_dspy_lm=MagicMock(return_value=MagicMock()),
        _build_predictor=MagicMock(return_value=predictor),
    ):
        spans = LLMDetectionEngine(LLMConfig(llm_name="local/tiny")).detect("note")

    assert spans == []


def test_llm_engine_from_config_falls_back_to_default_llm():
    from mmore.privacy.detection.constants import DEFAULT_LLM_CONFIG
    from mmore.privacy.detection.llm_engine import LLMDetectionEngine

    engine = LLMDetectionEngine.from_config(
        DetectionConfig(engine=DetectionEngineType.LLM, llm=None)
    )

    assert engine._llm_config is DEFAULT_LLM_CONFIG


# ==========================================================================
# Detector agent internals
# ==========================================================================


def test_dedupe_spans_keeps_highest_scoring_of_identical_spans():
    from mmore.privacy.agents.detector import _dedupe_spans

    deduped = _dedupe_spans(
        [span(0, 4, "PERSON", 0.6), span(0, 4, "PERSON", 0.9), span(5, 9, "MRN", 0.7)]
    )

    assert [(s.start, s.label, s.score) for s in deduped] == [
        (0, "PERSON", 0.9),
        (5, "MRN", 0.7),
    ]


@pytest.mark.parametrize(
    "text_len, span_count, expected_level",
    [(1000, 1, "low"), (1000, 8, "medium"), (1000, 25, "high")],
    ids=["low-density", "medium-density", "high-density"],
)
def test_risk_level_scales_with_span_density(text_len, span_count, expected_level):
    from mmore.privacy.agents.detector import _assess_risk

    chunks = ["x" * text_len]
    spans_per_chunk = [[span(i, i + 1) for i in range(span_count)]]

    risk = _assess_risk(chunks, spans_per_chunk)

    assert risk.count == span_count
    assert risk.level == expected_level


def test_resolve_engine_tool_rejects_unknown_engine():
    from mmore.privacy.agents.detector import _resolve_engine_tool
    from mmore.privacy.agents.registry import ToolNotRegisteredError

    with pytest.raises(ToolNotRegisteredError, match="regex"):
        _resolve_engine_tool("regex")


# ==========================================================================
# Sanitization primitives and strategies
# ==========================================================================


def test_select_non_overlapping_prefers_higher_score_then_longer_span():
    from mmore.privacy.sanitization.base import select_non_overlapping

    kept = select_non_overlapping(
        [span(0, 5, "PERSON", 0.6), span(3, 8, "PERSON", 0.9)]  # overlap, 0.9 wins
    )

    assert [(s.start, s.end) for s in kept] == [(3, 8)]


def test_apply_replacements_rewrites_right_to_left_preserving_offsets():
    from mmore.privacy.sanitization.base import apply_replacements

    text = "A John B Mary C"
    spans = [span(2, 6, "PERSON"), span(9, 13, "PERSON")]

    result = apply_replacements(text, spans, lambda s, original: f"<{s.label}>")

    assert result == "A <PERSON> B <PERSON> C"


def test_token_masking_reuses_token_for_repeated_value_when_consistent():
    from mmore.privacy.sanitization.token_masking_strategy import TokenMaskingStrategy

    policy = make_policy(consistency=True)
    chunks = ["John met John", "John left"]
    spans_per_chunk = [
        [span(0, 4, "PERSON"), span(9, 13, "PERSON")],
        [span(0, 4, "PERSON")],
    ]

    out = TokenMaskingStrategy().apply(chunks, spans_per_chunk, policy)

    assert out == ["[PERSON_1] met [PERSON_1]", "[PERSON_1] left"]


def test_token_masking_numbers_distinct_values_when_consistent():
    from mmore.privacy.sanitization.token_masking_strategy import TokenMaskingStrategy

    policy = make_policy(consistency=True)
    out = TokenMaskingStrategy().apply(
        ["John and Mary"], [[span(0, 4, "PERSON"), span(9, 13, "PERSON")]], policy
    )

    assert out == ["[PERSON_1] and [PERSON_2]"]


def test_entity_replacement_is_consistent_per_value_across_chunks():
    from mmore.privacy.sanitization.entity_replacement_strategy import (
        EntityReplacementStrategy,
    )

    policy = make_policy(consistency=True)
    out = EntityReplacementStrategy().apply(
        ["John here", "John there"],
        [[span(0, 4, "PERSON")], [span(0, 4, "PERSON")]],
        policy,
    )

    first = out[0].removesuffix(" here")
    second = out[1].removesuffix(" there")
    assert first == second and first != "John"


def test_presidio_strategy_replaces_spans_with_entity_labels():
    from mmore.privacy.sanitization.presidio_strategy import (
        PresidioSanitizationStrategy,
    )

    policy = make_policy(sanitization_strategy="presidio")
    out = PresidioSanitizationStrategy().apply(
        ["Call John Doe now"], [[span(5, 13, "PERSON")]], policy
    )

    assert out == ["Call <PERSON> now"]


def test_presidio_strategy_rejects_unsupported_operator():
    from mmore.privacy.sanitization.presidio_strategy import (
        PresidioSanitizationStrategy,
    )

    policy = make_policy(
        sanitization_strategy="presidio", sanitization_params={"operator": "shred"}
    )

    with pytest.raises(ValueError, match="Unsupported Presidio operator"):
        PresidioSanitizationStrategy().apply(
            ["Call John Doe"], [[span(5, 13, "PERSON")]], policy
        )


def test_synthetic_rewrite_leaves_chunks_without_spans_untouched():
    from mmore.privacy.sanitization.synthetic_rewrite_strategy import (
        SyntheticRewriteStrategy,
    )

    out = SyntheticRewriteStrategy().apply(["nothing sensitive"], [[]], make_policy())

    assert out == ["nothing sensitive"]


# ==========================================================================
# Base agent
# ==========================================================================


def base_agent_config(**overrides: str | list[str] | LLMConfig | None):
    from mmore.privacy.agents.config import AgentConfig

    base = AgentConfig(
        llm=LLMConfig(llm_name="gpt2", max_new_tokens=8),
        name="agent",
        system_prompt="You are helpful.",
    )
    return replace(base, **overrides)


@contextmanager
def fake_chat_model(responses: list[str]):
    fake = FakeListChatModel(responses=responses)
    with patch(
        "mmore.privacy.agents.base._build_chat_model", return_value=fake
    ) as build:
        yield fake, build


def test_base_agent_prepends_system_prompt():
    from langchain_core.messages import HumanMessage, SystemMessage

    from mmore.privacy.agents.base import BaseAgent

    captured = {}

    class Capturing(FakeListChatModel):
        def _call(self, messages, stop=None, run_manager=None, **kwargs):
            captured["messages"] = messages
            return super()._call(messages, stop, run_manager, **kwargs)

    fake = Capturing(responses=["ok"])
    with patch("mmore.privacy.agents.base._build_chat_model", return_value=fake):
        BaseAgent.from_config(base_agent_config(system_prompt="SYS")).invoke("hi")

    sent = captured["messages"]
    assert isinstance(sent[0], SystemMessage) and sent[0].content == "SYS"
    assert isinstance(sent[1], HumanMessage) and sent[1].content == "hi"


def test_base_agent_binds_registered_tools(isolated_tool_registry):
    from mmore.privacy.agents.base import BaseAgent
    from mmore.privacy.agents.registry import register_tool

    @register_tool("greet")
    def greet(name: str) -> str:
        return f"hi {name}"

    bound: dict[str, list[Callable]] = {}

    class ToolCapturing(FakeListChatModel):
        def bind_tools(self, tools, **_kwargs):
            bound["tools"] = list(tools)
            return self

    fake = ToolCapturing(responses=["ok"])
    with patch("mmore.privacy.agents.base._build_chat_model", return_value=fake):
        BaseAgent.from_config(base_agent_config(tools=["greet"])).invoke("go")

    assert bound["tools"] == [greet]


def test_base_agent_unknown_tool_fails_fast(isolated_tool_registry):
    from mmore.privacy.agents.base import BaseAgent
    from mmore.privacy.agents.registry import ToolNotRegisteredError

    with pytest.raises(ToolNotRegisteredError):
        BaseAgent.from_config(base_agent_config(tools=["does_not_exist"]))


def test_base_agent_caches_and_shares_one_model_across_agents():
    from mmore.privacy.agents.base import BaseAgent

    with fake_chat_model(["x"] * 4) as (_, build):
        left = BaseAgent.from_config(base_agent_config(name="left"))
        right = BaseAgent.from_config(base_agent_config(name="right"))
        assert build.call_count == 0  # lazy: nothing built at construction time

        left.invoke("q")
        right.invoke("q")
        assert build.call_count == 1  # both share the same cached model


def test_clear_llm_cache_forces_a_rebuild():
    from mmore.privacy.agents.base import BaseAgent, clear_llm_cache

    with fake_chat_model(["x", "x"]) as (_, build):
        BaseAgent.from_config(base_agent_config()).invoke("q")
        clear_llm_cache()
        BaseAgent.from_config(base_agent_config()).invoke("q")

    assert build.call_count == 2


def test_llm_property_without_config_raises_clear_error():
    from mmore.privacy.agents.base import BaseAgent

    agent = BaseAgent(config=SimpleNamespace(), llm_config=None)

    with pytest.raises(ValueError, match="LLM"):
        _ = agent.llm


def test_dspy_lm_without_config_raises_when_no_fallback():
    from mmore.privacy.agents.base import BaseAgent

    agent = BaseAgent(config=SimpleNamespace(), llm_config=None)

    with pytest.raises(ValueError, match="requires an LLM"):
        _ = agent.dspy_lm


def test_dspy_lm_falls_back_to_class_default_config():
    from mmore.privacy.agents.base import BaseAgent

    class FallbackAgent(BaseAgent):
        fallback_llm_config = LLMConfig(llm_name="local/tiny")

    lm = FallbackAgent(config=SimpleNamespace(), llm_config=None).dspy_lm

    from mmore.privacy.dspy_llm import LocalHFLM

    assert isinstance(lm, LocalHFLM)


def test_memory_checkpointer_persists_state_within_a_thread():
    from langchain_core.runnables.config import RunnableConfig

    from mmore.privacy.agents.base import BaseAgent

    thread: RunnableConfig = {"configurable": {"thread_id": "t-1"}}
    with fake_chat_model(["first", "second"]):
        agent = BaseAgent.from_config(base_agent_config(checkpointer="memory"))
        agent.invoke("q1", config=thread)
        agent.invoke("q2", config=thread)

    history = [m.content for m in agent.graph.get_state(thread).values["messages"]]
    assert history == ["q1", "first", "q2", "second"]


# ==========================================================================
# Checkpointer builder
# ==========================================================================


def test_build_checkpointer_returns_none_when_unset():
    from mmore.privacy.agents.checkpointer import build_checkpointer

    assert build_checkpointer(base_agent_config(checkpointer=None)) is None


def test_build_sqlite_checkpointer_requires_a_path():
    from mmore.privacy.agents.checkpointer import build_checkpointer

    with pytest.raises(ValueError, match="checkpoint_path"):
        build_checkpointer(base_agent_config(checkpointer="sqlite"))


def test_build_sqlite_checkpointer_creates_parent_directories(tmp_path):
    from langgraph.checkpoint.sqlite import SqliteSaver

    from mmore.privacy.agents.checkpointer import build_checkpointer

    db = tmp_path / "nested" / "checkpoints.db"
    saver = build_checkpointer(
        base_agent_config(checkpointer="sqlite", checkpoint_path=str(db))
    )

    assert isinstance(saver, SqliteSaver)
    assert db.parent.is_dir()


# ==========================================================================
# Analyzer
# ==========================================================================


def make_analyzer(config: PrivacyConfig):
    from mmore.privacy.agents.analyzer import ContextPolicyAnalyzerAgent

    return ContextPolicyAnalyzerAgent.from_config(config)


def test_build_policy_honours_fully_pinned_config_without_calling_llm():
    config = PrivacyConfig(
        domain="healthcare",
        detection=DetectionConfig(
            engine=DetectionEngineType.PRESIDIO,
            confidence_threshold=0.6,
            entity_types=["PERSON", "MRN"],
        ),
        sanitization=SanitizationConfig(
            strategy=SanitizationStrategyType.TOKEN_MASKING, consistency=False
        ),
    )

    policy = make_analyzer(config).build_policy("q", ["chunk"])

    assert policy.domain == "healthcare"
    assert policy.detection_engine == "presidio"
    assert policy.detection_params["confidence_threshold"] == 0.6
    assert policy.sensitive_entities == ["PERSON", "MRN"]
    assert policy.sanitization_strategy == "token_masking"
    assert policy.consistency is False


def test_infer_domain_defaults_to_global_when_prediction_fails():
    analyzer = make_analyzer(
        PrivacyConfig(
            context_analyzer=AnalyzerConfig(llm=LLMConfig(llm_name="local/tiny"))
        )
    )

    with patch(
        "mmore.privacy.agents.analyzer._predictor",
        return_value=MagicMock(side_effect=RuntimeError("boom")),
    ):
        assert analyzer._infer_domain("q", ["c"]) == "global"


def test_infer_domain_uses_predicted_domain_when_valid():
    analyzer = make_analyzer(
        PrivacyConfig(
            context_analyzer=AnalyzerConfig(llm=LLMConfig(llm_name="local/tiny"))
        )
    )

    with patch(
        "mmore.privacy.agents.analyzer._predictor",
        return_value=MagicMock(return_value=fake_prediction(domain="healthcare")),
    ):
        assert analyzer._infer_domain("q", ["c"]) == "healthcare"


def test_clean_label_additions_normalizes_and_drops_known_labels():
    from mmore.privacy.agents.analyzer import _clean_label_additions

    cleaned = _clean_label_additions(
        ["passport number", "PERSON", "", "bank-account", "PERSON"],
        current=["PERSON"],
    )

    assert cleaned == ["PASSPORTNUMBER", "BANKACCOUNT"]


def test_parse_params_maps_threshold_level_to_confidence():
    from mmore.privacy.agents.analyzer import _parse_params

    params = _parse_params("presidio", fake_prediction(threshold_level="high"))

    assert params == {"confidence_threshold": 0.85}


def test_parse_params_requires_multi_label_flag_for_gliner():
    from mmore.privacy.agents.analyzer import _parse_params

    assert _parse_params("gliner", fake_prediction(threshold_level="low")) is None
    assert _parse_params(
        "gliner", fake_prediction(threshold_level="low", multi_label=True)
    ) == {"confidence_threshold": 0.5, "multi_label": True}


def test_detection_unchanged_detects_sanitization_only_escalation():
    from mmore.privacy.agents.analyzer import _detection_unchanged

    before = make_policy()
    same_detection = make_policy(sanitization_strategy="synthetic_rewrite")
    changed_detection = make_policy(detection_engine="gliner")

    assert _detection_unchanged(before, same_detection) is True
    assert _detection_unchanged(before, changed_detection) is False


def test_operator_params_masks_and_warns_on_missing_encryption_key(caplog):
    analyzer = make_analyzer(PrivacyConfig())

    assert analyzer._operator_params("mask")["masking_char"] == "*"
    with caplog.at_level("WARNING"):
        params = analyzer._operator_params("encrypt")
    assert "key" in params and "ephemeral key" in caplog.text


def test_analyzer_escalates_from_adversary_leak_note_only_without_llm():
    from mmore.privacy.agents.state import PrivacyState
    from mmore.privacy.schemas.leakage import LeakageVerdict
    from mmore.privacy.schemas.report import PreCloudOutcome

    analyzer = make_analyzer(PrivacyConfig())  # no analyzer LLM -> note-only hardening
    policy = make_policy()
    verdict = LeakageVerdict(
        leaked=True,
        vector=AttackVector.RESIDUAL_SPAN,
        entity_type="PERSON",
        evidence="name in the clear",
        confidence=0.8,
        recommendation="tighten masking",
    )
    state = PrivacyState(policy=policy, verdict=verdict)

    update = analyzer._escalate(state, policy)

    assert update.get("outcome") == PreCloudOutcome.RE_LOOPED
    assert update.get("adversary_escalations") == 1
    assert len(update.get("escalation_log") or []) == 1
    hardened = update.get("policy")
    assert hardened is not None and "Escalation:" in hardened.domain_prompt


def test_analyzer_escalates_from_human_revision_feedback():
    from mmore.privacy.agents.state import PrivacyState

    analyzer = make_analyzer(PrivacyConfig())
    policy = make_policy()
    state = PrivacyState(
        policy=policy,
        revision_requested=True,
        human_feedback="also mask locations",
    )

    update = analyzer._escalate(state, policy)

    log = update.get("escalation_log") or []
    assert log[0].from_human_feedback is True
    hardened = update.get("policy")
    assert hardened is not None and "also mask locations" in hardened.domain_prompt


# ==========================================================================
# Leakage adversary
# ==========================================================================


def make_adversary(config: PrivacyConfig):
    from mmore.privacy.agents.adversary import AdversarialAgent

    return AdversarialAgent.from_config(config)


def test_adversary_disabled_reports_safe_without_probing():
    from mmore.privacy.agents.state import PrivacyState

    adversary = make_adversary(
        PrivacyConfig(leakage_adversary=LeakageAdversaryConfig(enabled=False))
    )

    update = adversary._node(PrivacyState(policy=make_policy()))

    assert update.get("safe") is True


def test_adversary_probe_keeps_strongest_signal_and_flags_leak():
    from mmore.privacy.schemas.leakage import LeakageVerdict

    adversary = make_adversary(
        PrivacyConfig(
            context_analyzer=AnalyzerConfig(llm=LLMConfig(llm_name="local/tiny")),
            leakage_adversary=LeakageAdversaryConfig(
                strategies=[AttackVector.RESIDUAL_SPAN, AttackVector.QUASI_IDENTIFIER],
                leakage_threshold=0.5,
            ),
        )
    )
    weak = LeakageVerdict(False, AttackVector.RESIDUAL_SPAN, None, "", 0.2)
    strong = LeakageVerdict(True, AttackVector.QUASI_IDENTIFIER, "PERSON", "e", 0.8)

    with patch.object(adversary, "_probe_vector", side_effect=[weak, strong]):
        verdict = adversary.probe(make_policy(), ["sanitized context"])

    assert verdict is strong
    assert verdict.leaked is True


def test_adversary_probe_returns_safe_for_empty_context():
    from mmore.privacy.schemas.leakage import SAFE_VERDICT

    adversary = make_adversary(
        PrivacyConfig(
            context_analyzer=AnalyzerConfig(llm=LLMConfig(llm_name="local/tiny"))
        )
    )

    assert adversary.probe(make_policy(), ["", "  "]) is SAFE_VERDICT


def test_adversary_lists_config_pinned_fields_for_the_report():
    adversary = make_adversary(
        PrivacyConfig(
            detection=DetectionConfig(engine=DetectionEngineType.PRESIDIO),
            sanitization=SanitizationConfig(
                strategy=SanitizationStrategyType.TOKEN_MASKING
            ),
        )
    )

    pinned = adversary._pinned_policy_fields()

    assert "detection engine (presidio)" in pinned
    assert "sanitization strategy (token_masking)" in pinned


# ==========================================================================
# Advisory verifier
# ==========================================================================


def make_verifier(config: PrivacyConfig):
    from mmore.privacy.agents.verifier import AdvisoryVerifierAgent

    return AdvisoryVerifierAgent.from_config(config)


def test_verifier_returns_clean_when_no_checks_configured():
    verifier = make_verifier(PrivacyConfig(verifier=VerifierConfig(checks=[])))

    verdict = verifier.verify("answer", ["sanitized"], ["raw"])

    assert verdict.clean


def test_verifier_raises_warning_above_threshold():
    verifier = make_verifier(
        PrivacyConfig(
            verifier=VerifierConfig(
                checks=[VerifierCheck.RESIDUAL_LEAKAGE], warn_threshold=0.5
            )
        )
    )
    leaked = fake_prediction(
        leaked=True, entity_type="PERSON", evidence="name returned", confidence=0.9
    )

    with patch.object(verifier, "_predict", return_value=leaked):
        verdict = verifier.verify("the patient is John", ["ctx"], ["raw"])

    assert not verdict.clean
    assert verdict.warnings[0].kind is VerifierCheck.RESIDUAL_LEAKAGE
    assert verdict.checks_run == ["residual_leakage"]


def test_verifier_suppresses_low_confidence_finding():
    verifier = make_verifier(
        PrivacyConfig(
            verifier=VerifierConfig(
                checks=[VerifierCheck.RESIDUAL_LEAKAGE], warn_threshold=0.5
            )
        )
    )
    faint = fake_prediction(
        leaked=True, entity_type="PERSON", evidence="", confidence=0.2
    )

    with patch.object(verifier, "_predict", return_value=faint):
        verdict = verifier.verify("answer", ["ctx"], ["raw"])

    assert verdict.clean
    assert verdict.checks_run == ["residual_leakage"]


def test_verifier_records_a_check_that_raises_as_failed():
    verifier = make_verifier(
        PrivacyConfig(verifier=VerifierConfig(checks=[VerifierCheck.FAITHFULNESS]))
    )

    with patch.object(verifier, "_predict", side_effect=RuntimeError("model down")):
        verdict = verifier.verify("answer", ["ctx"], ["raw"])

    assert verdict.checks_failed == ["faithfulness"]
    assert verdict.clean  # a failed check raises no warning


# ==========================================================================
# HITL gate logic
# ==========================================================================


@pytest.mark.parametrize(
    "resume, expected",
    [
        (1, "APPROVE"),
        ("2", "RETRY"),
        ("reject", "REJECT"),
        ({"choice": 1}, "APPROVE"),
        ({"action": "retry"}, "RETRY"),
        (9, None),
        ("nonsense", None),
    ],
    ids=[
        "menu-number",
        "numeric-string",
        "action-name",
        "choice-dict",
        "action-dict",
        "out-of-range",
        "unrecognized",
    ],
)
def test_gate_interprets_resume_values(resume, expected):
    from mmore.privacy.agents.gate import GateDecision, _interpret_decision

    decision = _interpret_decision(resume)

    assert decision == (GateDecision[expected] if expected else None)


def test_gate_extracts_feedback_only_from_structured_resume():
    from mmore.privacy.agents.gate import _extract_feedback

    assert _extract_feedback({"choice": 2, "feedback": "mask titles"}) == "mask titles"
    assert _extract_feedback({"choice": 2, "feedback": "  "}) is None
    assert _extract_feedback("2") is None


def test_gate_auto_approves_when_not_interactive():
    from mmore.privacy.agents.gate import HITLGateAgent
    from mmore.privacy.agents.state import PrivacyState
    from mmore.privacy.schemas.report import PreCloudOutcome

    gate = HITLGateAgent(PrivacyConfig(interactive=False))

    update = gate._node(PrivacyState(policy=make_policy(), risk=None))

    assert update.get("approved") is True
    assert update.get("outcome") == PreCloudOutcome.APPROVED


def test_gate_summary_reports_detected_entities():
    from mmore.privacy.agents.gate import build_gate_summary
    from mmore.privacy.agents.state import PrivacyState
    from mmore.privacy.schemas.risk import RiskAssessment

    state = PrivacyState(
        policy=make_policy(),
        risk=RiskAssessment(count=3, entity_counts={"PERSON": 2, "MRN": 1}),
    )

    summary = build_gate_summary(state, max_iterations=3)

    assert "PERSON: 2" in summary and "MRN: 1" in summary
    assert "Total sensitive spans: 3" in summary


# ==========================================================================
# Answer model
# ==========================================================================


def test_answer_model_identity_reports_backend_and_model():
    from mmore.privacy.agents.answer import AnswerAgent

    config = PrivacyConfig(answer=CloudLLMConfig(llm=LLMConfig(llm_name="gpt-4o-mini")))
    backend, model = AnswerAgent.from_config(config).identity

    assert backend == "OPENAI"
    assert model == "gpt-4o-mini"


def test_answer_model_requires_answer_llm_in_config():
    from mmore.privacy.agents.answer import AnswerAgent

    with pytest.raises(ValueError, match="answer.llm"):
        AnswerAgent.from_config(PrivacyConfig())


def test_answer_model_builds_prompt_from_sanitized_context_only():
    from langchain_core.messages import BaseMessage, SystemMessage

    from mmore.privacy.agents.answer import AnswerAgent

    sent: dict[str, list[BaseMessage]] = {}

    class MessageCapturing(FakeListChatModel):
        def _call(self, messages, stop=None, run_manager=None, **kwargs):
            sent["messages"] = messages
            return super()._call(messages, stop, run_manager, **kwargs)

    config = PrivacyConfig(answer=CloudLLMConfig(llm=LLMConfig(llm_name="gpt-4o-mini")))
    fake = MessageCapturing(responses=["a clean summary"])
    with patch("mmore.privacy.agents.base._build_chat_model", return_value=fake):
        answer = AnswerAgent.from_config(config).answer(
            "Who is the patient?", ["[PERSON_1] was admitted."], "Be concise."
        )

    assert answer == "a clean summary"
    system = next(m for m in sent["messages"] if isinstance(m, SystemMessage))
    assert "Be concise." in system.content
    human = sent["messages"][-1].content
    assert "[PERSON_1]" in human and "John" not in human


# ==========================================================================
# Report builder
# ==========================================================================


def test_report_builder_groups_residual_leakage_by_entity_type():
    from mmore.privacy.report_builder import _warning_summaries
    from mmore.privacy.schemas.verification import VerifierVerdict, VerifierWarning

    verdict = VerifierVerdict(
        warnings=[
            VerifierWarning(VerifierCheck.RESIDUAL_LEAKAGE, "PERSON", "e", 0.6),
            VerifierWarning(VerifierCheck.RESIDUAL_LEAKAGE, "MRN", "e", 0.9),
            VerifierWarning(VerifierCheck.RESIDUAL_LEAKAGE, "PERSON", "e", 0.7),
            VerifierWarning(VerifierCheck.FAITHFULNESS, "claim", "e", 0.8),
        ]
    )

    summaries = {(s.kind, s.entity_type): s for s in _warning_summaries(verdict)}

    person = summaries[(VerifierCheck.RESIDUAL_LEAKAGE, "PERSON")]
    assert person.count == 2 and person.confidence == 0.7
    assert summaries[(VerifierCheck.RESIDUAL_LEAKAGE, "MRN")].count == 1
    assert summaries[(VerifierCheck.FAITHFULNESS, None)].count == 1


def test_report_builder_requires_policy_and_outcome():
    from mmore.privacy.agents.state import PrivacyState
    from mmore.privacy.report_builder import build_report_record

    with pytest.raises(ValueError, match="policy"):
        build_report_record(PrivacyState())


def test_report_builder_splits_adversary_and_human_iterations():
    from mmore.privacy.agents.state import PrivacyState
    from mmore.privacy.report_builder import build_report_record
    from mmore.privacy.schemas.report import PreCloudOutcome

    record = build_report_record(
        PrivacyState(
            policy=make_policy(),
            outcome=PreCloudOutcome.APPROVED,
            total_escalations=3,
            adversary_escalations=2,
            answer="done",
        )
    )

    assert record.adversary_iterations == 2
    assert record.human_iterations == 1


# ==========================================================================
# UX stage naming
# ==========================================================================


@pytest.mark.parametrize(
    "raw, pretty",
    [
        ("gliner", "GLiNER"),
        ("token_masking", "Token masking"),
        ("presidio", "Presidio"),
    ],
    ids=["special-cased", "underscored", "capitalized"],
)
def test_tool_name_prettifies_engine_and_strategy_ids(raw, pretty):
    from mmore.privacy.ux import tool_name

    assert tool_name(raw) == pretty


# ==========================================================================
# End-to-end pipeline (CPU)
# ==========================================================================


class FakeAnalyzer:
    """Presidio-shaped analyzer that flags known substrings, for E2E stubbing."""

    def __init__(self, flagged: list[tuple[str, str, float]]):
        self._flagged = flagged

    def analyze(self, text, language, entities, score_threshold):
        results = []
        for substring, label, score in self._flagged:
            index = text.find(substring)
            if index < 0 or score < score_threshold:
                continue
            if entities and label not in entities:
                continue
            results.append(
                _presidio_result(index, index + len(substring), label, score)
            )
        return results


class EchoAnswerModel(FakeListChatModel):
    """Answer model that returns the context it was handed."""

    def _call(self, messages, stop=None, run_manager=None, **kwargs):
        return messages[-1].content  # the "Context: ...\n\nQuestion: ..." message


@contextmanager
def running_pipeline(
    config: PrivacyConfig,
    flagged,
    answer: str = "A safe summary.",
    echo_answer: bool = False,
):
    """Compile and run the real graph with a mocked detector and answer model."""
    from mmore.privacy.runner import run_privacy_query, setup_privacy

    model = (
        EchoAnswerModel(responses=[answer])
        if echo_answer
        else FakeListChatModel(responses=[answer])
    )
    with (
        patch(
            "mmore.privacy.detection.presidio_engine._load_presidio_analyzer",
            return_value=FakeAnalyzer(flagged),
        ),
        patch("mmore.privacy.agents.base._build_chat_model", return_value=model),
    ):
        graph, approver, _ = setup_privacy(config, interactive_ok=False)

        def run(query: str, chunks: list[str]):
            return run_privacy_query(graph, query, chunks, approver=approver)

        yield run


def healthcare_presidio_config(
    **overrides: LeakageAdversaryConfig | VerifierConfig,
) -> PrivacyConfig:
    base = PrivacyConfig(
        domain="healthcare",
        interactive=False,
        detection=DetectionConfig(
            engine=DetectionEngineType.PRESIDIO,
            confidence_threshold=0.5,
            entity_types=["PERSON", "MRN"],
        ),
        sanitization=SanitizationConfig(
            strategy=SanitizationStrategyType.TOKEN_MASKING, consistency=True
        ),
        leakage_adversary=LeakageAdversaryConfig(enabled=False),
        verifier=VerifierConfig(checks=[]),
        answer=CloudLLMConfig(llm=LLMConfig(llm_name="gpt-4o-mini")),
    )
    return replace(base, **overrides)


def test_pipeline_sanitizes_context_and_returns_answer_with_report():
    flagged = [("John Doe", "PERSON", 0.99), ("87654321", "MRN", 0.95)]
    with running_pipeline(healthcare_presidio_config(), flagged) as run:
        result = run("Summarize the visit", ["John Doe (MRN 87654321) was admitted."])

    assert result.answer == "A safe summary."
    assert "[PERSON_1]" in result.sanitized_chunks[0]
    assert "John Doe" not in result.sanitized_chunks[0]

    record = result.record
    assert record is not None
    assert record.detection_engine is DetectionEngineType.PRESIDIO
    assert record.detection.count == 2
    assert record.detection.entity_counts == {"PERSON": 1, "MRN": 1}


def test_pipeline_feeds_only_sanitized_context_to_answer_model():
    flagged = [("John Doe", "PERSON", 0.99), ("87654321", "MRN", 0.95)]
    with running_pipeline(
        healthcare_presidio_config(), flagged, echo_answer=True
    ) as run:
        result = run("Summarize the visit", ["John Doe (MRN 87654321) was admitted."])

    assert "John Doe" not in result.answer
    assert "87654321" not in result.answer
    assert "[PERSON_1]" in result.answer  # it did receive the sanitized context


def test_pipeline_escalation_loop_retries_then_clears_and_answers(
    isolated_tool_registry,
):
    from mmore.privacy.agents.adversary import AdversarialAgent
    from mmore.privacy.schemas.leakage import SAFE_VERDICT, LeakageVerdict

    config = healthcare_presidio_config(
        leakage_adversary=LeakageAdversaryConfig(enabled=True, max_iterations=2)
    )
    leak = LeakageVerdict(
        True, AttackVector.RESIDUAL_SPAN, "PERSON", "leak", 0.9, "retry"
    )
    flagged = [("John Doe", "PERSON", 0.99)]
    sanitize_spy = spy_on_registered_tool(
        isolated_tool_registry, "sanitize_token_masking"
    )

    with (
        patch.object(AdversarialAgent, "probe", side_effect=[leak, SAFE_VERDICT]),
        patch.object(AdversarialAgent, "recommend", return_value="tighten masking"),
    ):
        with running_pipeline(config, flagged) as run:
            result = run("Summarize", ["John Doe was here."])

    assert result.answer == "A safe summary."
    assert result.record is not None
    assert result.record.adversary_iterations == 1

    # Check the loop was relevant: a sanitization pass ran under the hardened policy
    policies = [call.args[2] for call in sanitize_spy.call_args_list]
    assert any("Escalation:" in policy.domain_prompt for policy in policies)


def test_pipeline_aborts_as_unsafe_when_escalations_are_exhausted():
    from mmore.privacy.agents.adversary import AdversarialAgent
    from mmore.privacy.schemas.leakage import LeakageVerdict
    from mmore.privacy.schemas.report import PreCloudOutcome, ReportOutcome

    config = healthcare_presidio_config(
        leakage_adversary=LeakageAdversaryConfig(
            enabled=True, max_iterations=1, abort_on_exhaustion=True
        )
    )
    always_leaks = LeakageVerdict(True, AttackVector.RESIDUAL_SPAN, "PERSON", "x", 0.9)

    with (
        patch.object(AdversarialAgent, "probe", return_value=always_leaks),
        patch.object(AdversarialAgent, "recommend", return_value=None),
    ):
        with running_pipeline(config, [("John Doe", "PERSON", 0.99)]) as run:
            result = run("Summarize", ["John Doe was here."])

    assert result.answer == ""
    assert result.outcome == PreCloudOutcome.ABORTED
    assert result.record is not None
    assert result.record.outcome == ReportOutcome.ABORTED_UNSAFE


def test_validate_privacy_config_requires_an_answer_model():
    from mmore.privacy.runner import validate_privacy_config

    with pytest.raises(ValueError, match="Answer model"):
        validate_privacy_config(PrivacyConfig())


# ==========================================================================
# Interactive gate: resume loop and terminal approver
# ==========================================================================


def interactive_gate_graph():
    from langgraph.checkpoint.memory import MemorySaver

    from mmore.privacy.agents.gate import HITLGateAgent

    return HITLGateAgent(
        PrivacyConfig(interactive=True), checkpointer=MemorySaver()
    ).graph


# A gate resume value: a menu number, an action name, or a {choice, feedback} map
GateResume = int | str | dict[str, str | int]


class ScriptedApprover:
    """Replays canned gate answers and records every payload it was shown."""

    def __init__(self, answers: list[GateResume]):
        self._answers = iter(answers)
        self.payloads: list[dict] = []

    def __call__(self, payload: dict) -> GateResume:
        self.payloads.append(payload)
        return next(self._answers)


def test_gate_resume_approves_and_records_the_decision():
    from mmore.privacy.runner import run_privacy_query
    from mmore.privacy.schemas.report import PreCloudOutcome

    approver = ScriptedApprover(["1"])
    result = run_privacy_query(
        interactive_gate_graph(), "q", ["chunk"], approver=approver
    )

    assert result.outcome == PreCloudOutcome.APPROVED
    assert approver.payloads[0]["options"][0]["choice"] == 1


def test_gate_resume_reprompts_after_an_invalid_choice():
    from mmore.privacy.runner import run_privacy_query
    from mmore.privacy.schemas.report import PreCloudOutcome

    approver = ScriptedApprover(["9", "approve"])
    result = run_privacy_query(
        interactive_gate_graph(), "q", ["chunk"], approver=approver
    )

    assert result.outcome == PreCloudOutcome.APPROVED
    assert "error" in approver.payloads[1]


def test_gate_resume_rejects_and_aborts():
    from mmore.privacy.runner import run_privacy_query
    from mmore.privacy.schemas.report import PreCloudOutcome

    result = run_privacy_query(
        interactive_gate_graph(), "q", ["chunk"], approver=ScriptedApprover(["3"])
    )

    assert result.outcome == PreCloudOutcome.REJECTED


def test_gate_without_approver_raises_on_interrupt():
    from mmore.privacy.runner import run_privacy_query

    with pytest.raises(RuntimeError, match="interactive"):
        run_privacy_query(interactive_gate_graph(), "q", ["chunk"])


GATE_PAYLOAD = {
    "summary": "Pre-cloud privacy review\n- Domain: healthcare",
    "options": [
        {"choice": 1, "action": "approve", "label": "Approve: clear the context"},
        {"choice": 2, "action": "retry", "label": "Revise: tighten and retry"},
        {"choice": 3, "action": "reject", "label": "Reject: abort the request"},
    ],
}


def test_terminal_approver_prints_review_and_returns_choice(monkeypatch, capsys):
    from mmore.privacy.gate_ui import terminal_approver

    monkeypatch.setattr("builtins.input", lambda _prompt: "1")

    assert terminal_approver(GATE_PAYLOAD) == "1"
    printed = capsys.readouterr().out
    assert "Pre-cloud privacy review" in printed and "[1]" in printed


def test_terminal_approver_collects_optional_revise_feedback(monkeypatch):
    from mmore.privacy.gate_ui import terminal_approver

    answers = iter(["2", "also mask job titles"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))

    assert terminal_approver(GATE_PAYLOAD) == {
        "choice": "2",
        "feedback": "also mask job titles",
    }


def test_terminal_approver_raises_when_stdin_is_closed(monkeypatch):
    from mmore.privacy.gate_ui import terminal_approver

    def closed(_prompt):
        raise EOFError

    monkeypatch.setattr("builtins.input", closed)

    with pytest.raises(RuntimeError, match="interactive: false"):
        terminal_approver(GATE_PAYLOAD)


def test_render_chunk_diff_colors_removed_and_replaced_words():
    from mmore.privacy.gate_ui import _render_chunk_diff
    from mmore.ux import str_brand

    rendered = _render_chunk_diff("Call John Doe now", "Call [PERSON] now")

    assert "\033[31mJohn Doe\033[0m" in rendered  # removed PII in red
    assert str_brand("[PERSON]") in rendered  # replacement in the mmore color


def test_render_chunk_diff_is_plain_when_nothing_changed():
    from mmore.privacy.gate_ui import _render_chunk_diff

    assert _render_chunk_diff("no pii here", "no pii here") == "no pii here"


# ==========================================================================
# GPU tests that load real models
# ==========================================================================

_CLINICAL_NOTE = "Jane Roe called the clinic from 555-987-6543 on Monday."


@pytest.mark.gpu
def test_gpu_presidio_detects_and_locates_real_pii():
    from mmore.privacy.detection.presidio_engine import PresidioEngine

    try:
        spans = PresidioEngine(
            sensitive_entities=["PERSON", "PHONE_NUMBER"], confidence_threshold=0.4
        ).detect(_CLINICAL_NOTE)
    except Exception as error:
        pytest.skip(f"Presidio model unavailable: {error}")

    assert {s.label for s in spans} >= {"PERSON"}
    # Each span's offsets map to real text in the note
    assert all(_CLINICAL_NOTE[s.start : s.end] for s in spans)
    person = next(s for s in spans if s.label == "PERSON")
    assert "Roe" in _CLINICAL_NOTE[person.start : person.end]


@pytest.mark.gpu
def test_gpu_gliner_detects_and_locates_real_pii():
    from mmore.privacy.detection.gliner_engine import GLiNEREngine

    note = "Patient John Smith was admitted on Monday."
    try:
        spans = GLiNEREngine(
            sensitive_entities=["PERSON"], confidence_threshold=0.3
        ).detect(note)
    except Exception as error:
        pytest.skip(f"GLiNER model unavailable: {error}")

    person = next(s for s in spans if s.label == "PERSON")
    assert note[person.start : person.end] == "John Smith"
    assert 0.0 <= person.score <= 1.0


def _skip_unless_local_llm(llm_name: str) -> None:
    """Load the small local LLM once (skip if missing) so the run can reuse it."""
    from mmore.privacy.dspy_llm import get_local_hf_pipeline

    try:
        get_local_hf_pipeline(llm_name)  # cached: agents and answer model reuse it
    except Exception as error:
        pytest.skip(f"local LLM unavailable: {error}")


def _skip_unless_presidio() -> None:
    """Load the real Presidio analyzer once, skipping the test if it is missing."""
    from mmore.privacy.detection.presidio_engine import _load_presidio_analyzer

    try:
        _load_presidio_analyzer()
    except Exception as error:
        pytest.skip(f"Presidio model unavailable: {error}")


@pytest.mark.gpu
def test_gpu_full_pipeline_routes_tools_and_produces_a_clean_report(
    isolated_tool_registry,
):
    """Whole pipeline on real models: verify the routed tools, then the report."""
    from mmore.privacy.runner import run_privacy_query, setup_privacy

    small_llm = LLMConfig(llm_name="Qwen/Qwen2.5-0.5B-Instruct", max_new_tokens=64)
    config = PrivacyConfig(
        domain="healthcare",
        interactive=False,
        detection=DetectionConfig(
            engine=DetectionEngineType.PRESIDIO,
            confidence_threshold=0.4,
            entity_types=["PERSON", "PHONE_NUMBER"],
        ),
        sanitization=SanitizationConfig(
            strategy=SanitizationStrategyType.TOKEN_MASKING, consistency=True
        ),
        leakage_adversary=LeakageAdversaryConfig(
            enabled=True,
            max_iterations=1,
            abort_on_exhaustion=False,  # always reach the answer model
            strategies=[AttackVector.RESIDUAL_SPAN],
        ),
        verifier=VerifierConfig(checks=[VerifierCheck.RESIDUAL_LEAKAGE]),
        answer=CloudLLMConfig(llm=small_llm),
        context_analyzer=AnalyzerConfig(llm=small_llm),
    )

    _skip_unless_presidio()
    _skip_unless_local_llm(small_llm.llm_name)

    graph, _, _ = setup_privacy(config, interactive_ok=False)
    detect_spy = spy_on_registered_tool(isolated_tool_registry, "detect_pii_presidio")
    sanitize_spy = spy_on_registered_tool(
        isolated_tool_registry, "sanitize_token_masking"
    )
    result = run_privacy_query(graph, "Summarize the note.", [_CLINICAL_NOTE])

    # The policy routed detection and sanitization to the configured tools
    assert detect_spy.called and sanitize_spy.called
    routed_policy = detect_spy.call_args.args[1]
    assert routed_policy.detection_engine == "presidio"
    assert sanitize_spy.call_args.args[2].sanitization_strategy == "token_masking"

    # Content: the real identifiers are gone and a typed mask took their place
    sanitized = result.sanitized_chunks[0]
    assert "Jane Roe" not in sanitized and "555-987-6543" not in sanitized
    assert "[PERSON_1]" in sanitized
    assert isinstance(result.answer, str) and result.answer.strip()

    # Structure: a fully populated, PII-free report record
    record = result.record
    assert record is not None
    assert record.request_id and record.timestamp
    assert record.detection_engine is DetectionEngineType.PRESIDIO
    assert record.sanitization_strategy is SanitizationStrategyType.TOKEN_MASKING
    assert record.detection.count >= 2  # PERSON + PHONE_NUMBER
    assert {"Detector", "Sanitizer", "Answer"} <= set(record.stage_seconds)
    assert VerifierCheck.RESIDUAL_LEAKAGE.value in (
        record.verifier_checks_run + record.verifier_checks_failed
    )


@pytest.mark.gpu
def test_gpu_pipeline_uses_slm_for_detection_and_sanitization(isolated_tool_registry):
    from mmore.privacy.detection.llm_engine import LLMDetectionEngine
    from mmore.privacy.runner import run_privacy_query, setup_privacy

    small_llm = LLMConfig(llm_name="Qwen/Qwen2.5-0.5B-Instruct", max_new_tokens=64)
    config = PrivacyConfig(
        domain="healthcare",
        interactive=False,
        detection=DetectionConfig(
            engine=DetectionEngineType.LLM,
            confidence_threshold=0.3,
            entity_types=["PERSON", "PHONE_NUMBER"],
            llm=small_llm,
        ),
        sanitization=SanitizationConfig(
            strategy=SanitizationStrategyType.SYNTHETIC_REWRITE, llm=small_llm
        ),
        leakage_adversary=LeakageAdversaryConfig(enabled=False),
        verifier=VerifierConfig(checks=[]),
        answer=CloudLLMConfig(llm=small_llm),
        context_analyzer=AnalyzerConfig(llm=small_llm),
    )

    scanned: list[str] = []
    real_detect = LLMDetectionEngine.detect

    def spy_detect(self, text):
        scanned.append(text)
        return real_detect(self, text)

    _skip_unless_local_llm(small_llm.llm_name)

    graph, _, _ = setup_privacy(config, interactive_ok=False)
    rewrite_spy = spy_on_registered_tool(
        isolated_tool_registry, "sanitize_synthetic_rewrite"
    )
    with patch.object(LLMDetectionEngine, "detect", spy_detect):
        result = run_privacy_query(graph, "Summarize the note.", [_CLINICAL_NOTE])

    # Check both stages ran on the SLM
    assert scanned
    assert rewrite_spy.called
    assert rewrite_spy.call_args.args[2].sanitization_strategy == "synthetic_rewrite"

    record = result.record
    assert record is not None
    assert record.detection_engine is DetectionEngineType.LLM
    assert record.sanitization_strategy is SanitizationStrategyType.SYNTHETIC_REWRITE
    assert {"Detector", "Sanitizer", "Answer"} <= set(record.stage_seconds)

    # A sanitized context and an answer come out (we dont check quality here)
    assert isinstance(result.sanitized_chunks[0], str) and result.sanitized_chunks[0]
    assert isinstance(result.answer, str) and result.answer.strip()


@pytest.mark.gpu
def test_gpu_hitl_feedback_reenters_loop():
    from mmore.privacy.agents.analyzer import ContextPolicyAnalyzerAgent
    from mmore.privacy.runner import run_privacy_query, setup_privacy
    from mmore.privacy.schemas.report import HITLDecision, PreCloudOutcome

    small_llm = LLMConfig(llm_name="Qwen/Qwen2.5-0.5B-Instruct", max_new_tokens=64)
    config = PrivacyConfig(
        domain="healthcare",
        interactive=True,
        detection=DetectionConfig(
            engine=DetectionEngineType.PRESIDIO,
            confidence_threshold=0.4,
            entity_types=["PERSON", "PHONE_NUMBER"],
        ),
        sanitization=SanitizationConfig(
            strategy=SanitizationStrategyType.TOKEN_MASKING, consistency=True
        ),
        leakage_adversary=LeakageAdversaryConfig(enabled=False),
        verifier=VerifierConfig(checks=[]),
        answer=CloudLLMConfig(llm=small_llm),
        context_analyzer=AnalyzerConfig(llm=small_llm),
    )

    # The human revises with feedback the first time, then approves the second time
    feedback = "also treat home addresses as sensitive"
    approver = ScriptedApprover([{"choice": 2, "feedback": feedback}, "1"])

    _skip_unless_presidio()
    _skip_unless_local_llm(small_llm.llm_name)

    # Capture the policy the analyzer emits on each pass to prove the re-loop
    emitted_policies = []
    real_node = ContextPolicyAnalyzerAgent._node

    def spy_node(self, state):
        update = real_node(self, state)
        policy = update.get("policy")
        if policy is not None:
            emitted_policies.append(policy)
        return update

    with patch.object(ContextPolicyAnalyzerAgent, "_node", spy_node):
        graph, _, _ = setup_privacy(config, interactive_ok=True)
        result = run_privacy_query(
            graph, "Summarize the note.", [_CLINICAL_NOTE], approver=approver
        )

    # The gate feedback sent the request back through the analyzer
    assert len(emitted_policies) == 2
    assert feedback in emitted_policies[1].domain_prompt

    # The run recorded revise-then-approve and produced an answer
    record = result.record
    assert record is not None
    assert record.human_iterations == 1
    assert [event.decision for event in record.hitl_events] == [
        HITLDecision.RETRY,
        HITLDecision.APPROVE,
    ]
    assert record.hitl_events[0].human_feedback == feedback
    assert result.outcome == PreCloudOutcome.APPROVED
    assert isinstance(result.answer, str) and result.answer.strip()
