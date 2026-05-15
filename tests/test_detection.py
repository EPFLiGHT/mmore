"""Tests for mmore.privacy.detection. Will be later replaced by E2E tests
once the complete privacy multi-agent system is implemented."""

from unittest.mock import MagicMock, patch

import pytest

from mmore.privacy.agents.registry import tool_registry
from mmore.privacy.detection.config import DetectionConfig
from mmore.privacy.detection.gliner_engine import (
    GLiNEREngine,
    clear_gliner_cache,
    detect_pii_gliner,
)
from mmore.privacy.detection.llm_engine import (
    LLMDetectionEngine,
    clear_llm_engine_cache,
    detect_pii_llm,
)
from mmore.privacy.detection.openai_filter_engine import (
    OpenAIFilterEngine,
    clear_openai_filter_cache,
    detect_pii_openai,
)
from mmore.privacy.detection.presidio_engine import (
    PresidioEngine,
    _build_clinical_recognizers,
    clear_presidio_cache,
    detect_pii_presidio,
)
from mmore.rag.llm import LLMConfig
from mmore.utils import load_config

# --------------------------------------------------------------------------
# Fixtures and mock helpers
# --------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clear_detection_engine_caches():
    """Drop the module-level engine caches after every test in this module."""
    yield
    clear_gliner_cache()
    clear_openai_filter_cache()
    clear_presidio_cache()
    clear_llm_engine_cache()


def _fake_gliner_model(predictions):
    model = MagicMock()
    model.predict_entities.return_value = predictions
    return model


def _fake_openai_pipeline(predictions):
    fake = MagicMock()
    fake.return_value = predictions
    return fake


def _fake_presidio_result(start, end, entity_type, score):
    r = MagicMock()
    r.start = start
    r.end = end
    r.entity_type = entity_type
    r.score = score
    return r


def _fake_presidio_analyzer(results):
    fake = MagicMock()
    fake.analyze.return_value = results
    return fake


def _fake_dspy_span(text, label, score):
    s = MagicMock()
    s.text = text
    s.label = label
    s.score = score
    return s


def _fake_dspy_predictor(spans):
    prediction = MagicMock()
    prediction.spans = spans
    predictor = MagicMock()
    predictor.return_value = prediction
    return predictor


def _patch_dspy_engine(predictor):
    return patch.multiple(
        "mmore.privacy.detection.llm_engine",
        _build_dspy_lm=MagicMock(return_value=MagicMock()),
        _build_predictor=MagicMock(return_value=predictor),
    )


# --------------------------------------------------------------------------
# Detection engine tests
# --------------------------------------------------------------------------


def test_detection_config_round_trips_via_load_config():
    raw = {
        "engine": "llm",
        "entity_types": ["PERSON", "EMAIL", "MRN"],
        "confidence_threshold": 0.8,
        "llm": {"llm_name": "gpt2", "max_new_tokens": 512},
    }

    cfg = load_config(raw, DetectionConfig)

    assert isinstance(cfg, DetectionConfig)
    assert cfg.engine == "llm"
    assert cfg.entity_types == ["PERSON", "EMAIL", "MRN"]
    assert cfg.confidence_threshold == 0.8
    assert isinstance(cfg.llm, LLMConfig)
    assert cfg.llm.llm_name == "gpt2"
    assert cfg.llm.max_new_tokens == 512


def test_detection_config_defaults_when_minimal():
    cfg = load_config({"engine": "presidio"}, DetectionConfig)

    assert cfg.engine == "presidio"
    assert cfg.entity_types == []
    assert cfg.confidence_threshold == 0.7
    assert cfg.llm is None


def test_detect_pii_gliner_is_registered():
    assert "detect_pii_gliner" in tool_registry
    assert tool_registry["detect_pii_gliner"] is detect_pii_gliner


def test_gliner_engine_returns_spans_on_synthetic_note():
    fake = _fake_gliner_model(
        [
            {
                "start": 0,
                "end": 10,
                "label": "PERSON",
                "score": 0.95,
                "text": "John Smith",
            },
            {
                "start": 11,
                "end": 31,
                "label": "EMAIL",
                "score": 0.88,
                "text": "john@hospital.org",
            },
        ]
    )
    with patch(
        "mmore.privacy.detection.gliner_engine._load_gliner_model",
        return_value=fake,
    ):
        engine = GLiNEREngine(confidence_threshold=0.5)
        spans = engine.detect("John Smith john@hospital.org called.")

    assert len(spans) == 2
    assert spans[0].label == "PERSON"
    assert spans[0].score == 0.95
    assert spans[1].label == "EMAIL"


def test_gliner_engine_passes_threshold_to_model():
    fake = _fake_gliner_model([])
    with patch(
        "mmore.privacy.detection.gliner_engine._load_gliner_model",
        return_value=fake,
    ):
        engine = GLiNEREngine(confidence_threshold=0.7)
        engine.detect("synthetic note")

    fake.predict_entities.assert_called_once()
    kwargs = fake.predict_entities.call_args.kwargs
    assert kwargs["threshold"] == 0.7
    assert kwargs["text"] == "synthetic note"


def test_gliner_engine_loads_model_lazily_once():
    fake = _fake_gliner_model([])
    with patch(
        "mmore.privacy.detection.gliner_engine._load_gliner_model",
        return_value=fake,
    ) as mock_load:
        engine = GLiNEREngine()
        assert mock_load.call_count == 0
        engine.detect("first call")
        engine.detect("second call")
        assert mock_load.call_count == 1


def test_gliner_engine_shares_model_cache_across_instances():
    fake = _fake_gliner_model([])
    with patch(
        "mmore.privacy.detection.gliner_engine._load_gliner_model",
        return_value=fake,
    ) as mock_load:
        a = GLiNEREngine(confidence_threshold=0.4)
        b = GLiNEREngine(entity_types=["PERSON"], confidence_threshold=0.9)
        a.detect("x")
        b.detect("y")
        assert mock_load.call_count == 1


def test_gliner_engine_instances_apply_their_own_threshold():
    fake = _fake_gliner_model([])
    with patch(
        "mmore.privacy.detection.gliner_engine._load_gliner_model",
        return_value=fake,
    ):
        GLiNEREngine(confidence_threshold=0.4).detect("x")
        GLiNEREngine(confidence_threshold=0.9).detect("y")

    thresholds = [
        call.kwargs["threshold"] for call in fake.predict_entities.call_args_list
    ]
    assert thresholds == [0.4, 0.9]


def test_gliner_engine_from_config_propagates_threshold_and_labels():
    cfg = DetectionConfig(
        engine="gliner",
        entity_types=["PERSON", "MRN"],
        confidence_threshold=0.55,
    )
    engine = GLiNEREngine.from_config(cfg)

    fake = _fake_gliner_model([])
    with patch(
        "mmore.privacy.detection.gliner_engine._load_gliner_model",
        return_value=fake,
    ):
        engine.detect("synthetic")

    kwargs = fake.predict_entities.call_args.kwargs
    assert kwargs["threshold"] == 0.55
    assert kwargs["labels"] == ["PERSON", "MRN"]


def test_detect_pii_openai_is_registered():
    assert "detect_pii_openai" in tool_registry
    assert tool_registry["detect_pii_openai"] is detect_pii_openai


def test_openai_filter_engine_returns_spans_on_synthetic_note():
    fake = _fake_openai_pipeline(
        [
            {"start": 0, "end": 10, "entity_group": "PERSON", "score": 0.95},
            {"start": 11, "end": 31, "entity_group": "EMAIL", "score": 0.88},
        ]
    )
    with patch(
        "mmore.privacy.detection.openai_filter_engine._load_openai_filter_pipeline",
        return_value=fake,
    ):
        engine = OpenAIFilterEngine(confidence_threshold=0.5)
        spans = engine.detect("John Smith john@hospital.org called.")

    assert len(spans) == 2
    assert spans[0].label == "PERSON"
    assert spans[1].label == "EMAIL"


def test_openai_filter_engine_filters_below_threshold():
    fake = _fake_openai_pipeline(
        [
            {"start": 0, "end": 10, "entity_group": "PERSON", "score": 0.95},
            {"start": 11, "end": 25, "entity_group": "EMAIL", "score": 0.30},
        ]
    )
    with patch(
        "mmore.privacy.detection.openai_filter_engine._load_openai_filter_pipeline",
        return_value=fake,
    ):
        engine = OpenAIFilterEngine(confidence_threshold=0.7)
        spans = engine.detect("synthetic")

    assert len(spans) == 1
    assert spans[0].label == "PERSON"


def test_openai_filter_engine_restricts_to_entity_types():
    fake = _fake_openai_pipeline(
        [
            {"start": 0, "end": 10, "entity_group": "PERSON", "score": 0.95},
            {"start": 11, "end": 21, "entity_group": "DATE", "score": 0.90},
        ]
    )
    with patch(
        "mmore.privacy.detection.openai_filter_engine._load_openai_filter_pipeline",
        return_value=fake,
    ):
        engine = OpenAIFilterEngine(entity_types=["PERSON"], confidence_threshold=0.5)
        spans = engine.detect("synthetic")

    assert len(spans) == 1
    assert spans[0].label == "PERSON"


def test_openai_filter_engine_shares_pipeline_cache_across_instances():
    fake = _fake_openai_pipeline([])
    with patch(
        "mmore.privacy.detection.openai_filter_engine._load_openai_filter_pipeline",
        return_value=fake,
    ) as mock_load:
        OpenAIFilterEngine().detect("x")
        OpenAIFilterEngine(confidence_threshold=0.9).detect("y")
        assert mock_load.call_count == 1


def test_openai_filter_engine_from_config_applies_threshold_and_labels():
    cfg = DetectionConfig(
        engine="openai_filter",
        entity_types=["PERSON"],
        confidence_threshold=0.6,
    )
    engine = OpenAIFilterEngine.from_config(cfg)

    fake = _fake_openai_pipeline(
        [
            {"start": 0, "end": 10, "entity_group": "PERSON", "score": 0.95},
            {"start": 11, "end": 20, "entity_group": "PERSON", "score": 0.50},
            {"start": 21, "end": 30, "entity_group": "EMAIL", "score": 0.95},
        ]
    )
    with patch(
        "mmore.privacy.detection.openai_filter_engine._load_openai_filter_pipeline",
        return_value=fake,
    ):
        spans = engine.detect("synthetic")

    assert len(spans) == 1
    assert spans[0].label == "PERSON"
    assert spans[0].score == 0.95


def test_detect_pii_presidio_is_registered():
    assert "detect_pii_presidio" in tool_registry
    assert tool_registry["detect_pii_presidio"] is detect_pii_presidio


def test_presidio_engine_returns_spans_on_synthetic_note():
    fake = _fake_presidio_analyzer(
        [
            _fake_presidio_result(0, 10, "PERSON", 0.95),
            _fake_presidio_result(11, 31, "EMAIL_ADDRESS", 0.88),
        ]
    )
    with patch(
        "mmore.privacy.detection.presidio_engine._load_presidio_analyzer",
        return_value=fake,
    ):
        engine = PresidioEngine(confidence_threshold=0.5)
        spans = engine.detect("John Smith john@hospital.org called.")

    assert len(spans) == 2
    assert spans[0].label == "PERSON"
    assert spans[0].score == 0.95
    assert spans[1].label == "EMAIL_ADDRESS"


def test_presidio_engine_passes_threshold_and_entity_types_to_analyzer():
    fake = _fake_presidio_analyzer([])
    with patch(
        "mmore.privacy.detection.presidio_engine._load_presidio_analyzer",
        return_value=fake,
    ):
        engine = PresidioEngine(
            entity_types=["PERSON", "MRN"], confidence_threshold=0.55
        )
        engine.detect("synthetic note")

    kwargs = fake.analyze.call_args.kwargs
    assert kwargs["text"] == "synthetic note"
    assert kwargs["score_threshold"] == 0.55
    assert kwargs["entities"] == ["PERSON", "MRN"]
    assert kwargs["language"] == "en"


def test_presidio_engine_loads_analyzer_lazily_once():
    fake = _fake_presidio_analyzer([])
    with patch(
        "mmore.privacy.detection.presidio_engine._load_presidio_analyzer",
        return_value=fake,
    ) as mock_load:
        engine = PresidioEngine()
        assert mock_load.call_count == 0
        engine.detect("first")
        engine.detect("second")
        assert mock_load.call_count == 1


def test_presidio_engine_shares_analyzer_cache_across_instances():
    fake = _fake_presidio_analyzer([])
    with patch(
        "mmore.privacy.detection.presidio_engine._load_presidio_analyzer",
        return_value=fake,
    ) as mock_load:
        PresidioEngine().detect("x")
        PresidioEngine(confidence_threshold=0.9).detect("y")
        assert mock_load.call_count == 1


def test_presidio_engine_from_config_propagates_threshold_and_labels():
    cfg = DetectionConfig(
        engine="presidio",
        entity_types=["PERSON", "MRN"],
        confidence_threshold=0.55,
    )
    engine = PresidioEngine.from_config(cfg)

    fake = _fake_presidio_analyzer([])
    with patch(
        "mmore.privacy.detection.presidio_engine._load_presidio_analyzer",
        return_value=fake,
    ):
        engine.detect("synthetic")

    kwargs = fake.analyze.call_args.kwargs
    assert kwargs["score_threshold"] == 0.55
    assert kwargs["entities"] == ["PERSON", "MRN"]


def test_presidio_clinical_recognizers_cover_mrn_hospital_date_insurance_id():
    recognizers = _build_clinical_recognizers()

    supported = {r.supported_entities[0] for r in recognizers}
    assert supported == {"MRN", "HOSPITAL_DATE", "INSURANCE_ID"}

    by_entity = {r.supported_entities[0]: r for r in recognizers}
    mrn_regexes = [p.regex for p in by_entity["MRN"].patterns]
    assert any("MRN" in regex for regex in mrn_regexes)

    hospital_date_regexes = [p.regex for p in by_entity["HOSPITAL_DATE"].patterns]
    assert any(r"\d{4}-\d{2}-\d{2}" in regex for regex in hospital_date_regexes)

    insurance_regexes = [p.regex for p in by_entity["INSURANCE_ID"].patterns]
    assert any(r"[A-Z]" in regex for regex in insurance_regexes)


def test_detect_pii_llm_is_registered():
    assert "detect_pii_llm" in tool_registry
    assert tool_registry["detect_pii_llm"] is detect_pii_llm


def test_llm_engine_returns_spans_on_synthetic_note():
    note = "John Smith john@hospital.org called."
    predictor = _fake_dspy_predictor(
        [
            _fake_dspy_span("John Smith", "PERSON", 0.95),
            _fake_dspy_span("john@hospital.org", "EMAIL", 0.88),
        ]
    )
    cfg = LLMConfig(llm_name="gpt2", max_new_tokens=128)
    with _patch_dspy_engine(predictor):
        engine = LLMDetectionEngine(cfg, confidence_threshold=0.5)
        spans = engine.detect(note)

    assert len(spans) == 2
    assert spans[0].label == "PERSON"
    assert spans[0].start == 0
    assert spans[0].end == len("John Smith")
    assert spans[0].score == 0.95
    assert spans[1].label == "EMAIL"
    assert note[spans[1].start : spans[1].end] == "john@hospital.org"


def test_llm_engine_filters_below_threshold():
    note = "John Smith and email me at the hidden line."
    predictor = _fake_dspy_predictor(
        [
            _fake_dspy_span("John Smith", "PERSON", 0.95),
            _fake_dspy_span("hidden", "EMAIL", 0.30),
        ]
    )
    cfg = LLMConfig(llm_name="gpt2")
    with _patch_dspy_engine(predictor):
        engine = LLMDetectionEngine(cfg, confidence_threshold=0.7)
        spans = engine.detect(note)

    assert len(spans) == 1
    assert spans[0].label == "PERSON"


def test_llm_engine_clamps_out_of_range_scores_to_unit_interval():
    note = "John and Mary worked together."
    predictor = _fake_dspy_predictor(
        [
            _fake_dspy_span("John", "PERSON", 1.7),  # above 1
            _fake_dspy_span("Mary", "PERSON", -0.3),  # below 0
        ]
    )
    cfg = LLMConfig(llm_name="gpt2")
    with _patch_dspy_engine(predictor):
        engine = LLMDetectionEngine(cfg, confidence_threshold=0.0)
        spans = engine.detect(note)

    assert len(spans) == 2
    assert spans[0].score == 1.0
    assert spans[1].score == 0.0


def test_llm_engine_skips_fragments_not_in_source_text():
    note = "Patient John Smith called."
    predictor = _fake_dspy_predictor(
        [
            _fake_dspy_span("John Smith", "PERSON", 0.95),
            _fake_dspy_span("Jane Doe", "PERSON", 0.95),  # not in note
        ]
    )
    cfg = LLMConfig(llm_name="gpt2")
    with _patch_dspy_engine(predictor):
        engine = LLMDetectionEngine(cfg, confidence_threshold=0.5)
        spans = engine.detect(note)

    assert len(spans) == 1
    assert spans[0].label == "PERSON"
    assert note[spans[0].start : spans[0].end] == "John Smith"


def test_llm_engine_passes_text_and_entity_types_to_predictor():
    predictor = _fake_dspy_predictor([])
    cfg = LLMConfig(llm_name="gpt2")
    with _patch_dspy_engine(predictor):
        engine = LLMDetectionEngine(
            cfg, entity_types=["PERSON", "MRN"], confidence_threshold=0.5
        )
        engine.detect("synthetic note")

    call_kwargs = predictor.call_args.kwargs
    assert call_kwargs["text"] == "synthetic note"
    assert call_kwargs["entity_types"] == ["PERSON", "MRN"]


def test_llm_engine_returns_empty_on_predictor_failure():
    predictor = MagicMock()
    predictor.side_effect = ValueError("malformed structured output")
    cfg = LLMConfig(llm_name="gpt2")
    with _patch_dspy_engine(predictor):
        engine = LLMDetectionEngine(cfg, confidence_threshold=0.5)
        spans = engine.detect("synthetic")

    assert spans == []


def test_llm_engine_skips_malformed_individual_spans():
    bad = MagicMock()
    bad.text = "John"
    bad.label = "PERSON"
    bad.score = "not-a-float"
    good = _fake_dspy_span("John", "PERSON", 0.95)
    predictor = _fake_dspy_predictor([bad, good])
    cfg = LLMConfig(llm_name="gpt2")
    with _patch_dspy_engine(predictor):
        engine = LLMDetectionEngine(cfg, confidence_threshold=0.5)
        spans = engine.detect("John called.")

    assert len(spans) == 1
    assert spans[0].score == 0.95


def test_llm_engine_from_config_requires_llm_block():
    cfg = DetectionConfig(
        engine="llm", entity_types=[], confidence_threshold=0.7, llm=None
    )
    with pytest.raises(ValueError, match="DetectionConfig.llm"):
        LLMDetectionEngine.from_config(cfg)


def test_llm_engine_hf_provider_routes_to_local_hf_lm():
    from mmore.privacy.detection.llm_engine import _build_dspy_lm, _LocalHFLM

    cfg = LLMConfig(llm_name="some-org/some-model", max_new_tokens=16)
    assert cfg.provider == "HF"
    assert cfg.base_url is None

    with patch(
        "mmore.privacy.detection.llm_engine._load_local_hf_pipeline",
        return_value=MagicMock(),
    ):
        lm = _build_dspy_lm(cfg)

    assert isinstance(lm, _LocalHFLM)


def test_llm_engine_openai_provider_still_uses_litellm():
    import dspy

    from mmore.privacy.detection.llm_engine import _build_dspy_lm, _LocalHFLM

    cfg = LLMConfig(llm_name="gpt-4o-mini", max_new_tokens=16)
    assert cfg.provider == "OPENAI"

    with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
        lm = _build_dspy_lm(cfg)

    assert isinstance(lm, dspy.LM)
    assert not isinstance(lm, _LocalHFLM)


def test_llm_engine_from_config_propagates_threshold_and_labels():
    cfg = DetectionConfig(
        engine="llm",
        entity_types=["PERSON", "MRN"],
        confidence_threshold=0.55,
        llm=LLMConfig(llm_name="gpt2", max_new_tokens=128),
    )
    engine = LLMDetectionEngine.from_config(cfg)
    note = "John Smith and Mary Jones met."
    predictor = _fake_dspy_predictor(
        [
            _fake_dspy_span("John Smith", "PERSON", 0.95),
            _fake_dspy_span("Mary Jones", "PERSON", 0.40),
        ]
    )
    with _patch_dspy_engine(predictor):
        spans = engine.detect(note)

    assert len(spans) == 1
    assert spans[0].score == 0.95
    assert predictor.call_args.kwargs["entity_types"] == ["PERSON", "MRN"]


@pytest.mark.parametrize(
    "engine_name, expected_tool",
    [
        ("gliner", detect_pii_gliner),
        ("openai_filter", detect_pii_openai),
        ("presidio", detect_pii_presidio),
        ("llm", detect_pii_llm),
    ],
)
def test_detection_engine_name_resolves_to_registered_tool(engine_name, expected_tool):
    cfg = load_config(
        {
            "engine": engine_name,
            "entity_types": ["PERSON"],
            "confidence_threshold": 0.5,
        },
        DetectionConfig,
    )
    assert cfg.engine == engine_name

    tool_name = f"detect_pii_{engine_name.replace('openai_filter', 'openai')}"
    assert tool_name in tool_registry
    assert tool_registry[tool_name] is expected_tool
