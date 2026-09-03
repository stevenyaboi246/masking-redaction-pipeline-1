from types import SimpleNamespace

from redaction_pipeline.detection import AnthropicDetector, AzureLanguageDetector
from redaction_pipeline.models import Document, PageText, Word


def _document():
    text = "Patient Maria Vasquez visited Northside Clinic"
    words = []
    cursor = 0
    for token in text.split():
        start = text.index(token, cursor)
        end = start + len(token)
        words.append(Word(1, token, start, end, (start, 10, end, 20)))
        cursor = end
    return Document("fake.pdf", [PageText(1, text, words)])


def test_claude_exact_text_is_mapped_to_geometry():
    detector = object.__new__(AnthropicDetector)
    detector.model = "test-model"
    response = SimpleNamespace(content=[SimpleNamespace(
        type="text", text='[{"entity_type":"patient_name","text":"Maria Vasquez","confidence":0.97}]'
    )])
    detector.client = SimpleNamespace(messages=SimpleNamespace(create=lambda **kwargs: response))
    spans = detector.detect(_document())
    assert spans[0].entity_type == "patient_name"
    assert spans[0].boxes
    assert spans[0].engine == "anthropic"


def test_azure_categories_are_mapped_to_internal_taxonomy():
    detector = object.__new__(AzureLanguageDetector)
    entity = SimpleNamespace(category="Person", offset=8, length=13, confidence_score=0.94)
    result = SimpleNamespace(is_error=False, entities=[entity])
    detector.client = SimpleNamespace(recognize_pii_entities=lambda docs: [result])
    spans = detector.detect(_document())
    assert spans[0].entity_type == "person_name"
    assert spans[0].engine == "azure_ai_language"
