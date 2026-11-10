"""Local de-identification for the English course data; no remote recognizers.

Automated redaction is imperfect. Inspect the prepared text locally before use;
remove remaining identifiers and identifying stories. It is not proof of anonymity.
"""
from functools import lru_cache
from threading import RLock

_lock = RLock()


@lru_cache(maxsize=1)
def _analyzer():
    try:
        import spacy
        from presidio_analyzer import AnalyzerEngine
        from presidio_analyzer.nlp_engine import NlpEngineProvider
    except ImportError as exc:
        raise RuntimeError(
            'Install requirements.txt and run: python -m spacy download en_core_web_sm. '
            'No request has been sent.'
        ) from exc
    if not spacy.util.is_package('en_core_web_sm'):
        raise RuntimeError('Privacy model missing. Run: python -m spacy download en_core_web_sm. No request has been sent.')
    engine = NlpEngineProvider(nlp_configuration={
        'nlp_engine_name': 'spacy',
        'models': [{'lang_code': 'en', 'model_name': 'en_core_web_sm'}],
    }).create_engine()
    return AnalyzerEngine(nlp_engine=engine, supported_languages=['en'])


@lru_cache(maxsize=8192)
def anonymize(text: str) -> str:
    """Replace detected identifiers with <REDACTED>, entirely on this machine.

    Includes people, locations, dates, email, phone, URL, IP and ID/account
    recognizers supplied by Presidio. English only; false positives/negatives occur.
    Overlapping detections are merged, so no fragments of an overlap survive.
    """
    if not isinstance(text, str):
        raise TypeError('Expected text for local anonymization.')
    if not text:
        return text
    with _lock:
        results = _analyzer().analyze(text=text, language='en', score_threshold=0.35)
    spans = []
    for result in sorted(results, key=lambda r: (r.start, r.end)):
        if spans and result.start <= spans[-1][1]:
            spans[-1] = (spans[-1][0], max(spans[-1][1], result.end))
        else:
            spans.append((result.start, result.end))
    for start, end in reversed(spans):
        text = text[:start] + '<REDACTED>' + text[end:]
    return text


def _input(value):
    if isinstance(value, str):
        return anonymize(value)
    if isinstance(value, list):
        return [_input(item) for item in value]
    if isinstance(value, dict):
        # Course helpers support inline text only, never uploads, URLs or stored
        # conversation references that could bypass local de-identification.
        if value.get('type') not in (None, 'message', 'input_text', 'output_text'):
            raise ValueError('Only inline text messages are supported by the privacy helper.')
        if set(value) - {'role', 'type', 'content', 'text'}:
            raise ValueError('Unsupported message fields; use inline role/content messages.')
        return {k: _input(v) if k in ('content', 'text') else v for k, v in value.items()}
    raise TypeError('Input must be text or a list of inline text messages.')


def sanitize_request(kwargs: dict) -> dict:
    """Redact all course prompt text before computing a cache key or sending it."""
    if kwargs.get('stream') or any(kwargs.get(k) for k in ('previous_response_id', 'conversation', 'tools')):
        raise ValueError('Course privacy helpers support non-streaming, self-contained text calls only.')
    result = dict(kwargs)
    result['input'] = _input(result['input'])
    if result.get('instructions') is not None:
        result['instructions'] = anonymize(result['instructions'])
    if 'questions' in result:
        result['questions'] = [dict(q, instructions=anonymize(q['instructions'])) for q in result['questions']]
    # Do not permit arbitrary metadata/prompt templates to carry unredacted data.
    if any(result.get(k) for k in ('metadata', 'prompt')):
        raise ValueError('Metadata and stored prompt templates are unsupported in course helpers.')
    return result
