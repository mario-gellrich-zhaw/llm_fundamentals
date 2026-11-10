"""Small, explicit output checks shared by the course notebooks."""
import json
import re

LABELS = ('positive', 'negative', 'mixed')


def parse_label(text):
    """Accept a one-label answer or an explicit final Overall: label line.

    Ignore local-model thinking blocks. Never guess a label from explanation words.
    """
    text = re.sub(r'<think>.*?</think>', '', text or '', flags=re.DOTALL).strip().lower()
    final = re.search(r'\boverall:\s*(positive|negative|mixed)\s*[.!]?\s*$', text)
    if final:
        return final.group(1)
    single = re.fullmatch(r'(positive|negative|mixed)[.!]?', text)
    return single.group(1) if single else 'unparsed'


def response_label(response):
    """Represent refused/incomplete/invalid structured answers as explicit outcomes."""
    if any(getattr(part, 'type', '') == 'refusal'
           for item in response.output for part in getattr(item, 'content', [])):
        return 'refused'
    if response.status != 'completed':
        return 'incomplete'
    try:
        label = json.loads(response.output_text)['sentiment']
    except (ValueError, KeyError, TypeError):
        return 'unparsed'
    return label if isinstance(label, str) and label in LABELS else 'unparsed'
