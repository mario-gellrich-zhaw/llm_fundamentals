"""Token-budgeted passages for the E5 exercise (all processing is local)."""
import pandas as pd


def chunk_reviews(frame, tokenizer, max_tokens=512, overlap=64):
    """Keep original text spans, stable local citations and overlapping E5 windows.

    Count the passage prefix and special tokens too. Verify the final substring
    because tokenizing a substring can differ at word boundaries.
    """
    budget = max_tokens - len(tokenizer.encode('passage: ', add_special_tokens=False)) - tokenizer.num_special_tokens_to_add(pair=False)
    if not 0 <= overlap < budget:
        raise ValueError('Overlap must be smaller than the passage token budget.')
    rows = []
    for position, (_, review) in enumerate(frame.iterrows(), 1):
        text = review['text']
        offsets = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)['offset_mapping']
        start, part = 0, 1
        while start < len(offsets):
            end = min(start + budget, len(offsets))
            while end > start:
                passage = text[offsets[start][0]:offsets[end - 1][1]]
                if len(tokenizer.encode('passage: ' + passage, add_special_tokens=True)) <= max_tokens:
                    break
                end -= 1
            if end == start:
                raise ValueError('A passage cannot fit the embedding token budget.')
            rows.append({**review.to_dict(), 'text': passage,
                         'citation_id': f'review_{position:04d}_part_{part:02d}'})
            if end == len(offsets):
                break
            start = max(start + 1, end - overlap)
            part += 1
    if not rows:
        raise ValueError('No nonempty review text to index.')
    return pd.DataFrame(rows).reset_index(drop=True)
