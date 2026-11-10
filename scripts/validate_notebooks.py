"""Exercise the five practical notebooks locally with fake API/embedding/Ollama clients.

No paid requests, model downloads or shell setup. Real local anonymization runs.
This verifies notebook wiring, not hosted-model quality or actual E5/Ollama speed.
Run after installing requirements and en_core_web_sm: python scripts/validate_notebooks.py
"""
from __future__ import annotations
import contextlib
import hashlib
import io
import json
from pathlib import Path
import socket
import sys
import tempfile
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
import zipfile

import numpy as np
from openai.types.responses import Response
from openai.types.decision import Decision
import llm_course
from llm_course import client

ROOT = Path(__file__).resolve().parents[1]


class Tokenizer:
    """Character tokens: deterministic, no model files or network."""
    def encode(self, text, add_special_tokens=False):
        return list(text) + ([0, 1] if add_special_tokens else [])
    def num_special_tokens_to_add(self, pair=False):
        return 2
    def __call__(self, text, **kwargs):
        return {'offset_mapping': [(i, i + 1) for i in range(len(text))]}


class Embeddings:
    def __init__(self, *args, **kwargs):
        self.tokenizer = Tokenizer()
        self.max_seq_length = 512
    def encode(self, texts, **kwargs):
        vectors = np.array([list(hashlib.sha256(t.encode()).digest()) for t in texts], dtype=float)
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def response(**kwargs):
    text = '{"sentiment":"negative"}' if 'text' in kwargs else 'Overall: negative.'
    return Response.model_validate({
        'id':'resp_offline','object':'response','created_at':1,'model':'gpt-6-luna','status':'completed',
        'parallel_tool_calls':True,'tool_choice':'auto','tools':[],
        'output':[{'id':'msg_offline','type':'message','status':'completed','role':'assistant',
                   'content':[{'type':'output_text','text':text,'annotations':[]}]}],
        'usage': {'input_tokens':100,'output_tokens':10,'total_tokens':110,
                  'input_tokens_details':{'cached_tokens':0,'cache_write_tokens':0},
                  'output_tokens_details':{'reasoning_tokens':5}},
    })


def decision(**kwargs):
    return Decision.model_validate({'model':'gpt-6-luna','answers':[
        {'type':'choice','name':'sentiment','choice':'negative','confidence':.9,
         'probabilities':[{'value':'negative','probability':.9},{'value':'positive','probability':.05},{'value':'mixed','probability':.05}]}],
        'usage':{'input_tokens':100,'output_tokens':10,'total_tokens':110,
                 'input_tokens_details':{'cached_tokens':0,'cache_write_tokens':0},
                 'output_tokens_details':{'reasoning_tokens':5}}})


def main():
    fake_module = ModuleType('sentence_transformers')
    fake_module.SentenceTransformer = Embeddings
    fake_api = SimpleNamespace(responses=SimpleNamespace(create=response), decisions=SimpleNamespace(create=decision))
    fake_local = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **kw: SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='Overall: negative.'))]))))
    archive = ROOT / 'data/llm_fundamentals_data.zip'
    if not archive.exists():
        raise FileNotFoundError('Download the Moodle zip into data/ first; no API is used by this validator.')
    import matplotlib
    matplotlib.use('Agg')
    with tempfile.TemporaryDirectory() as tmp, contextlib.ExitStack() as stack:
        data = Path(tmp) / 'data'
        data.mkdir()
        (data / 'processed').mkdir()
        with zipfile.ZipFile(archive) as z:
            for filename in ('ikea_reviews.csv', 'businesses_mcdonalds_ikea.csv', 'fewshot_examples.json'):
                (data / filename).write_bytes(z.read(filename))
        stack.enter_context(patch.object(llm_course, 'DATA', data))
        stack.enter_context(patch.object(client, 'CACHE_DIR', Path(tmp) / 'cache'))
        stack.enter_context(patch.object(client, 'get_client', return_value=fake_api))
        stack.enter_context(patch('openai.OpenAI', return_value=fake_local))
        stack.enter_context(patch.dict(sys.modules, {'sentence_transformers':fake_module}))
        stack.enter_context(patch.object(socket.socket, 'connect', side_effect=AssertionError('Network disabled in offline validation')))
        for folder in sorted(ROOT.glob('0[1-5]_*')):
            path = next(folder.glob('*.ipynb'))
            notebook = json.loads(path.read_text())
            namespace = {'__name__':'__main__'}
            count = 0
            for index, cell in enumerate(notebook['cells']):
                if cell['cell_type'] != 'code':
                    continue
                source = ''.join(cell['source'])
                if any(line.startswith('!') for line in source.splitlines()):
                    continue  # Never install/start/download Ollama in this check.
                try:
                    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                        exec(compile(source, f'{path.name}:cell{index}', 'exec'), namespace)
                except Exception as exc:
                    raise RuntimeError(f'{path.name}, cell {index} failed: {type(exc).__name__}') from exc
                count += 1
            if folder.name.startswith('03_'):
                assert not set(namespace['test'].text) & {ex['text'] for ex in namespace['fewshot']}
            print(f'PASS {path.relative_to(ROOT)} ({count} Python cells; fake clients)', flush=True)
    print('No paid calls, remote inference, downloads or original-data changes.')


if __name__ == '__main__':
    main()
