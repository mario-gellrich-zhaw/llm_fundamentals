"""Offline regression checks: no real API client or network requests."""
import concurrent.futures
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from openai.types.responses import Response
from openai.types.decision import Decision
from llm_course import client, privacy
from llm_course.evaluation import parse_label, response_label
import pandas as pd


def response(text='{"sentiment":"negative"}', status='completed', refusal=False):
    return Response.model_validate({
        'id':'resp_test','created_at':1,'model':'gpt-6-luna','object':'response',
        'status':status,'parallel_tool_calls':True,'tool_choice':'auto','tools':[],
        'output':[{'id':'msg_test','type':'message','status':'completed','role':'assistant',
                   'content':[{'type':'refusal','refusal':'No'}] if refusal else
                   [{'type':'output_text','text':text,'annotations':[]}]}],
    })


class PrivacyTests(unittest.TestCase):
    def tearDown(self):
        privacy.anonymize.cache_clear()

    def test_overlapping_spans_leave_no_identifier_fragments(self):
        detector=SimpleNamespace(analyze=lambda **kw: [SimpleNamespace(start=0,end=4),SimpleNamespace(start=0,end=10)])
        with patch.object(privacy,'_analyzer',return_value=detector):
            self.assertEqual(privacy.anonymize('John Smith was helpful.'),'<REDACTED> was helpful.')

    def test_history_examples_and_decision_instructions_are_redacted(self):
        def redact(s): return s.replace('john@example.org','<REDACTED>')
        with patch.object(privacy,'anonymize',side_effect=redact):
            kwargs=privacy.sanitize_request({'input':[{'role':'user','content':[{'type':'input_text','text':'john@example.org'}]}],
               'instructions':'Example: john@example.org','questions':[{'instructions':'john@example.org','type':'choice'}]})
        self.assertNotIn('john@example.org',json.dumps(kwargs))

    def test_non_text_references_and_uploads_are_blocked(self):
        with self.assertRaises(ValueError):
            privacy.sanitize_request({'input':[{'type':'input_file','file_id':'file_raw'}]})
        with self.assertRaises(ValueError):
            privacy.sanitize_request({'input':'safe','previous_response_id':'resp_raw'})

    def test_detector_failure_prevents_client_access(self):
        privacy.anonymize.cache_clear()
        with patch.object(privacy,'_analyzer',side_effect=RuntimeError('missing model')), patch.object(client,'get_client') as get:
            with self.assertRaises(RuntimeError):client.ask(model='gpt-6-luna',input='John Smith')
            get.assert_not_called()


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.sent=[]
        def create(**kwargs):
            self.sent.append(kwargs);time.sleep(.01);return response()
        self.fake=SimpleNamespace(responses=SimpleNamespace(create=create),decisions=SimpleNamespace(create=create))
        self.patches=[patch.object(client,'CACHE_DIR',Path(self.tmp.name)),
                      patch.object(client,'get_client',return_value=self.fake),
                      patch.object(privacy,'anonymize',side_effect=lambda s:s.replace('John Smith','<REDACTED>'))]
        for p in self.patches:p.start()

    def tearDown(self):
        for p in reversed(self.patches):p.stop()
        self.tmp.cleanup()

    def test_duplicate_requests_are_sent_once_and_raw_text_never_reaches_client(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            results=list(pool.map(lambda _:client.ask(model='gpt-6-luna',input='John Smith complains.'),range(8)))
        self.assertEqual(len(self.sent),1)
        self.assertEqual(self.sent[0]['input'],'<REDACTED> complains.')
        self.assertFalse(self.sent[0]['store'])
        self.assertEqual(len(results),8)
        self.assertEqual(len(list(Path(self.tmp.name).glob('*.json'))),1)
        self.assertEqual(list(Path(self.tmp.name).glob('*.tmp')),[])
        # ask_many cache hits must work without even constructing an API client.
        with patch.object(client,'get_client',side_effect=AssertionError('cache should need no key')):
            self.assertEqual(len(client.ask_many(['John Smith complains.']*2,model='gpt-6-luna')),2)

    def test_corrupt_cache_is_replaced_and_new_run_sends_new_request(self):
        client.ask(model='gpt-6-luna',input='safe')
        path=next(Path(self.tmp.name).glob('*.json'));path.write_text('{broken')
        with self.assertWarns(UserWarning):client.ask(model='gpt-6-luna',input='safe')
        client.ask(model='gpt-6-luna',input='safe',run=1)
        self.assertEqual(len(self.sent),3)

    def test_incomplete_and_refused_responses_are_not_cached(self):
        for result in [response(status='incomplete'),response(refusal=True)]:
            self.fake.responses.create=lambda **kw:result
            client.ask(model='gpt-6-luna',input='safe')
            self.assertEqual(list(Path(self.tmp.name).glob('*.json')),[])

    def test_decisions_are_redacted_cached_and_refusals_are_not_cached(self):
        choice=Decision.model_validate({'id':'dec_test','object':'decision','usage':{'input_tokens':1,'output_tokens':1,'total_tokens':2,'input_tokens_details':{'cached_tokens':0,'cache_write_tokens':0},'output_tokens_details':{'reasoning_tokens':0}},'created_at':1,'model':'gpt-6-luna',
          'answers':[{'type':'choice','name':'sentiment','choice':'negative','confidence':.9,
                      'probabilities':[{'value':'negative','probability':.9}]}]})
        calls=[]
        def create(**kw):calls.append(kw);return choice
        self.fake.decisions.create=create
        request={'model':'gpt-6-luna','input':'John Smith','questions':[{'type':'choice','name':'sentiment','instructions':'John Smith','choices':[{'value':'negative'}]}]}
        client.ask_decision(**request);client.ask_decision(**request)
        self.assertEqual(len(calls),1)
        self.assertNotIn('John Smith',json.dumps(calls))
        refused=Decision.model_validate({'id':'dec_no','object':'decision','usage':{'input_tokens':1,'output_tokens':1,'total_tokens':2,'input_tokens_details':{'cached_tokens':0,'cache_write_tokens':0},'output_tokens_details':{'reasoning_tokens':0}},'created_at':1,'model':'gpt-6-luna','answers':[{'type':'refusal','name':'sentiment'}]})
        self.fake.decisions.create=lambda **kw:refused
        client.ask_decision(**request,run=1)
        self.assertEqual(len(list(Path(self.tmp.name).glob('*.json'))),1)


class OutputTests(unittest.TestCase):
    def test_final_label_parser_does_not_read_explanation_or_substrings(self):
        for text,label in [('Not positive. Overall: negative.','negative'),
          ('<think>positive or negative?</think>Overall: mixed','mixed'),('positive.','positive'),
          ('not positive','unparsed'),('positively','unparsed'),('', 'unparsed')]:
            self.assertEqual(parse_label(text),label)

    def test_structured_failures_are_explicit(self):
        for result,label in [(response(),'negative'),(response(refusal=True),'refused'),
          (response(status='incomplete'),'incomplete'),(response('bad json'),'unparsed'),
          (response('{"sentiment": []}'),'unparsed')]:self.assertEqual(response_label(result),label)


ROOT = Path(__file__).resolve().parents[1]


def code_cell(notebook, marker):
    """Source of the first code cell of a notebook that contains the marker text."""
    cells = json.loads((ROOT / notebook).read_text())['cells']
    return next(''.join(c['source']) for c in cells
                if c['cell_type'] == 'code' and marker in ''.join(c['source']))


class NotebookTests(unittest.TestCase):
    """The notebook cells that protect data, guide students and compute costs."""

    def test_notebook_01_saves_only_anonymized_text(self):
        notebook = '01_Explore_Yelp_Data/explore_yelp_data.ipynb'
        with tempfile.TemporaryDirectory() as tmp:
            sample = pd.DataFrame({'id': ['R001'], 'stars': [1], 'text': ['John Smith was rude.']})
            ns = {'DATA': Path(tmp), 'os': os, 'pd': pd, 'sample': sample, 'tqdm': lambda values: values,
                  'anonymize': lambda text: text.replace('John Smith', '<REDACTED>')}
            import contextlib
            import io
            with contextlib.redirect_stdout(io.StringIO()):
                exec(code_cell(notebook, '# Keep the original text for the comparison'), ns)
                exec(code_cell(notebook, '# Save the sample without the original texts'), ns)
            saved = (Path(tmp) / 'processed/reviews_sample.csv').read_text()
            self.assertIn('<REDACTED> was rude.', saved)
            self.assertNotIn('John Smith', saved)
            self.assertNotIn('original', saved)

    def test_api_notebooks_explain_which_notebook_to_run_first(self):
        cases = [('03_Classify_Reviews/classify_reviews.ipynb', 'sample_path = ', 'notebook 01'),
                 ('04_Mini_RAG/mini_rag.ipynb', 'sample_path = ', 'notebook 01'),
                 ('05_Local_Model/local_model.ipynb', 'silver_path = ', 'notebook 03')]
        with tempfile.TemporaryDirectory() as tmp:
            # A raw file must never be used as an automatic fallback.
            (Path(tmp) / 'ikea_reviews.csv').write_text('review_id,text\nraw-id,raw personal information\n')
            for notebook, marker, hint in cases:
                with self.subTest(notebook=notebook), self.assertRaisesRegex(AssertionError, hint):
                    exec(code_cell(notebook, marker), {'DATA': Path(tmp), 'pd': pd})

    def test_notebook_04_retrieve_combines_keyword_and_meaning_ranks(self):
        import numpy as np
        source = code_cell('04_Mini_RAG/mini_rag.ipynb', 'def retrieve(').split('# Our three questions')[0]
        ns = {'pd': pd, 'tokenize': str.split,
              'df': pd.DataFrame({'id': ['R001', 'R002', 'R003'], 'stars': [1, 2, 3], 'text': ['a', 'b', 'c']}),
              'bm25': SimpleNamespace(get_scores=lambda query: [3., 2., 1.]),        # ranks 1, 2, 3
              'embeddings': np.array([[.1], [.9], [.5]]),                            # ranks 3, 1, 2
              'model': SimpleNamespace(encode=lambda text, **kw: np.array([1.]))}
        exec(source, ns)
        self.assertEqual(ns['retrieve']('question', k=2)['id'].tolist(), ['R002', 'R001'])

    def test_notebook_05_cost_uses_input_and_output_tokens_without_cache(self):
        cell = code_cell('05_Local_Model/local_model.ipynb', '# API model: the same reviews')
        calls = []
        def fake_ask(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(output_text='negative', usage=SimpleNamespace(input_tokens=100, output_tokens=10))
        ns = {'test': pd.DataFrame({'text': ['safe'] * 20}), 'ask': fake_ask, 'time': time,
              'MODEL': 'fake', 'PROMPT': 'label: ', 'parse_label': parse_label,
              'PRICE_INPUT_PER_M': 0.10, 'PRICE_OUTPUT_PER_M': 0.50, 'print': lambda *a, **k: None}
        exec(cell, ns)
        self.assertAlmostEqual(ns['api_cost'], (2000 * 0.10 + 200 * 0.50) / 1_000_000)
        self.assertTrue(all(call['use_cache'] is False for call in calls))
        self.assertEqual(ns['test']['api'].tolist(), ['negative'] * 20)


if __name__=='__main__':unittest.main()
