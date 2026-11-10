"""Offline regression checks: no real API client or network requests."""
import concurrent.futures
import json
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
from llm_course.retrieval import chunk_reviews
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


class NotebookPreparationTests(unittest.TestCase):
    def test_notebook_01_saves_redacted_copies_without_overwriting_sources(self):
        import contextlib
        import io
        path = Path(__file__).resolve().parents[1] / '01_Explore_Yelp_Data/explore_yelp_data.ipynb'
        notebook = json.loads(path.read_text())
        preparation = next(''.join(c['source']) for c in notebook['cells']
                           if c['cell_type'] == 'code' and '# Anonymize the review texts locally' in ''.join(c['source']))
        save = next(''.join(c['source']) for c in notebook['cells']
                    if c['cell_type'] == 'code' and '# Save prepared copies locally' in ''.join(c['source']))
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp)
            source = pd.DataFrame({'review_id':['local-id'], 'stars':[1], 'text':['John Smith was rude.']})
            source.to_csv(data / 'ikea_reviews.csv', index=False)
            examples = [{'text':'John Smith was helpful.', 'sentiment':'positive', 'note':'John Smith'}]
            (data / 'fewshot_examples.json').write_text(json.dumps(examples))
            originals = {name:(data / name).read_bytes() for name in ['ikea_reviews.csv', 'fewshot_examples.json']}
            ns = {'DATA':data, 'df':source, 'json':json,
                  'tqdm':lambda values,**kw:values,
                  'anonymize':lambda text:text.replace('John Smith','<REDACTED>')}
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(preparation, str(path), 'exec'), ns)
                exec(compile(save, str(path), 'exec'), ns)
            prepared = pd.read_csv(data / 'processed/ikea_reviews_anonymized.csv')
            self.assertEqual(prepared.loc[0,'text'], '<REDACTED> was rude.')
            self.assertEqual(prepared.loc[0,'review_id'], 'local-id')
            self.assertEqual(source.loc[0,'text'], 'John Smith was rude.')
            self.assertNotIn('John Smith', (data / 'processed/fewshot_examples_anonymized.json').read_text())
            for name, content in originals.items():
                self.assertEqual((data / name).read_bytes(), content)
            # A failed rerun leaves stale notebook variables: saving must still stop.
            outputs = {path:path.read_bytes() for path in (data / 'processed').iterdir()}
            def failed_detector(text):
                raise RuntimeError('missing detector')
            ns['anonymize'] = failed_detector
            with self.assertRaisesRegex(RuntimeError, 'missing detector'):
                exec(compile(preparation, str(path), 'exec'), ns)
            with self.assertRaisesRegex(RuntimeError, 'preparation'):
                exec(compile(save, str(path), 'exec'), ns)
            for output, content in outputs.items():
                self.assertEqual(output.read_bytes(), content)

    def test_later_notebooks_stop_when_prepared_files_are_missing(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp)
            # A raw file must never be used as an automatic fallback.
            (data / 'ikea_reviews.csv').write_text('review_id,text\nraw-id,raw personal information\n')
            for folder in sorted(root.glob('0[2-5]_*')):
                path = next(folder.glob('*.ipynb'))
                notebook = json.loads(path.read_text())
                load = next(''.join(c['source']) for c in notebook['cells']
                            if c['cell_type'] == 'code' and "reviews_path = DATA / 'processed/" in ''.join(c['source']))
                with self.subTest(notebook=path.name), self.assertRaisesRegex(FileNotFoundError, '01_Explore_Yelp_Data'):
                    exec(compile(load, str(path), 'exec'), {'DATA':data, 'pd':pd, 'json':json})


class NotebookCostTests(unittest.TestCase):
    def test_timing_cell_accounts_for_reads_writes_and_reasoning_once(self):
        path = Path(__file__).resolve().parents[1] / '05_Local_Model/local_model.ipynb'
        notebook = json.loads(path.read_text())
        cell = next(''.join(c['source']) for c in notebook['cells']
                    if c['cell_type'] == 'code' and '# Uncached requests:' in ''.join(c['source']))
        result = SimpleNamespace(status='completed', output_text='Overall: negative.', usage=SimpleNamespace(
            input_tokens=100, output_tokens=10,
            input_tokens_details=SimpleNamespace(cached_tokens=40, cache_write_tokens=20)))
        namespace = {'test':pd.DataFrame({'text':['safe']*20}), 'ask':lambda **kw:result,
                     'time':time, 'MODEL':'fake', 'PROMPT':'label: ', 'parse_label':parse_label,
                     'PRICE_INPUT_PER_M':0.10, 'PRICE_OUTPUT_PER_M':0.50}
        exec(compile(cell, str(path), 'exec'), namespace)
        self.assertAlmostEqual(namespace['api_cost'], 0.000238)
        self.assertEqual(namespace['output_tokens'], 200)  # Includes reasoning already.


class CharTokenizer:
    # One character per token makes the boundary/overlap invariant inspectable.
    def encode(self,text,add_special_tokens=False):return list(text)+([0,1] if add_special_tokens else [])
    def num_special_tokens_to_add(self,pair=False):return 2
    def __call__(self,text,**kwargs):return {'offset_mapping':[(i,i+1) for i in range(len(text))]}


class RetrievalTests(unittest.TestCase):
    def test_every_character_is_covered_and_no_passage_exceeds_budget(self):
        tokenizer=CharTokenizer();text='0123456789'*20
        chunks=chunk_reviews(pd.DataFrame([{'review_id':'private-id','text':text}]),tokenizer,max_tokens=30,overlap=5)
        self.assertGreater(len(chunks),1)
        self.assertTrue(chunks.citation_id.is_unique)
        self.assertTrue(all(len(tokenizer.encode('passage: '+t,add_special_tokens=True))<=30 for t in chunks.text))
        rebuilt=chunks.iloc[0].text+''.join(t[5:] for t in chunks.text.iloc[1:])
        self.assertEqual(rebuilt,text)
        self.assertTrue(all('private-id' not in c for c in chunks.citation_id))


    def test_retrieval_keeps_distinct_source_reviews(self):
        import numpy as np
        path = Path(__file__).resolve().parents[1] / '04_Mini_RAG/mini_rag.ipynb'
        notebook = json.loads(path.read_text())
        source = next(''.join(c['source']) for c in notebook['cells']
                      if c['cell_type'] == 'code' and 'def retrieve(' in ''.join(c['source']))
        function_source = source.split('# Our three questions')[0]
        chunks = pd.DataFrame({'review_id':['A','A','B'], 'text':['part1','part2','other']})
        namespace = {'pd':pd, 'chunks':chunks, 'tokenize':str.split,
                     'bm25':SimpleNamespace(get_scores=lambda q:[3.,2.,1.]),
                     'model':SimpleNamespace(encode=lambda *a,**kw:np.array([[1.,0.]])),
                     'embeddings':np.array([[1.,0.],[1.,0.],[1.,0.]])}
        exec(compile(function_source, str(path), 'exec'), namespace)
        self.assertEqual(namespace['retrieve']('question', k=2)['review_id'].tolist(), ['A','B'])


if __name__=='__main__':unittest.main()
