"""Read-only BrowseComp-Plus retrieval using the official Anserini index."""

import json


class FixedCorpus:
    def __init__(self, index_path, tokenizer_path):
        import jnius_config
        jnius_config.add_options('-Xmx4g', '-XX:ActiveProcessorCount=2')
        from pyserini.pyclass import autoclass
        from transformers import AutoTokenizer

        searcher_type = autoclass('io.anserini.search.SimpleSearcher')
        self.searcher = searcher_type(str(index_path))
        self.num_docs = self.searcher.get_total_num_docs()
        if self.num_docs != 100195:
            raise ValueError('Index does not have the fixed corpus document count')
        self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)

    def prefix(self, text, limit):
        encoded = self.tokenizer(text, add_special_tokens=False, truncation=True,
                                 max_length=limit, return_offsets_mapping=True)
        if not encoded['offset_mapping']:
            return '', 0
        end = encoded['offset_mapping'][-1][1]
        result = text[:end]
        count = len(self.tokenizer.encode(result, add_special_tokens=False))
        # Token boundaries may split a multi-byte character. Preserve an exact
        # text prefix and verify the token cap after choosing a character end.
        while count > limit:
            result = result[:-1]
            count = len(self.tokenizer.encode(result, add_special_tokens=False))
        return result, count

    def raw_document(self, docid):
        document = self.searcher.doc(str(docid))
        if document is None:
            raise KeyError('Unknown fixed-corpus document ID')
        return json.loads(document.get('raw'))['contents']

    def search(self, query):
        if not isinstance(query, str) or not query.strip():
            raise ValueError('Search query must be a nonempty string')
        hits = self.searcher.search(query, 10)
        results = []
        for hit in hits:
            raw = json.loads(hit.lucene_document.get('raw'))
            snippet, count = self.prefix(raw['contents'], 512)
            results.append({'docid': hit.docid, 'url': 'bcp://' + hit.docid,
                            'score': float(hit.score), 'snippet': snippet,
                            'snippet_tokens': count})
        return {'results': results}

    def fetch(self, reference):
        if not isinstance(reference, str) or not reference.startswith('bcp://'):
            raise ValueError('Document reads require a bcp:// reference from search')
        docid = reference.removeprefix('bcp://')
        full = self.raw_document(docid)
        text, count = self.prefix(full, 8192)
        return {'docid': docid, 'url': reference, 'text': text,
                'document_tokens': count, 'truncated': len(text) < len(full)}
