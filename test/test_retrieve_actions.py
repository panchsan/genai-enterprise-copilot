from types import SimpleNamespace

from app.nodes.retrieve import retrieve


class FakeDoc:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata


class FakeVectorDB:
    def __init__(self, results_by_filter):
        self.results_by_filter = results_by_filter

    def similarity_search_with_score(self, query, k, filter=None):
        key = str(filter)
        return self.results_by_filter.get(key, [])


def test_answer_by_source_requires_source():
    vectordb = FakeVectorDB(results_by_filter={})
    state = {
        "query": "What does this say?",
        "action": "answer_by_source",
        "filters": {},
        "target_sources": [],
    }
    result = retrieve(state, vectordb)
    assert result["retrieval_status"] == "missing_required_source"