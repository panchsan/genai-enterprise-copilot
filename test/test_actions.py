from app.nodes.query_understanding import analyze_query
from app.nodes.validate_retrieval import validate_retrieval


def test_explicit_action_overrides_detected_action():
    state = {
        "query": "What is the hybrid work policy?",
        "action": "compare_documents",
        "filters": {},
    }
    result = analyze_query(state)
    assert result["action"] == "compare_documents"


def test_detect_summary_action_when_not_explicit():
    state = {
        "query": "Summarize hr_policy.txt",
        "filters": {},
    }
    result = analyze_query(state)
    assert result["action"] == "summarize_document"


def test_validate_compare_requires_two_sources():
    state = {
        "action": "compare_documents",
        "retrieval_status": "found",
        "retrieved_docs": [
            {
                "page_content": "x",
                "metadata": {"source": "hr_policy.txt"},
            }
        ],
        "top_score": 0.5,
        "retrieval_scores": [0.5],
        "filters": {},
    }
    result = validate_retrieval(state)
    assert result["retrieval_decision"] == "no_docs"


def test_validate_qa_grounded_when_score_good():
    state = {
        "action": "qa",
        "retrieval_status": "found",
        "retrieved_docs": [
            {
                "page_content": "x",
                "metadata": {"source": "hr_policy.txt"},
            }
        ],
        "top_score": 0.6,
        "retrieval_scores": [0.6],
        "filters": {},
    }
    result = validate_retrieval(state)
    assert result["retrieval_decision"] == "grounded"