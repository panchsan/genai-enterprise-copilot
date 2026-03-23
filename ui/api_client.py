import requests
from typing import Any, Dict, Optional


class RAGApiClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def chat(
        self,
        query: str,
        session_id: str,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload = {
            "query": query,
            "session_id": session_id,
            "filters": filters or {},
        }

        response = requests.post(
            f"{self.base_url}/chat",
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        return response.json()

    def get_history(self, session_id: str) -> Dict[str, Any]:
        response = requests.get(
            f"{self.base_url}/history/{session_id}",
            timeout=60,
        )
        response.raise_for_status()
        return response.json()

    def list_sessions(self) -> Dict[str, Any]:
        response = requests.get(
            f"{self.base_url}/sessions",
            timeout=60,
        )
        response.raise_for_status()
        return response.json()
    
    def delete_session(self, session_id: str):
        response = requests.delete(
            f"{self.base_url}/sessions/{session_id}",
            timeout=30,
        )
        response.raise_for_status()
        return response.json()