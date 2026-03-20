from collections import defaultdict
from typing import Dict, List


class InMemorySessionStore:
    def __init__(self):
        self.store: Dict[str, List[dict]] = defaultdict(list)

    def get_history(self, session_id: str) -> List[dict]:
        return self.store[session_id]

    def append(self, session_id: str, role: str, content: str) -> None:
        self.store[session_id].append({
            "role": role,
            "content": content
        })


memory_store = InMemorySessionStore()