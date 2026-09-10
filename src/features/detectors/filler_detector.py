"""
FillerDetector: 간투어(filled pause) 탐지 — 사전 매칭 기반.

한계: "그"처럼 지시대명사와 간투사 양쪽으로 쓰이는 단어는 문맥 구분이
안 되어 오탐 가능성이 있음. 정밀도를 높이려면 전후 무음(pause) 여부를
함께 보는 방식으로 확장 필요 (현재는 텍스트만으로 판단).
"""

from .base_detector import BaseDetector


class FillerDetector(BaseDetector):
    DEFAULT_LEXICON = {"어", "음", "그", "저", "이제", "막", "그니까", "뭐랄까", "그러니까"}

    def __init__(self, lexicon: set[str] | None = None):
        self.lexicon = lexicon or self.DEFAULT_LEXICON

    def detect(self, tokens: list[str]) -> list[dict]:
        return [
            {"position": i, "token": tok}
            for i, tok in enumerate(tokens)
            if tok in self.lexicon
        ]

    def positions(self, tokens: list[str]) -> set[int]:
        """다른 탐지기가 필러와 겹치는 토큰을 제외할 때 참조하는 헬퍼."""
        return {item["position"] for item in self.detect(tokens)}
