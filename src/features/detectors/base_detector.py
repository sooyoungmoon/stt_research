"""
모든 언어 오류 탐지기가 따르는 공통 인터페이스.

새 탐지기(예: 향후 LM 기반 WrongWordDetector 재구현)를 추가할 때도
이 인터페이스만 지키면 LinguisticErrorFeatureExtractor 쪽 코드는
전혀 손댈 필요가 없다 (Strategy 패턴).
"""

from abc import ABC, abstractmethod


class BaseDetector(ABC):
    @abstractmethod
    def detect(self, tokens: list[str]) -> list[dict]:
        """토큰 리스트를 받아 탐지된 항목들을 리스트로 반환.
        각 항목은 최소한 'position' 또는 'start'/'end' 키를 포함해야 한다."""
        raise NotImplementedError

    def count(self, tokens: list[str]) -> int:
        return len(self.detect(tokens))
