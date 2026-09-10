"""
LinguisticErrorFeatureExtractor: FillerDetector, RepetitionDetector,
WrongWordDetector를 묶어 [3] Feature Extraction Engine의 언어 지표 산출을
하나의 진입점으로 제공하는 Facade.

개별 탐지기 교체(예: WrongWordDetector를 LM 기반으로 재구현)는 이 클래스를
수정하지 않고도 가능하다 — 생성자에서 주입만 바꾸면 된다.
"""

from dataclasses import dataclass, field

from .detectors import FillerDetector, RepetitionDetector, WrongWordDetector
from .detectors.base_detector import BaseDetector


@dataclass
class LinguisticErrorFeatures:
    fillers: list[dict] = field(default_factory=list)
    repetitions: list[dict] = field(default_factory=list)
    wrong_words: list[dict] = field(default_factory=list)

    @property
    def filler_cnt(self) -> int:
        return len(self.fillers)

    @property
    def repetition_cnt(self) -> int:
        return len(self.repetitions)

    @property
    def wrong_word_cnt(self) -> int:
        return len(self.wrong_words)

    def as_dict(self) -> dict:
        """모델 학습용 피처 테이블에 바로 넣을 수 있는 평탄화된(flat) 형태."""
        return {
            "filler_cnt": self.filler_cnt,
            "repeat_cnt": self.repetition_cnt,
            "wrong_cnt": self.wrong_word_cnt,
        }


class LinguisticErrorFeatureExtractor:
    def __init__(
        self,
        filler_detector: BaseDetector | None = None,
        repetition_detector: BaseDetector | None = None,
        wrong_word_detector: BaseDetector | None = None,
    ):
        self.filler_detector = filler_detector or FillerDetector()
        # RepetitionDetector는 FillerDetector 결과와 겹치지 않도록 내부에서 참조하므로 함께 주입
        self.repetition_detector = repetition_detector or RepetitionDetector(filler_detector=self.filler_detector)
        self.wrong_word_detector = wrong_word_detector or WrongWordDetector()

    def extract(self, text: str) -> LinguisticErrorFeatures:
        tokens = text.split()
        return LinguisticErrorFeatures(
            fillers=self.filler_detector.detect(tokens),
            repetitions=self.repetition_detector.detect(tokens),
            wrong_words=self.wrong_word_detector.detect(tokens),
        )
