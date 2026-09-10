"""
WrongWordDetector: 비문/어색한 단어 탐지 — 1단계 규칙 기반 베이스라인.

방식: KoNLPy Okt 형태소 분석기가 토큰을 분석하지 못하거나 'Unknown'으로
태깅하는 경우를 "사전에 없는/오인식된 단어" 후보로 플래그한다.

명확한 한계 (지난 대화에서 정리한 내용 그대로):
  - 철자/미등록어 오류만 잡는다. "성냥약은"처럼 실제 단어처럼 보이지만
    문맥상 어색한 malapropism은 이 규칙으로 잡히지 않을 수 있다.
  - 문맥 의존적 오류(문법은 맞지만 뜻이 안 맞는 경우)는 원천적으로
    탐지 불가 — 이 부분은 KoBERT/KoGPT2 perplexity 기반으로 교체 예정
    (지난 대화의 '2단계: 비문만 LM 기반으로 선별 확장' 로드맵 참고).

이 클래스는 BaseDetector 인터페이스만 지키면 되므로, 향후 LM 기반
구현으로 교체할 때 LinguisticErrorFeatureExtractor 쪽 코드는 수정할
필요가 없다.
"""

from konlpy.tag import Okt

from .base_detector import BaseDetector


class WrongWordDetector(BaseDetector):
    UNKNOWN_POS_TAGS = {"Unknown", "Foreign", "Alpha"}

    def __init__(self):
        self._okt = Okt()

    def detect(self, tokens: list[str]) -> list[dict]:
        results = []
        position = 0
        for tok in tokens:
            pos_tags = self._okt.pos(tok)
            if any(tag in self.UNKNOWN_POS_TAGS for _, tag in pos_tags):
                results.append({"position": position, "token": tok, "reason": "unrecognized_by_morph_analyzer"})
            position += 1
        return results
