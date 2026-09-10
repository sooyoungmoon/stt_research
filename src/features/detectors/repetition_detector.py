"""
RepetitionDetector: 반복(자기수정형 disfluency) 탐지 — 규칙 기반.

4단계 규칙 (우선순위 순):
  1) exact_adjacent : 인접 토큰 완전 일치
  2) prefix_adjacent: 인접 토큰 중 뒤 토큰이 앞 토큰을 접두어로 포함 (자기수정 확장)
  3) phonetic       : 자모(jamo) 편집거리 근접 (STT 오인식/발음 유사 false start)
  4) windowed       : window 범위 내 동일 토큰 재등장 (비연속 반복)

FillerDetector와 겹치는 위치(1음절 필러 등)는 반복 탐지 후보에서 제외해
오탐을 줄인다.
"""

from jamo import h2j
import Levenshtein

from .base_detector import BaseDetector
from .filler_detector import FillerDetector


class RepetitionDetector(BaseDetector):
    def __init__(self, window: int = 4, phonetic_max_dist: int = 2, filler_detector: FillerDetector | None = None):
        self.window = window
        self.phonetic_max_dist = phonetic_max_dist
        self.filler_detector = filler_detector or FillerDetector()

    @staticmethod
    def _jamo_distance(a: str, b: str) -> int:
        return Levenshtein.distance(h2j(a), h2j(b))

    def detect(self, tokens: list[str]) -> list[dict]:
        filler_positions = self.filler_detector.positions(tokens)
        results: list[dict] = []
        used: set[int] = set()

        i = 0
        n = len(tokens)
        while i < n - 1:
            if i in used or i in filler_positions:
                i += 1
                continue

            cur = tokens[i]
            matched = False

            # 1) 완전 일치
            if tokens[i + 1] == cur:
                results.append({"type": "exact_adjacent", "start": i, "end": i + 1, "span": tokens[i:i + 2]})
                used.update([i, i + 1])
                matched = True

            # 2) 접두어 일치
            elif len(cur) >= 2 and (i + 1) not in filler_positions and tokens[i + 1].startswith(cur):
                results.append({"type": "prefix_adjacent", "start": i, "end": i + 1, "span": tokens[i:i + 2]})
                used.update([i, i + 1])
                matched = True

            # 3) 자모 편집거리 근접
            elif (
                len(cur) >= 2
                and (i + 1) not in filler_positions
                and len(tokens[i + 1]) >= 2
                and self._jamo_distance(cur, tokens[i + 1]) <= self.phonetic_max_dist
            ):
                results.append({"type": "phonetic", "start": i, "end": i + 1, "span": tokens[i:i + 2]})
                used.update([i, i + 1])
                matched = True

            # 4) 근접 윈도우 내 재등장 (비연속)
            if not matched:
                for gap in range(2, self.window + 1):
                    j = i + gap
                    if j >= n:
                        break
                    if tokens[j] == cur and j not in filler_positions:
                        end = j
                        if end + 1 < n and tokens[end + 1].startswith(cur):
                            end += 1
                        results.append({"type": "windowed", "start": i, "end": end, "span": tokens[i:end + 1]})
                        used.update(range(i, end + 1))
                        matched = True
                        break

            i += 1

        return results
