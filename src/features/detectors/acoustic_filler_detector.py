"""
AcousticFillerDetector: VAD 발화구간 vs Whisper 단어 타임스탬프 대조 기반
필러(간투어) 탐지.

주의: 이 클래스는 BaseDetector를 상속하지 않는다. BaseDetector.detect()는
'토큰 리스트'를 입력으로 가정하는데, 이 탐지기는 텍스트가 아니라
시간 정보(VAD 구간 + Whisper word timestamp)를 입력으로 쓰기 때문이다.
같은 인터페이스로 억지로 맞추면 오히려 혼란을 준다 — 대신 메서드명을
detect_from_timestamps()로 명확히 구분한다.

원리:
  VAD는 "사람이 소리를 냈다"고 판단한 구간을 알려주고, Whisper는 그중
  실제로 '단어'로 인식한 부분만 타임스탬프를 준다. 두 정보를 겹쳐서
  "VAD는 발화라고 했는데 Whisper가 아무 단어도 인식 못 한 구간"을
  찾으면, 그게 곧 Whisper가 전사에서 누락시킨 필러/머뭇거림 후보다.
"""

from dataclasses import dataclass


@dataclass
class AcousticFillerCandidate:
    start_sec: float
    end_sec: float

    @property
    def duration(self) -> float:
        return self.end_sec - self.start_sec


class AcousticFillerDetector:
    def __init__(self, min_gap_sec: float = 0.15, max_gap_sec: float = 1.5):
        """
        min_gap_sec: 이보다 짧은 미인식 구간은 정렬 오차(alignment noise)로
                     보고 무시한다. 너무 낮게 잡으면 오탐이 급증한다.
        max_gap_sec: 이보다 긴 미인식 구간은 필러라기보다 STT가 통째로
                     놓친 발화(다른 문제)일 가능성이 높아 별도 검토 대상으로
                     분리한다 (필러 카운트에는 포함하지 않음).
        """
        self.min_gap_sec = min_gap_sec
        self.max_gap_sec = max_gap_sec

    def detect_from_timestamps(
        self,
        vad_segments: list[tuple[float, float]],
        word_timestamps: list[dict],
    ) -> list[AcousticFillerCandidate]:
        """
        vad_segments: [(start_sec, end_sec), ...] — AudioPreprocessor.get_speech_segments() 결과
        word_timestamps: [{"word": str, "start": float, "end": float}, ...] — Whisper 단어별 타임스탬프
        """
        words_sorted = sorted(word_timestamps, key=lambda w: w["start"])
        candidates: list[AcousticFillerCandidate] = []

        for seg_start, seg_end in vad_segments:
            cursor = seg_start
            seg_words = [w for w in words_sorted if w["end"] > seg_start and w["start"] < seg_end]

            for w in seg_words:
                w_start = max(w["start"], seg_start)
                w_end = min(w["end"], seg_end)
                gap = w_start - cursor
                if gap >= self.min_gap_sec:
                    candidates.append(AcousticFillerCandidate(cursor, w_start))
                cursor = max(cursor, w_end)

            trailing_gap = seg_end - cursor
            if trailing_gap >= self.min_gap_sec:
                candidates.append(AcousticFillerCandidate(cursor, seg_end))

        # min_gap_sec ~ max_gap_sec 범위만 "필러 후보"로 채택.
        # max_gap_sec 초과분은 별도 검토용으로 남기고 필러 카운트에서 제외.
        return [c for c in candidates if c.duration <= self.max_gap_sec]

    def unmatched_long_gaps(
        self,
        vad_segments: list[tuple[float, float]],
        word_timestamps: list[dict],
    ) -> list[AcousticFillerCandidate]:
        """max_gap_sec를 초과하는 미인식 구간 — 필러가 아니라 STT가
        아예 못 알아들은 발화일 가능성이 높으므로 별도로 확인 권장."""
        all_gaps = self._raw_gaps(vad_segments, word_timestamps)
        return [g for g in all_gaps if g.duration > self.max_gap_sec]

    def _raw_gaps(self, vad_segments, word_timestamps) -> list[AcousticFillerCandidate]:
        words_sorted = sorted(word_timestamps, key=lambda w: w["start"])
        gaps = []
        for seg_start, seg_end in vad_segments:
            cursor = seg_start
            seg_words = [w for w in words_sorted if w["end"] > seg_start and w["start"] < seg_end]
            for w in seg_words:
                w_start = max(w["start"], seg_start)
                w_end = min(w["end"], seg_end)
                if w_start - cursor >= self.min_gap_sec:
                    gaps.append(AcousticFillerCandidate(cursor, w_start))
                cursor = max(cursor, w_end)
            if seg_end - cursor >= self.min_gap_sec:
                gaps.append(AcousticFillerCandidate(cursor, seg_end))
        return gaps


def extract_word_timestamps(whisper_result: dict) -> list[dict]:
    """openai-whisper의 transcribe(..., word_timestamps=True) 결과에서
    segments[].words[] 를 평탄화해 [{"word", "start", "end"}, ...] 로 변환."""
    words = []
    for segment in whisper_result.get("segments", []):
        for w in segment.get("words", []):
            words.append({"word": w["word"].strip(), "start": w["start"], "end": w["end"]})
    return words