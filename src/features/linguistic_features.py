"""
[3] Feature Extraction Engine - 언어 지표 유틸리티

탐지 로직(간투어/반복/비문)은 detectors/ 패키지와
linguistic_error_extractor.py로 이전되었다. 이 파일에는 탐지와
무관한 유틸리티만 남긴다:
  - validate_from_labels(): AI-Hub 라벨(script_tag_txt)에서 REP/WR/FIL
    태그를 직접 파싱 -> detectors/ 결과와 대조할 정답(ground truth)
  - compute_ttr(), compute_speech_rate(): 어휘 다양성/발화속도 계산
"""

import re
from konlpy.tag import Okt

_okt = Okt()

TAG_PATTERN = re.compile(r"<(REP|WR|FIL)>(.*?)</\1>")


def validate_from_labels(script_tag_txt: str) -> dict:
    """AI-Hub script_tag_txt에서 태그 개수를 세어 라벨(repeat_cnt 등)과 대조 검증"""
    counts = {"REP": 0, "WR": 0, "FIL": 0}
    for tag_type, _content in TAG_PATTERN.findall(script_tag_txt):
        counts[tag_type] += 1
    return {
        "repeat_cnt_label": counts["REP"],
        "wrong_cnt_label": counts["WR"],
        "filler_cnt_label": counts["FIL"],
    }


def compute_ttr(text: str) -> float:
    """Type-Token Ratio: 어휘 다양성. 형태소(명사/동사/형용사) 기준으로 계산."""
    morphs = [
        word for word, pos in _okt.pos(text)
        if pos in ("Noun", "Verb", "Adjective")
    ]
    if not morphs:
        return 0.0
    return len(set(morphs)) / len(morphs)


def compute_speech_rate(word_cnt: int, duration_sec: float) -> dict:
    """다이어그램의 WPM(분당 어절 수), Articulation Rate(초당 어절 수)"""
    if duration_sec <= 0:
        return {"wpm": 0.0, "articulation_rate": 0.0}
    return {
        "wpm": word_cnt / (duration_sec / 60.0),
        "articulation_rate": word_cnt / duration_sec,
    }