"""
AudioPreprocessor 테스트 스크립트

사용법:
    python test_audio_preprocess.py "D:/path/to/sample.wav"

주의: src/preprocessing 폴더가 같은 프로젝트 안에 있어야 import가 됩니다.
      (pitch_pipeline/src 를 기준으로 실행하거나, 아래처럼 sys.path에 추가)
"""

import sys
import os
import whisper
import librosa
import parselmouth
import numpy as np
import json
import re
from jiwer import wer as jiwer_wer, cer as jiwer_cer



# src 폴더를 import 경로에 추가 (프로젝트 구조에 맞게 경로만 조정하면 됨)
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

# tests/ 의 부모 디렉토리(= stt_research/)를 import 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.audio_preprocess import AudioPreprocessor

def compute_silence_features(audio_path, top_db=30):
    """무음/휴지 관련 지표"""
    y, sr = librosa.load(audio_path, sr=16000)
    total_duration = len(y) / sr

    intervals = librosa.effects.split(y, top_db=top_db)
    speech_duration = sum((end - start) for start, end in intervals) / sr
    silence_duration = total_duration - speech_duration
    silence_ratio = silence_duration / total_duration if total_duration > 0 else 0

    # 개별 무음 구간 길이 분포 (구간 사이의 gap 계산)
    pause_lengths = []
    for i in range(1, len(intervals)):
        gap = (intervals[i][0] - intervals[i - 1][1]) / sr
        if gap > 0.1:  # 100ms 이상만 유의미한 pause로 간주
            pause_lengths.append(gap)

    return {
        "total_duration_sec": round(total_duration, 2),
        "speech_duration_sec": round(speech_duration, 2),
        "silence_ratio": round(silence_ratio, 3),
        "num_pauses": len(pause_lengths),
        "mean_pause_sec": round(np.mean(pause_lengths), 3) if pause_lengths else 0,
        "max_pause_sec": round(np.max(pause_lengths), 3) if pause_lengths else 0,
    }

def compute_speech_rate(result, total_duration_sec):
    """발화속도: SPM(음절/분) — 전체시간 기준 vs 발화구간 기준"""
    text = result["text"]
    korean_syllables = sum(1 for c in text if '가' <= c <= '힣')

    # 실제 발화 구간 시간(무음 제외) — 세그먼트 시간 합산
    speech_duration = sum(seg["end"] - seg["start"] for seg in result["segments"])

    spm_total = korean_syllables / (total_duration_sec / 60) if total_duration_sec > 0 else 0
    articulation_rate = korean_syllables / (speech_duration / 60) if speech_duration > 0 else 0

    return {
        "syllable_count": korean_syllables,
        "spm_total": round(spm_total, 2),          # 전체시간 기준 (긴장으로 인한 정지 포함)
        "articulation_rate_spm": round(articulation_rate, 2),  # 순수 발화속도
        "speech_duration_sec": round(speech_duration, 2),
    }


def compute_pitch_features(audio_path):
    """피치(F0) 관련 지표 — parselmouth(Praat) 사용"""
    snd = parselmouth.Sound(audio_path)
    pitch = snd.to_pitch()
    f0_values = pitch.selected_array['frequency']
    f0_values = f0_values[f0_values > 0]  # 무성구간 제외

    if len(f0_values) == 0:
        return {"f0_mean": None, "f0_std": None, "f0_range": None}

    return {
        "f0_mean_hz": round(float(f0_values.mean()), 2),
        "f0_std_hz": round(float(f0_values.std()), 2),
        "f0_range_hz": round(float(f0_values.max() - f0_values.min()), 2),
        "f0_min_hz": round(float(f0_values.min()), 2),
        "f0_max_hz": round(float(f0_values.max()), 2),
    }


def normalize(text: str) -> str:
    """비교를 위한 정규화: 마침표/쉼표 제거, 연속 공백 정리, 앞뒤 공백 제거"""
    text = re.sub(r"[.,!?]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_stt_accuracy(hypothesis, label_path):
    """STT 정확도 계산 (WER, CER)"""
    with open(label_path, "r", encoding="utf-8") as f:
        label_data = json.load(f)
    reference = label_data["presentation"]["presen_script"]
    hypothesis = normalize(hypothesis)   
    reference = normalize(reference)    

    print("Hypothesis text:", hypothesis)
    print("\n---\n")
    print("Reference text:", reference)

    # 단순화된 WER/CER 계산 (공백 기준)
    #hyp_words = hypothesis.split()
    #ref_words = reference.split()
    #wer = (len(ref_words) - sum(1 for h, r in zip(hyp_words, ref_words) if h == r)) / max(len(ref_words), 1)
    #cer = (len(reference) - sum(1 for h, r in zip(hypothesis, reference) if h == r)) / max(len(reference), 1)
    wer_value = jiwer_wer(reference, hypothesis)
    cer_value = jiwer_cer(reference, hypothesis)

    return wer_value, cer_value

def load_and_transcribe(audio_path, model_size="base"):
    """Whisper로 STT + 세그먼트별 타임스탬프 획득"""
    model = whisper.load_model(model_size)
    result = model.transcribe(audio_path, language="ko", verbose=False)
    return result



def main():
    if len(sys.argv) < 2:
        print("사용법: python test_audio_preprocess.py <오디오파일경로> <라벨파일경로>")
        sys.exit(1)

    audio_path = sys.argv[1]
    label_path = sys.argv[2]

    if not os.path.exists(audio_path):
        print(f"[오류] 파일을 찾을 수 없습니다: {audio_path}")
        sys.exit(1)
    if not os.path.exists(label_path):
        print(f"[오류] 파일을 찾을 수 없습니다: {label_path}")
        sys.exit(1)

    print(f"[2-A] AudioPreprocessor 테스트 시작")

    print(f"(1) AudioPreprocessor 초기화 중... (Silero-VAD 모델 다운로드/로드)")
    preprocessor = AudioPreprocessor(target_sr=16000, long_pause_threshold_sec=5.0)

    print(f"(2) 오디오 로드 및 16kHz 리샘플링: {audio_path}")
    waveform = preprocessor.load_and_resample(audio_path)
    duration_sec = waveform.shape[0] / preprocessor.target_sr
    print(f"      -> 로드 완료. shape={tuple(waveform.shape)}, 길이={duration_sec:.2f}초")

    print(f"(3) 무음(pause) 구간 탐지 중...")
    pauses = preprocessor.detect_pauses(waveform)
    metrics = preprocessor.pause_metrics(waveform)

    print("\n=== 개별 무음 구간 ===")
    if not pauses:
        print("  탐지된 무음 구간 없음")
    for i, p in enumerate(pauses, 1):
        marker = " <- Long Pause" if p.duration >= preprocessor.long_pause_threshold_sec else ""
        print(f"  {i:2d}. {p.start_sec:6.2f}s ~ {p.end_sec:6.2f}s  (길이 {p.duration:.2f}s){marker}")

    print("\n=== 집계 지표 (pause_metrics) ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    print(f"[2-B] STT 테스트 시작")

    print(f"(1) STT 변환 중... ({audio_path})")
    result = load_and_transcribe(audio_path, model_size = "base")
    
   
    
    print("(2) STT 정확도 계산")
    wer_value, cer_value = compute_stt_accuracy(result["text"], label_path)
    print(f"WER: {wer_value}, CER: {cer_value}")


    print(f"[3] 특징 추출 시작")
    print(f"[3-1] 음향 지표 추출")
    # 피치 추출 
    pitch_feats = compute_pitch_features(audio_path)
    print(f"    -> 피치 특징: {pitch_feats}")


    print(f"[3-2] 언어 & 유창성 지표")
    silence_feats = compute_silence_features(audio_path)

    speech_rate_feats =  compute_speech_rate(result, silence_feats["total_duration_sec"])
    print(f"    -> 언어 & 유창성 특징: {silence_feats}")
    print(f"    -> 발화속도 특징: {speech_rate_feats}")

if __name__ == "__main__":
    main()
