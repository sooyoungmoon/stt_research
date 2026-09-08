import whisper
import librosa
import parselmouth
import numpy as np
import json
import re
from jiwer import wer as jiwer_wer, cer as jiwer_cer

# ===== 설정 =====
# AUDIO_PATH = "./data/test_ko.wav"
AUDIO_PATH = "./data/A00_S01_F_C_01_030_02_WA_MO.wav"
LABEL_PATH = "./data/A00_S01_F_C_01_030_02_WA_MO_presentation.json"
FILLER_WORDS = ["어", "음", "그", "저", "이제", "그니까", "약간"]  # 한국어 대표 필러 (프로젝트별로 조정 필요)


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


def compute_voice_quality(audio_path):
    """Jitter / Shimmer — 발성의 미세한 불규칙성(긴장·불안정성 관련 지표)"""
    snd = parselmouth.Sound(audio_path)
    point_process = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)", 75, 500)

    try:
        jitter = parselmouth.praat.call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        shimmer = parselmouth.praat.call(
            [snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6
        )
    except Exception:
        # 무음이 많거나 유성음 검출이 안 되면 계산 실패 가능
        jitter, shimmer = None, None

    return {
        "jitter_local": round(jitter, 5) if jitter is not None else None,
        "shimmer_local": round(shimmer, 5) if shimmer is not None else None,
    }


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


def compute_energy_features(audio_path):
    """음량(에너지) 관련 지표"""
    y, sr = librosa.load(audio_path, sr=16000)
    rms = librosa.feature.rms(y=y)[0]

    return {
        "rms_mean": round(float(rms.mean()), 4),
        "rms_std": round(float(rms.std()), 4),
    }


def compute_filler_features(text, filler_words=FILLER_WORDS):
    """필러(간투사) 빈도 — STT 텍스트 기반 단순 키워드 매칭
    주의: Whisper가 필러를 누락하는 경우가 많아 실제 빈도보다 과소 추정될 수 있음.
    정밀 분석을 원하면 별도 VAD+분류 모델이 필요 (본문 참고).
    """
    filler_count = 0
    matched = {}
    for word in filler_words:
        count = len(re.findall(word, text))
        if count > 0:
            matched[word] = count
        filler_count += count

    return {
        "filler_total_count": filler_count,
        "filler_breakdown": matched,
    }


def analyze_audio(audio_path, model_size="base"):
    """전체 파이프라인 실행 — STT + 모든 음향-언어 지표 추출"""
    print(f"[1/6] STT 변환 중... ({audio_path})")
    result = load_and_transcribe(audio_path, model_size)

    #print(f"STT 결과: {result['text']}")

    print("STT 정확도 계산")
    wer_value, cer_value = compute_stt_accuracy(result["text"], LABEL_PATH)
    #print(f"WER: {wer_value}, CER: {cer_value}")

    print("[2/6] 무음/휴지 지표 계산 중...")
    silence_feats = compute_silence_features(audio_path)

    print("[3/6] 발화속도 계산 중...")
    speech_rate_feats = compute_speech_rate(result, silence_feats["total_duration_sec"])

    print("[4/6] 피치(F0) 지표 계산 중...")
    pitch_feats = compute_pitch_features(audio_path)

    print("[5/6] 음질(Jitter/Shimmer) 지표 계산 중...")
    voice_quality_feats = compute_voice_quality(audio_path)

    print("[6/6] 에너지 및 필러 지표 계산 중...")
    energy_feats = compute_energy_features(audio_path)
    filler_feats = compute_filler_features(result["text"])

    # 결과 통합
    output = {
        "audio_path": audio_path,
        "transcript": result["text"].strip(),
     
        "segments": [
            {"start": round(s["start"], 2), "end": round(s["end"], 2), "text": s["text"].strip()}
            for s in result["segments"]
        ],
        "wer": wer_value,
        "cer": cer_value,
        "speech_rate": speech_rate_feats,
        "pitch": pitch_feats,
        "voice_quality": voice_quality_feats,
        "silence_pause": silence_feats,
        "energy": energy_feats,
        "filler": filler_feats,
    }
    return output


if __name__ == "__main__":
    result = analyze_audio(AUDIO_PATH, model_size="base")

    print("\n" + "=" * 50)
    print("=== 전체 결과 요약 ===")
    print("=" * 50)
    #print(f"인식 텍스트: {result['transcript']}")    
    print(f"[정확도]        WER: {result['wer']}, CER: {result['cer']}")
    print(f"\n[발화속도]     SPM(전체): {result['speech_rate']['spm_total']}  "
          f"조음속도: {result['speech_rate']['articulation_rate_spm']}")
    print(f"[피치]         평균: {result['pitch']['f0_mean_hz']}Hz  "
          f"범위: {result['pitch']['f0_range_hz']}Hz")
    print(f"[음질]         Jitter: {result['voice_quality']['jitter_local']}  "
          f"Shimmer: {result['voice_quality']['shimmer_local']}")
    print(f"[무음/휴지]    무음비율: {result['silence_pause']['silence_ratio']}  "
          f"pause 횟수: {result['silence_pause']['num_pauses']}")
    print(f"[필러]         총 {result['filler']['filler_total_count']}회  "
          f"{result['filler']['filler_breakdown']}")

    # 세그먼트 정보 출력
    print("\n[세그먼트 정보]")
    # 총 세그먼트 수 출력
    print(f"총 세그먼트 수: {len(result['segments'])}")
    for segment in result["segments"]:
        print(f"[세그먼트] 시작: {segment['start']}  끝: {segment['end']}  텍스트: {segment['text']}")


    # JSON으로 저장 — 여러 샘플을 모아서 데이터셋 구축할 때 사용
    with open("acoustic_linguistic_features.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("\n결과가 acoustic_linguistic_features.json 에 저장되었습니다.")