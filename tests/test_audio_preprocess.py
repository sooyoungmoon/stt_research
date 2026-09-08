"""
AudioPreprocessor 테스트 스크립트

사용법:
    python test_audio_preprocess.py "D:/path/to/sample.wav"

주의: src/preprocessing 폴더가 같은 프로젝트 안에 있어야 import가 됩니다.
      (pitch_pipeline/src 를 기준으로 실행하거나, 아래처럼 sys.path에 추가)
"""

import sys
import os

# src 폴더를 import 경로에 추가 (프로젝트 구조에 맞게 경로만 조정하면 됨)
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

# tests/ 의 부모 디렉토리(= stt_research/)를 import 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.audio_preprocess import AudioPreprocessor


def main():
    if len(sys.argv) < 2:
        print("사용법: python test_audio_preprocess.py <오디오파일경로>")
        sys.exit(1)

    audio_path = sys.argv[1]
    if not os.path.exists(audio_path):
        print(f"[오류] 파일을 찾을 수 없습니다: {audio_path}")
        sys.exit(1)

    print(f"[1/3] AudioPreprocessor 초기화 중... (Silero-VAD 모델 다운로드/로드)")
    preprocessor = AudioPreprocessor(target_sr=16000, long_pause_threshold_sec=5.0)

    print(f"[2/3] 오디오 로드 및 16kHz 리샘플링: {audio_path}")
    waveform = preprocessor.load_and_resample(audio_path)
    duration_sec = waveform.shape[0] / preprocessor.target_sr
    print(f"      -> 로드 완료. shape={tuple(waveform.shape)}, 길이={duration_sec:.2f}초")

    print(f"[3/3] 무음(pause) 구간 탐지 중...")
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


if __name__ == "__main__":
    main()
