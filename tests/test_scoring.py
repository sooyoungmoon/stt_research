import io
import re
import numpy as np
import soundfile as sf
import whisper
from datasets import load_dataset, Audio
from jiwer import wer, cer

# ===== 1단계: Zeroth-Korean 샘플 10개를 이어붙여 긴 오디오 생성 =====

ds = load_dataset("kresnik/zeroth_korean", split="test", streaming=True)
ds = ds.cast_column("audio", Audio(decode=False))

audio_chunks = []
texts = []
for i, sample in enumerate(ds):
    if i >= 10:
        break
    audio_bytes = sample["audio"]["bytes"]
    data, sr = sf.read(io.BytesIO(audio_bytes))
    audio_chunks.append(data)
    texts.append(sample["text"])

long_audio = np.concatenate(audio_chunks)
sf.write("test_ko_long.wav", long_audio, sr)

reference_text = " ".join(texts)  # 정답 전체를 하나의 문자열로

print("=== 정답 텍스트 ===")
print(reference_text)
print(f"\n총 길이: {len(long_audio) / sr:.1f}초\n")

# ===== 2단계: Whisper로 STT + 세그먼트별 타임스탬프 =====

model = whisper.load_model("medium")
result = model.transcribe("test_ko_long.wav", language="ko", verbose=False)

hypothesis_text = result["text"]

print("=== Whisper 인식 결과 (전체) ===")
print(hypothesis_text)
print()

print("=== 세그먼트별 타임스탬프 ===")
for seg in result["segments"]:
    print(f"[{seg['start']:.2f}s -> {seg['end']:.2f}s] {seg['text']}")

# ===== 3단계: WER / CER 평가 =====

def normalize(text: str) -> str:
    """비교를 위한 정규화: 마침표/쉼표 제거, 연속 공백 정리, 앞뒤 공백 제거"""
    text = re.sub(r"[.,!?]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

ref_norm = normalize(reference_text)
hyp_norm = normalize(hypothesis_text)

word_error_rate = wer(ref_norm, hyp_norm)
char_error_rate = cer(ref_norm, hyp_norm)

print("\n=== 평가 결과 ===")
print(f"정답(정규화):   {ref_norm}")
print(f"인식(정규화):   {hyp_norm}")
print(f"WER (단어 오류율): {word_error_rate * 100:.2f}%")
print(f"CER (음절 오류율): {char_error_rate * 100:.2f}%")