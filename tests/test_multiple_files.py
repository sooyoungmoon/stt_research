import io
import numpy as np
import soundfile as sf
import whisper
from datasets import load_dataset, Audio

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

print("=== 정답 텍스트 (문장 구분: /) ===")
print(" / ".join(texts))
print(f"\n총 길이: {len(long_audio) / sr:.1f}초\n")

# ===== 2단계: Whisper로 STT + 세그먼트별 타임스탬프 =====

model = whisper.load_model("base")
result = model.transcribe("test_ko_long.wav", language="ko", verbose=False)

print("=== Whisper 인식 결과 (전체) ===")
print(result["text"])
print()

print("=== 세그먼트별 타임스탬프 ===")
for seg in result["segments"]:
    print(f"[{seg['start']:.2f}s -> {seg['end']:.2f}s] {seg['text']}")