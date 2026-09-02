import whisper
from datasets import load_dataset, Audio
import soundfile as sf
import io

ds = load_dataset("kresnik/zeroth_korean", split="test", streaming=True)
ds = ds.cast_column("audio", Audio(decode=False))
sample = next(iter(ds))

# 원본 바이트를 soundfile로 직접 디코딩
audio_bytes = sample["audio"]["bytes"]
data, samplerate = sf.read(io.BytesIO(audio_bytes))
sf.write("test_ko.wav", data, samplerate)

print("정답 텍스트:", sample["text"])

model = whisper.load_model("medium")
result = model.transcribe("test_ko.wav", language="ko", verbose=False)
print("전체 텍스트:", result["text"])
print()


# 앞쪽 30초만 처리
""" audio = whisper.load_audio("test_ko.wav")
audio = whisper.pad_or_trim(audio)
mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)
_, probs = model.detect_language(mel)
print(f"Detected language: {max(probs, key=probs.get)}")
options = whisper.DecodingOptions()
result = whisper.decode(model, mel, options) 
print(result.text)
"""

