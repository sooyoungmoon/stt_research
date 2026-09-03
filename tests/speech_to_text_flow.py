import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import whisper

# 한글 폰트 설정 (Windows 기준 - 없으면 그래프 제목이 깨짐)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ===== 오디오 로드 =====
AUDIO_PATH = "test_ko.wav"
y, sr = librosa.load(AUDIO_PATH, sr=16000)  # 16kHz로 리샘플링
print(f"오디오 길이: {len(y)/sr:.2f}초, 샘플 수: {len(y)}, 샘플링레이트: {sr}Hz")

# 공통 파라미터 (Whisper와 동일하게 맞춤)
N_FFT = 400       # 25ms @ 16kHz
# N_FFT = 800 # 50ms @ 16kHz
#N_FFT = 2048  # 128ms @ 16kHz


HOP_LENGTH = 160  # 10ms
N_MELS = 80

fig, axes = plt.subplots(6, 1, figsize=(12, 18))

# ===== 1단계: 원본 파형 =====
librosa.display.waveshow(y, sr=sr, ax=axes[0], color='steelblue')
axes[0].set_title("1. 원본 음성 파형 (샘플링·양자화 완료 상태)")
axes[0].set_xlabel("시간 (초)")
axes[0].set_ylabel("진폭")

# ===== 2단계: 윈도잉 효과 비교 (한 프레임만 확대) =====
frame_start = len(y) // 3
frame = y[frame_start:frame_start + N_FFT]
window = np.hamming(N_FFT)
windowed_frame = frame * window

axes[1].plot(frame, label="윈도잉 전 (raw frame)", alpha=0.6)
axes[1].plot(windowed_frame, label="윈도잉 후 (Hamming)", linewidth=2)
axes[1].set_title("2. 프레이밍·윈도잉 (25ms 프레임 1개 확대)")
axes[1].set_xlabel("샘플")
axes[1].legend()

# ===== 3단계: FFT → 스펙트로그램 =====
S = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH, window='hamming')
power_spec = np.abs(S) ** 2
db_spec = librosa.power_to_db(power_spec, ref=np.max)

img1 = librosa.display.specshow(
    db_spec, sr=sr, hop_length=HOP_LENGTH,
    x_axis='time', y_axis='hz', ax=axes[2], cmap='magma'
)
axes[2].set_title("3. FFT 스펙트로그램 (선형 주파수 스케일)")
fig.colorbar(img1, ax=axes[2], format="%+2.0f dB")

# ===== 4단계: Mel 필터뱅크 자체 모양 =====
mel_filters = librosa.filters.mel(sr=sr, n_fft=N_FFT, n_mels=N_MELS)
for i in range(0, N_MELS, 8):  # 8개마다 하나씩만 그려서 안 겹치게
    axes[3].plot(mel_filters[i])
axes[3].set_title("4. Mel 필터뱅크 모양 (일부 필터만 표시, 저주파일수록 촘촘)")
axes[3].set_xlabel("FFT bin")
axes[3].set_ylabel("필터 가중치")

# ===== 5단계: Mel 스펙트로그램 + 로그 (Whisper 방식) =====
mel_spec = librosa.feature.melspectrogram(
    y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
)
log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

img2 = librosa.display.specshow(
    log_mel_spec, sr=sr, hop_length=HOP_LENGTH,
    x_axis='time', y_axis='mel', ax=axes[4], cmap='magma'
)
axes[4].set_title("5. Log-Mel 스펙트로그램 (Whisper 입력과 동일한 방식)")
fig.colorbar(img2, ax=axes[4], format="%+2.0f dB")

# ===== 6단계: MFCC (DCT까지 적용, 전통 방식) =====
mfcc = librosa.feature.mfcc(
    y=y, sr=sr, n_mfcc=13, n_fft=N_FFT, hop_length=HOP_LENGTH
)
img3 = librosa.display.specshow(
    mfcc, sr=sr, hop_length=HOP_LENGTH, x_axis='time', ax=axes[5], cmap='coolwarm'
)
axes[5].set_title("6. MFCC (DCT 적용, 13차원 - 전통 방식)")
axes[5].set_ylabel("MFCC 계수")
fig.colorbar(img3, ax=axes[5])

plt.tight_layout()
plt.savefig("stt_pipeline_stages.png", dpi=150)
plt.show()

# ===== 참고: Whisper 실제 함수와 shape 비교 =====
print("\n=== Shape 비교 ===")
print(f"librosa Mel 스펙트로그램:  {mel_spec.shape}  (n_mels={N_MELS}, 프레임수)")
print(f"librosa MFCC (DCT 포함):   {mfcc.shape}  (n_mfcc=13, 프레임수)")

audio_w = whisper.load_audio(AUDIO_PATH)
audio_w = whisper.pad_or_trim(audio_w)
mel_w = whisper.log_mel_spectrogram(audio_w)
print(f"Whisper log-Mel:           {tuple(mel_w.shape)}  (80채널, 30초 고정=3000프레임)")