"""
[2-A] Audio Signal Preprocessing
- 16kHz 리샘플링
- Silero-VAD 기반 발화/무음(pause) 구간 탐지
- 다이어그램 기준: 5초 이상 무음(Long Pause) 횟수/시간 산출
"""

import numpy as np
import soundfile as sf
import torch
import torchaudio
from dataclasses import dataclass


@dataclass
class PauseSegment:
    start_sec: float
    end_sec: float

    @property
    def duration(self) -> float:
        return self.end_sec - self.start_sec


class AudioPreprocessor:
    def __init__(self, target_sr: int = 16000, long_pause_threshold_sec: float = 5.0):
        self.target_sr = target_sr
        self.long_pause_threshold_sec = long_pause_threshold_sec
        # Silero-VAD 로드 (torch.hub 캐시 사용)
        self.vad_model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad", model="silero_vad", trust_repo=True
        )
        (self.get_speech_timestamps, _, self.read_audio, _, _) = utils

    def load_and_resample(self, filepath: str) -> torch.Tensor:
        """오디오를 target_sr(16kHz)로 리샘플링해 로드

        torchaudio.load()는 내부적으로 torchcodec(FFmpeg 공유 라이브러리 필요)을
        거치는데, Windows 환경에서 FFmpeg DLL을 못 찾는 문제가 흔해 soundfile로
        직접 읽는다. WAV/FLAC 등 비압축 포맷은 soundfile만으로 충분하다.
        """
        data, sr = sf.read(filepath, dtype="float32", always_2d=True)  # (samples, channels)
        waveform = torch.from_numpy(data.T)  # (channels, samples)
        if waveform.shape[0] > 1:  # 스테레오 -> 모노
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(sr, self.target_sr)
            waveform = resampler(waveform)
        return waveform.squeeze(0)

    def detect_pauses(self, waveform: torch.Tensor) -> list[PauseSegment]:
        """발화 구간(speech_timestamps) 사이의 간격을 무음 구간으로 계산"""
        speech_ts = self.get_speech_timestamps(
            waveform, self.vad_model, sampling_rate=self.target_sr
        )
        pauses = []
        total_samples = waveform.shape[0]

        # 발화 시작 전 무음
        if speech_ts and speech_ts[0]["start"] > 0:
            pauses.append(PauseSegment(0.0, speech_ts[0]["start"] / self.target_sr))

        # 발화 구간 사이 무음
        for i in range(len(speech_ts) - 1):
            gap_start = speech_ts[i]["end"] / self.target_sr
            gap_end = speech_ts[i + 1]["start"] / self.target_sr
            if gap_end > gap_start:
                pauses.append(PauseSegment(gap_start, gap_end))

        # 마지막 발화 이후 무음
        if speech_ts and speech_ts[-1]["end"] < total_samples:
            pauses.append(
                PauseSegment(speech_ts[-1]["end"] / self.target_sr, total_samples / self.target_sr)
            )

        return pauses

    def pause_metrics(self, waveform: torch.Tensor) -> dict:
        """다이어그램 [3] 음향지표: Long Pause 횟수/시간"""
        pauses = self.detect_pauses(waveform)
        long_pauses = [p for p in pauses if p.duration >= self.long_pause_threshold_sec]
        return {
            "total_pause_cnt": len(pauses),
            "total_pause_sec": sum(p.duration for p in pauses),
            "long_pause_cnt": len(long_pauses),
            "long_pause_sec": sum(p.duration for p in long_pauses),
        }