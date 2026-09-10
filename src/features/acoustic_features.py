"""
[3] Feature Extraction Engine - 음향 지표 (Acoustic Metrics)
Parselmouth(Praat Python 바인딩)로 다음을 추출:
  - 피치(F0): Mean, Std, Dynamic Range
  - 음성 떨림: Jitter(%), Shimmer(%)
  - 음질 안정성: HNR (Harmonics-to-Noise Ratio)
"""

import parselmouth
from parselmouth.praat import call


class AcousticFeatureExtractor:
    def __init__(self, filepath: str, f0_min: float = 75.0, f0_max: float = 500.0):
        self.filepath = filepath
        # 일반 성인 화자 기준 F0 탐색 범위(Hz). 필요 시 화자 성별/연령대에 맞춰 조정.
        self.f0_min = f0_min
        self.f0_max = f0_max

    def extract(self) -> dict:
        sound = parselmouth.Sound(self.filepath)

        # --- 피치(F0) ---
        pitch = sound.to_pitch(pitch_floor=self.f0_min, pitch_ceiling=self.f0_max)
        f0_values = pitch.selected_array["frequency"]
        f0_values = f0_values[f0_values != 0]  # 무성음(0Hz) 제거

        if len(f0_values) == 0:
            f0_mean = f0_std = f0_range = 0.0
        else:
            f0_mean = float(f0_values.mean())
            f0_std = float(f0_values.std())
            f0_range = float(f0_values.max() - f0_values.min())

        # --- Jitter / Shimmer (PointProcess 필요) ---
        point_process = call(sound, "To PointProcess (periodic, cc)", self.f0_min, self.f0_max)
        jitter_local = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        shimmer_local = call(
            [sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6
        )

        # --- HNR ---
        harmonicity = sound.to_harmonicity_cc(minimum_pitch=self.f0_min)
        hnr = call(harmonicity, "Get mean", 0, 0)

        return {
            "f0_mean_hz": f0_mean,
            "f0_std_hz": f0_std,
            "f0_range_hz": f0_range,
            "jitter_local_pct": float(jitter_local) * 100 if jitter_local == jitter_local else None,
            "shimmer_local_pct": float(shimmer_local) * 100 if shimmer_local == shimmer_local else None,
            "hnr_db": float(hnr) if hnr == hnr else None,  # NaN 체크 (Praat이 무음 구간에서 NaN 반환 가능)
        }
