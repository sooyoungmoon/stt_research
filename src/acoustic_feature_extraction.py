
import opensmile
import os
import sys

audio_path = "./data/A00_S01_F_C_01_030_02_WA_MO.wav"

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)

y = smile.process_file(audio_path)

# 2. 콘솔에 각 지표 이름과 값 모두 출력
print(f"=== Audio: {audio_path} (Total Features: {len(smile.feature_names)}) ===")
# DataFrame의 첫 번째 행 데이터를 딕셔너리 형태로 변환
feature_dict = y.iloc[0].to_dict()

for idx, (feat_name, feat_val) in enumerate(feature_dict.items(), start=1):
    print(f"[{idx:02d}] {feat_name}: {feat_val:.6f}")

# 3. 파일로 저장
output_dir = "./output"
os.makedirs(output_dir, exist_ok=True)

# 3-1. CSV 형식 저장 (열: 각 지표, 행: 발화 파일)
csv_output_path = os.path.join(output_dir, "acoustic_features_opensmile.csv")
y.to_csv(csv_output_path, index=True)
print(f"\n[Saved CSV] -> {csv_output_path}")

# 3-2. (선택) JSON 형식 저장 (키-값 구조로 파이프라인 연동에 용이)
json_output_path = os.path.join(output_dir, "acoustic_features_opensmile.json")
y.iloc[0].to_json(json_output_path, indent=4)
print(f"[Saved JSON] -> {json_output_path}")

