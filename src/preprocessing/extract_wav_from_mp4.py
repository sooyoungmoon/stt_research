"""
지정한 디렉토리 하위(하위 디렉토리 포함)의 모든 .mp4 파일에서 오디오를
추출해, 같은 디렉토리에 동일한 파일명의 .wav로 저장한다.
이미 같은 이름의 .wav가 존재하면 건너뛴다.

FFmpeg CLI를 직접 호출한다 (torchaudio/torchcodec 경유가 아님) — 지난번
겪으신 Windows torchcodec DLL 문제와 무관하게 독립적으로 동작한다.
FFmpeg가 시스템에 설치되어 PATH에 잡혀 있어야 한다 (ffmpeg -version 으로 확인 가능).

사용법:
    python extract_wav_from_mp4.py "D:/path/to/pitch_videos"
"""

import subprocess
import sys
from pathlib import Path


def check_ffmpeg_available() -> bool:
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def extract_wav(mp4_path: Path, wav_path: Path, sample_rate: int = 16000) -> bool:
    """ffmpeg -i input.mp4 -ar 16000 -ac 1 output.wav 형태로 오디오만 추출.
    성공 시 True, 실패 시 False를 반환하고 stderr를 출력한다."""
    cmd = [
        "ffmpeg",
        "-y",                       # 기존 파일 있어도 덮어쓰기(단, 우리는 이미 존재 여부를 미리 걸러냄)
        "-i", str(mp4_path),
        "-vn",                      # 비디오 스트림 제외
        "-ar", str(sample_rate),    # 샘플레이트 16kHz (파이프라인 표준과 일치)
        "-ac", "1",                 # 모노
        "-loglevel", "error",       # 불필요한 로그 억제, 에러만 출력
        str(wav_path),
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print(f"    [실패] {mp4_path.name}: {result.stderr.strip()}")
        return False
    return True


def process_directory(root_dir: str, sample_rate: int = 16000) -> None:
    root = Path(root_dir)
    if not root.is_dir():
        print(f"[오류] 디렉토리를 찾을 수 없습니다: {root_dir}")
        sys.exit(1)

    mp4_files = sorted(root.rglob("*.mp4"))
    # 대소문자 구분 없는 파일시스템(Windows)에서는 *.MP4도 rglob("*.mp4")가 잡아주지만,
    # 리눅스/맥처럼 대소문자를 구분하는 환경 대비 대문자 확장자도 별도로 수집.
    mp4_files += sorted(root.rglob("*.MP4"))
    mp4_files = sorted(set(mp4_files))

    if not mp4_files:
        print(f"[알림] {root_dir} 하위에서 .mp4 파일을 찾지 못했습니다.")
        return

    print(f"총 {len(mp4_files)}개 MP4 파일 발견.\n")

    converted, skipped, failed = 0, 0, 0
    for mp4_path in mp4_files:
        wav_path = mp4_path.with_suffix(".wav")

        if wav_path.exists():
            print(f"[건너뜀] 이미 존재: {wav_path.name}")
            skipped += 1
            continue

        print(f"[변환 중] {mp4_path.name} -> {wav_path.name}")
        if extract_wav(mp4_path, wav_path, sample_rate=sample_rate):
            converted += 1
        else:
            failed += 1

    print(f"\n완료 — 변환: {converted}건, 건너뜀: {skipped}건, 실패: {failed}건")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python extract_wav_from_mp4.py <디렉토리경로>")
        sys.exit(1)

    if not check_ffmpeg_available():
        print("[오류] ffmpeg를 찾을 수 없습니다. PATH에 등록되어 있는지 확인해 주세요.")
        print("       (PowerShell에서 'ffmpeg -version'으로 직접 확인 가능)")
        sys.exit(1)

    process_directory(sys.argv[1])