import sys
import io
from PIL import Image
from pathlib import Path
from collections import Counter

# 콘솔 출력 인코딩 강제 변경
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')

# 이미지 폴더 경로
image_dir = Path("data/bassimages")
resolutions = []

for img_path in image_dir.rglob("*"):
    if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
        try:
            with Image.open(img_path) as img:
                resolutions.append(img.size)
        except Exception as e:
            print(f"이미지를 열 수 없습니다: {img_path.name} - {e}")

if resolutions:
    width_dist = Counter(w for w, h in resolutions)
    height_dist = Counter(h for w, h in resolutions)

    print(f"\n총 이미지 수: {len(resolutions)}장")
    print(f"가장 작은 해상도: {min(resolutions, key=lambda x: x[0]*x[1])} (가로 x 세로)")
    print(f"가장 큰 해상도: {max(resolutions, key=lambda x: x[0]*x[1])} (가로 x 세로)")
    print(f"이미지 평균 크기 (픽셀 수): {sum(w*h for w, h in resolutions) // len(resolutions)}")

    print(f"\n[가로 크기 분포] → {dict(width_dist)}")
    print(f"[세로 크기 분포] → {dict(height_dist)}")
else:
    print("처리할 이미지가 없습니다.")
