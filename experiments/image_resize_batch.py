from PIL import Image
from pathlib import Path

# 원본 이미지 폴더 경로
input_dir = Path("data/bassimages")
# 변환된 이미지 저장 폴더
output_dir = input_dir / "resized_images"
output_dir.mkdir(exist_ok=True)

# 목표 해상도
target_size = (1920, 1080)

for img_path in input_dir.rglob("*"):
    if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
        try:
            with Image.open(img_path) as img:
                # 이미지 비율 유지하면서 리사이즈 (thumbnail)
                img.thumbnail(target_size, Image.LANCZOS)

                # 새 배경 만들기 (검정색)
                new_img = Image.new("RGB", target_size, (0, 0, 0))
                paste_pos = (
                    (target_size[0] - img.size[0]) // 2,
                    (target_size[1] - img.size[1]) // 2
                )
                new_img.paste(img, paste_pos)

                # 저장 파일명 그대로 저장
                new_img.save(output_dir / img_path.name)
        except Exception as e:
            print(f"이미지 변환 실패: {img_path.name} - {e}")
