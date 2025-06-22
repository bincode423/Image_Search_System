import os

def trim_images_to_limit(dataset_path='my_dataset', threshold=40):
    for keyword in os.listdir(dataset_path):
        keyword_path = os.path.join(dataset_path, keyword)
        
        if os.path.isdir(keyword_path):
            # 모든 파일 목록 (정렬 포함)
            image_files = sorted([
                f for f in os.listdir(keyword_path)
                if os.path.isfile(os.path.join(keyword_path, f))
            ])

            image_count = len(image_files)
            
            if image_count <= threshold:
                print(f"✅ {keyword} ({image_count} images) - 유지됨")
            else:
                # 초과 이미지 삭제
                extra_images = image_files[threshold:]
                for img in extra_images:
                    os.remove(os.path.join(keyword_path, img))
                print(f"🗑️ {keyword} - {len(extra_images)}개 이미지 삭제됨 (총 {image_count}개 → {threshold}개)")
trim_images_to_limit()
