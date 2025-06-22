import os

print(len(os.listdir('my_dataset')))
def print_keywords_with_few_images(dataset_path='my_dataset', threshold=40):
    for keyword in os.listdir(dataset_path):
        keyword_path = os.path.join(dataset_path, keyword)
        
        if os.path.isdir(keyword_path):
            image_files = [
                f for f in os.listdir(keyword_path)
                if os.path.isfile(os.path.join(keyword_path, f))
            ]
            
            if len(image_files) < threshold:
                print(f"{keyword} ({len(image_files)} images)")

# 실행
print_keywords_with_few_images()