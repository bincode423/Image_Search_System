from concurrent.futures import ThreadPoolExecutor
from icrawler.builtin import GoogleImageCrawler
import os

def download_images(keyword, max_num=50, save_root='my_dataset'):
    folder_name = keyword.replace(" ", "_")
    save_path = os.path.join(save_root, folder_name)
    os.makedirs(save_path, exist_ok=True)

    crawler = GoogleImageCrawler(storage={'root_dir': save_path})
    crawler.crawl(
        keyword=keyword,
        max_num=max_num,
        filters={'type': 'photo'}
    )
    print(f'✅ finish_{keyword}')

keywords = ['acorn', 'airplane', 'anchor', 'apple', 'ball', 'banana', 'bridge', 'candle', 'carrot',
            'chair', 'cloud', 'desk', 'door', 'drum', 'eagle', 'egg', 'elephant', 'flag', 'guitar',
            'helicopter', 'honey', 'house', 'iceberg', 'island', 'jacket', 'jet', 'jungle', 'kangaroo',
            'lamp', 'lemon', 'mountain', 'needle', 'ocean', 'pencil', 'queen', 'quilt', 'river', 'robot',
            'star', 'sunflower', 'table', 'train', 'umbrella', 'vest', 'violin', 'volcano', 'waterfall',
            'window', 'x-ray', 'xylophone', 'yacht', 'yogurt', 'yoyo', 'zebra', 'zoo']

max_workers = 5
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = [executor.submit(download_images, kw, 50) for kw in keywords]
    for future in futures:
        future.result()