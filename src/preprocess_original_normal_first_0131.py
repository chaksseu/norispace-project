import os
import random
from PIL import Image, ImageEnhance
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm

def augment_image(filename, input_dir, output_dir, num_augmentations, valid_extensions):
    input_path = os.path.join(input_dir, filename)
    base_name, ext = os.path.splitext(filename)
    
    try:
        with Image.open(input_path) as img:
            img = img.convert("RGB")  # RGB로 변환

            for i in range(num_augmentations):
                augmented_img = img.copy()

                # 랜덤 색조 전처리 적용
                # Color
                if random.random() < 0.8:
                    factor = random.uniform(0.8, 1.2)
                    augmented_img = ImageEnhance.Color(augmented_img).enhance(factor)

                # Brightness
                if random.random() < 0.8:
                    factor = random.uniform(0.8, 1.2)
                    augmented_img = ImageEnhance.Brightness(augmented_img).enhance(factor)

                # Contrast
                if random.random() < 0.8:
                    factor = random.uniform(0.8, 1.2)
                    augmented_img = ImageEnhance.Contrast(augmented_img).enhance(factor)

                # Sharpness
                if random.random() < 0.8:
                    factor = random.uniform(0.8, 1.2)
                    augmented_img = ImageEnhance.Sharpness(augmented_img).enhance(factor)

                # 고유한 파일명 생성
                augmented_filename = f"{base_name}_aug_{i+1}{ext}"
                output_path = os.path.join(output_dir, augmented_filename)

                # 이미지 저장
                augmented_img.save(output_path)
        
        return f"{filename} 증강 완료."
    except Exception as e:
        return f"{filename} 처리 중 오류 발생: {e}"

def main(input_dir, output_dir, num_augmentations=1):
    # 출력 폴더가 없으면 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 이미지로 취급할 확장자 목록 (필요에 따라 추가/제거)
    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
    
    # 입력 폴더 내 유효한 이미지 파일 리스트
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_extensions)]
    
    if not image_files:
        print("입력 폴더에 유효한 이미지 파일이 없습니다.")
        return
    
    # 멀티프로세싱을 위한 Pool 설정 (CPU 코어 수에 맞춤)
    pool_size = cpu_count()
    pool = Pool(pool_size)
    
    # 부분 함수 생성
    func = partial(
        augment_image,
        input_dir=input_dir,
        output_dir=output_dir,
        num_augmentations=num_augmentations,
        valid_extensions=valid_extensions
    )
    
    # tqdm을 사용하여 진행 상황 표시
    results = []
    for result in tqdm(pool.imap_unordered(func, image_files), total=len(image_files)):
        results.append(result)
    
    pool.close()
    pool.join()
    
    # 결과 출력
    for res in results:
        print(res)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="이미지 색조 증강 스크립트")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="dataset",
        help="원본 이미지가 있는 폴더 경로"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="augmented_dataset",
        help="증강된 이미지를 저장할 폴더 경로"
    )
    parser.add_argument(
        "--num_augmentations",
        type=int,
        default=16,
        help="각 원본 이미지당 생성할 증강 이미지의 개수 (기본값: 16)"
    )
    
    args = parser.parse_args()
    
    main(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        num_augmentations=args.num_augmentations
    )
