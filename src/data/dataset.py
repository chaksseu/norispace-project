import os
import random
from PIL import Image
from torch.utils.data import Dataset

class ANP_Dataset(Dataset):
    """
    Anchor-Positive-Negative Dataset
    (anchor, pos, neg) 구조를 사용해 Triplet 형태로 반환.

    폴더 구조 예:
    processed_dataset_test/
    ├── train/
    │   ├── anchor/
    │   │   ├── normal_fax_1_anchor.png
    │   │   ├── normal_fax_2_anchor.png
    │   │   ...
    │   ├── pos/
    │   │   ├── normal_fax_1_pos.png
    │   │   ├── normal_fax_2_pos.png
    │   │   ...
    │   └── neg/
    │       ├── normal_fax_1_neg_0.png
    │       ├── normal_fax_1_neg_1.png
    │       └── ...
    ├── val/
    │   ├── anchor/
    │   ├── pos/
    │   └── neg/
    └── test/
        ├── anchor/
        ├── pos/
        └── neg/
    """

    def __init__(self, root_dir, mode="train", transform=None):
        """
        Args:
            root_dir  (str): 전처리된 데이터 최상위 폴더, ex) "processed_dataset_test"
            mode      (str): "train", "val", or "test"
            transform (callable): 이미지를 불러온 뒤 적용할 transform
        """
        super().__init__()
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform

        self.anchor_dir = os.path.join(root_dir, mode, "anchor")
        self.pos_dir    = os.path.join(root_dir, mode, "pos")
        self.neg_dir    = os.path.join(root_dir, mode, "neg")

        if not all([os.path.isdir(self.anchor_dir),
                    os.path.isdir(self.pos_dir),
                    os.path.isdir(self.neg_dir)]):
            raise ValueError(f"Mode='{mode}' 폴더 구조가 올바르지 않습니다. "
                             f"anchor/pos/neg 폴더가 존재해야 합니다.")

        # (1) anchor 폴더 내 파일 목록 수집
        # 예: "normal_fax_1_anchor.png"
        anchor_files = [f for f in os.listdir(self.anchor_dir)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        # (2) prefix를 뽑아서, pos/neg에서 대응되는 파일을 찾기 위해 사용
        # "normal_fax_1_anchor" -> prefix: "normal_fax_1"
        self.samples = []  # [(prefix, anchor_file), ...]

        for af in anchor_files:
            if "_anchor" not in af:
                continue
            prefix = af.rsplit("_anchor", 1)[0]  # 맨 뒤에서 "_anchor" 제거
            # 예: "normal_fax_1_anchor.png" -> prefix="normal_fax_1"
            self.samples.append((prefix, af))

        # 결과적으로 self.samples:
        # [("normal_fax_1", "normal_fax_1_anchor.png"), ("normal_fax_2", "normal_fax_2_anchor.png"), ...]
        # neg폴더 안에는 동일 prefix + "_neg_xxx.png" 가 있을 것이고,
        # pos폴더 안에는 동일 prefix + "_pos.png" 가 있을 것.

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        prefix, anchor_filename = self.samples[idx]

        anchor_path = os.path.join(self.anchor_dir, anchor_filename)
        # pos 파일
        pos_filename = f"{prefix}_pos.png"  # ex) "normal_fax_1_pos.png"
        pos_path = os.path.join(self.pos_dir, pos_filename)

        # neg 파일들 (ex: "normal_fax_1_neg_0.png", "normal_fax_1_neg_1.png", ...)
        all_neg_files = []
        for fname in os.listdir(self.neg_dir):
            if fname.startswith(prefix + "_neg_") and fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                all_neg_files.append(fname)
        if not all_neg_files:
            raise ValueError(f"Neg 파일을 찾을 수 없습니다: prefix={prefix}")

        # 하나를 랜덤 선택
        neg_filename = random.choice(all_neg_files)
        neg_path = os.path.join(self.neg_dir, neg_filename)

        # 이미지 로드
        anchor_img = Image.open(anchor_path).convert("RGB")
        pos_img    = Image.open(pos_path).convert("RGB")
        neg_img    = Image.open(neg_path).convert("RGB")

        # transform 적용
        if self.transform:
            anchor_img = self.transform(anchor_img)
            pos_img    = self.transform(pos_img)
            neg_img    = self.transform(neg_img)

        return {
            "anchor": anchor_img,
            "positive": pos_img,
            "negative": neg_img
        }


class AP_Eval_Dataset(Dataset):
    """
    Anchor-Positive 평가용 데이터셋
    (anchor, pos) 이미지 로드 후, 특정 기준(거리 threshold 등)으로 분류 여부 평가.

    ex) processed_dataset_test/
        ├── train/
        ├── val/
        └── test/
           ├─ anchor/
           ├─ pos/
           └─ neg/
    여기서는 'neg'를 사용하지 않고, (anchor, pos)만 매칭하여 반환.
    """

    def __init__(self, root_dir, mode="test", transform=None):
        """
        Args:
            root_dir (str): e.g. "processed_dataset_test"
            mode     (str): "train"/"val"/"test" 중 하나. (주로 'test' or 'val'에 사용)
        """
        super().__init__()
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform

        self.anchor_dir = os.path.join(root_dir, mode, "anchor")
        self.pos_dir    = os.path.join(root_dir, mode, "pos")

        if not (os.path.isdir(self.anchor_dir) and os.path.isdir(self.pos_dir)):
            raise ValueError(f"Mode='{mode}' 폴더 구조가 올바르지 않습니다. anchor/pos 폴더 확인 필요.")

        anchor_files = [f for f in os.listdir(self.anchor_dir)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg')) and "_anchor" in f]

        self.samples = []
        for af in anchor_files:
            prefix = af.rsplit("_anchor", 1)[0]
            # pos 파일
            pos_filename = f"{prefix}_pos.png"
            pos_path = os.path.join(self.pos_dir, pos_filename)
            if os.path.exists(pos_path):
                self.samples.append((prefix, af, pos_filename))
            else:
                # pos가 없으면 스킵
                pass

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        prefix, anchor_file, pos_file = self.samples[idx]

        anchor_path = os.path.join(self.anchor_dir, anchor_file)
        pos_path    = os.path.join(self.pos_dir,   pos_file)

        anchor_img = Image.open(anchor_path).convert("RGB")
        pos_img    = Image.open(pos_path).convert("RGB")

        if self.transform:
            anchor_img = self.transform(anchor_img)
            pos_img    = self.transform(pos_img)

        # 라벨이 따로 없는 구조이므로, 필요하다면 prefix 등에서 label 추론 가능(전처리 시 normal/fraud?)
        # 여기서는 그냥 prefix만 반환
        return {
            "anchor": anchor_img,
            "pos": pos_img,
            "prefix": prefix  # 예: "normal_fax_6"
        }
