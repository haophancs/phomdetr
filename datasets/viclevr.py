import json
import os
from pathlib import Path

import torch
import torch.utils.data
from PIL import Image

import datasets.transforms as T

ALL_ATTRIBUTES = [
    "nhỏ",
    "lớn",
    "xám",
    "đỏ",
    "xanh dương",
    "xanh lá",
    "nâu",
    "tím",
    "xanh lơ",
    "vàng",
    "hình lập phương",
    "hình cầu",
    "hình trụ",
    "cao su",
    "kim loại",
]


def encode_answer(target, answer):
    if answer in ["có", "không"]:
        target["answer_type"] = torch.as_tensor(0, dtype=torch.long)
        target["answer_binary"] = torch.as_tensor(0.0 if answer == "không" else 1.0)
        target["answer_attr"] = torch.as_tensor(-100, dtype=torch.long)
        target["answer_reg"] = torch.as_tensor(-100, dtype=torch.long)
    elif answer in ALL_ATTRIBUTES:
        target["answer_type"] = torch.as_tensor(1, dtype=torch.long)
        target["answer_binary"] = torch.as_tensor(0.0)
        target["answer_attr"] = torch.as_tensor(ALL_ATTRIBUTES.index(answer), dtype=torch.long)
        target["answer_reg"] = torch.as_tensor(-100, dtype=torch.long)
    else:
        target["answer_type"] = torch.as_tensor(2, dtype=torch.long)
        target["answer_binary"] = torch.as_tensor(0.0)
        target["answer_attr"] = torch.as_tensor(-100, dtype=torch.long)
        target["answer_reg"] = torch.as_tensor(int(answer), dtype=torch.long)
    return target


def decode_answer(encoded_answer: int, answer_type: int):
    assert answer_type in [0, 1, 2]
    if answer_type == 0:
        assert encoded_answer in [0, 1]
        return ["không", "có"][encoded_answer]
    if answer_type == 1:
        assert encoded_answer in range(len(ALL_ATTRIBUTES))
        return ALL_ATTRIBUTES[encoded_answer]
    return int(encoded_answer)


class ViClevrQuestion(torch.utils.data.Dataset):
    def __init__(self, img_folder, ann_file, transforms):
        super(ViClevrQuestion, self).__init__()
        self.transforms = transforms
        self.root = img_folder
        with open(ann_file, "r") as f:
            self.questions = json.load(f)["questions"]

    def decode_answer(self, encoded_answer: int, answer_type: int):
        return decode_answer(encoded_answer, answer_type)

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]
        img = Image.open(os.path.join(self.root, question["image_filename"])).convert("RGB")
        target = {
            "questionId": question["question_index"] if "question_index" in question else idx,
            "caption": question["question"],
        }
        if "answer" in question:
            target = encode_answer(target, question["answer"])

        if self.transforms is not None:
            img, _ = self.transforms(
                img,
                {
                    "boxes": torch.zeros(0, 4),
                    "labels": torch.zeros(0),
                    "iscrowd": torch.zeros(0),
                    "positive_map": torch.zeros(0),
                },
            )
        return img, target


def make_clevr_transforms(image_set, cautious=False):
    normalize = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    scales = [256, 288, 320, 352, 384]

    if image_set == "train":
        horizontal = [] if cautious else [T.RandomHorizontalFlip()]
        return T.Compose(
            horizontal
            + [
                T.RandomSelect(
                    T.RandomResize(scales, max_size=512),
                    T.Compose(
                        [
                            T.RandomResize([320, 352, 384]),
                            T.RandomSizeCrop(256, 512, respect_boxes=cautious),
                            T.RandomResize(scales, max_size=512),
                        ]
                    ),
                ),
                normalize,
            ]
        )

    if image_set == "val" or image_set == 'test':
        return T.Compose(
            [
                # T.RandomResize([480], max_size=1333),
                normalize,
            ]
        )

    raise ValueError(f"unknown {image_set}")


def build(image_set, args):
    assert args.no_detection, "viclevr doesn't have boxes, please disable detection"
    im_set = image_set
    if args.test:
        im_set = "test"
    ann_file = Path(args.viclevr_ann_path) / f"VICLEVR_{im_set}_questions.json"
    img_dir = Path(args.viclevr_img_path) / f"{im_set}"

    print("loading ", img_dir, ann_file)
    return ViClevrQuestion(
        img_dir,
        ann_file,
        transforms=make_clevr_transforms(image_set, cautious=True),
    )
