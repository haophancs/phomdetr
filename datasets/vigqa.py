import json
import os
from pathlib import Path

import torch
import torch.utils.data
from PIL import Image

import datasets.transforms as T


class ViGQAQuestion(torch.utils.data.Dataset):
    def __init__(self, img_folder, ques_file, ann_file, answer2label_file, subset, transforms):
        super(ViGQAQuestion, self).__init__()
        self.transforms = transforms
        self.root = img_folder
        self.subset = subset
        with open(ques_file, "r") as f:
            self.questions = json.load(f)["questions"]
        with open(ann_file, "r") as f:
            self.answer_map = {
                it["question_id"]: it["multiple_choice_answer"]
                for it in json.load(f)["annotations"]
            }
        with open(answer2label_file, "r") as f:
            self.answer2label = {
                it["answer"]: it["label"]
                for it in [json.loads(line) for line in f.readlines()]
            }
            self.label2answer = {v: k for k, v in self.answer2label.items()}

    def decode_answer(self, encoded_answer: int, answer_type: int):
        assert answer_type == 1
        return self.label2answer[encoded_answer]

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]
        answer = self.answer_map[question["question_id"]]
        image_filename = "vigqa_{0}_{1}.png".format(
            self.subset.lower(),
            str.zfill(str(question["image_id"]), 5),
        )
        img = Image.open(os.path.join(self.root, image_filename)).convert("RGB")
        target = {
            "questionId": question["question_id"], "caption": question["question"],
            "answer_type": torch.as_tensor(1, dtype=torch.long),
            "answer_binary": torch.as_tensor(0.0),
            "answer_attr": torch.as_tensor(self.answer2label[answer], dtype=torch.long),
            "answer_reg": torch.as_tensor(-100, dtype=torch.long),
        }

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


def make_gqa_transforms(image_set, cautious=False):
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
    assert args.no_detection, "vigqa doesn't have boxes, please disable detection"
    im_set = image_set
    if args.test:
        im_set = "test"
    ann_file = Path(args.vqa_ann_path) / f"vigqa_{im_set}_annotations.json"
    ques_file = Path(args.vqa_ann_path) / f"vigqa_{im_set}_questions.json"
    img_dir = Path(args.vqa_img_path) / f"{im_set}"

    print("loading ", img_dir, ann_file)
    return ViGQAQuestion(
        img_folder=img_dir,
        ques_file=ques_file,
        ann_file=ann_file,
        answer2label_file=args.vqa_answer2label_path,
        subset=im_set,
        transforms=make_gqa_transforms(image_set, cautious=True),
    )
