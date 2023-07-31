# Vi-MDETR evaluated in viclevr dataset
## Download dataset
```
gdown 1nn7ZehICLvwaNEuu6tll3h73523ke77O
unzip VICLEVR.zip
```
## Installation
```
git clone https://github.com/haophancs/mdetr
cd /path/to/mdetr
pip install -r requirements.txt
mkdir VICLEVR
```

## Training
```
python main.py \
  --batch_size 32 \
  --dataset_config configs/viclevr.json \
  --backbone "resnet18" \
  --num_queries 25 \
  --schedule linear_with_warmup \
  --text_encoder_type vinai/phobert-base \
  --output_dir viclevr \
  --epochs 30 \
  --lr_drop 40
```

## Evaluation
```
python main.py \
  --batch_size 32 \
  --dataset_config configs/viclevr.json \
  --num_queries 25 \
  --text_encoder_type vinai/phobert-base \
  --backbone resnet18 \
  --resume viclevr/BEST_checkpoint.pth \
  --eval \
  --test
```