# Vi-MDETR evaluated in viclevr dataset
## Installation
```
git clone https://github.com/haophancs/mdetr
cd /path/to/mdetr
pip install -r requirements.txt
mkdir datasets
mkdir -p outputs/viclevr
```

## Download dataset
```
cd /path/to/mdetr/datasets
# download VICLEVR.zip
unzip VICLEVR.zip
```

## Training
```
python main.py \
  --dataset_config configs/viclevr.json \
  --backbone "resnet18" \
  --num_queries 25 \
  --schedule linear_with_warmup \
  --text_encoder_type vinai/phobert-base \
  --output_dir outputs/viclevr \
  --lr 5e-5 \
  --lr_backbone 5e-5 \
  --text_encoder_lr 5e-5 \
  --batch_size 16 \
  --epochs 30 \
  --weight_decay 0.01 \
  --dropout 0.5 \
  --seed 42
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
