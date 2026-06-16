# MambaMPD

## Installation

```bash
pip install -r requirements.txt
```

The VSS encoder requires the [`mamba-ssm`](https://github.com/state-spaces/mamba)
CUDA kernels (NVIDIA GPU + CUDA toolchain). The ImageNet-pretrained VMamba-Tiny
encoder checkpoint is at `pretrained/vssmtiny_dp01_ckpt_epoch_292.pth` and is
loaded into the encoder at the start of training.

## Datasets

### M4D (SAR, 5 classes)

Expected layout:

```
<M4D root>/
├── train/
│   ├── images/        # *.jpg
│   └── labels_1D/     # *.png  (single-channel class ids)
└── test/
    ├── images/
    └── labels_1D/
```

### MADOS (multispectral, 11 bands, 15 classes)

Download from https://doi.org/10.5281/zenodo.10664073 and place under
`./data/MADOS` together with the official `splits/{train,val,test}_X.txt`.

## Usage

### Train on M4D

```bash
python train.py \
    --dir_dataset "/path/to/M4D Oil Spill Detection Dataset" \
    --pretrained_ckpt pretrained/vssmtiny_dp01_ckpt_epoch_292.pth \
    --batch_size 8 --num_epochs 100 --learning_rate 0.001 --which_optimizer sgd
```

Configuration: `config/mambampd_m4d.yaml`.

### Evaluate on M4D

```bash
python eval.py \
    --dir_dataset "/path/to/M4D Oil Spill Detection Dataset" \
    --file_model_weights mambampd/mambampd_best.pt \
    --dir_save_preds preds/
```

### Ablations (M4D)

```bash
python train.py ... --use_faa 0            # without FAA
python train.py ... --use_ega 0            # without EGA
python train.py ... --deep_supervision 0   # without deep supervision
```

### Train on MADOS

```bash
python train_mados.py --path ./data/MADOS \
    --pretrained_ckpt pretrained/vssmtiny_dp01_ckpt_epoch_292.pth \
    --batch 4 --epochs 100
```

Configuration: `config/mambampd_mados.yaml`.

### Evaluate on MADOS

```bash
python eval_mados.py --path ./data/MADOS \
    --model_path trained_models_mados/model_best.pth --split test
```

### Demo app

```bash
streamlit run app.py
```
