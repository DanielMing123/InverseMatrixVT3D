# Step-by-step installation instructions

**a. Create a conda virtual environment and activate it.**
```shell
conda create -n InverseMatrixVT3D python=3.8 -y
conda activate InverseMatrixVT3D
```

**b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/).**
```shell
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

**c. Install mmengine, mmcv, mmdet, mmdet3d, and mmseg.**
```shell
pip install -U openmim
mim install mmengine
mim install 'mmcv>=2.0.0rc4'
mim install 'mmdet>=3.0.0'
mim install "mmdet3d>=1.1.0"
pip install "mmsegmentation>=1.0.0"
```

**d. Install others.**
```shell
pip install focal_loss_torch
git clone https://github.com/CoinCheung/pytorch-loss.git
cd pytorch-loss
python setup.py install
```
**e. Download backbone pretrain weight.**
```shell
cd InverseMatrixVT3D
mkdir ckpt
wget https://github.com/zhiqi-li/storage/releases/download/v1.0/r101_dcn_fcos3d_pretrain.pth
```
