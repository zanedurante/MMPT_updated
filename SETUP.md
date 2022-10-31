# Setup Instructions

## Create conda env
conda create -n videoclip python=3.8
conda activate videoclip

## Install PyTorch

### cpu only
`conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 cpuonly -c pytorch`
### gpu
`conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 cudatoolkit=11.3 -c pytorch -c conda-forge`

## Get MMPT_updated (if you you haven't cloned it already)
`git clone https://github.com/zanedurante/MMPT_updated`

## Get fairseq
```
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install -e .
export MKL_THREADING_LAYER=GNU  # fairseq may need this for numpy.
```

## install video clip (MMPT)
```
cd ../MMPT_updated
pip install -e .
pip install transformers
pip install pytorchvideo
```

## install pre-trained video and text tokenizers
```
mkdir pretrained_models
cd pretrained_models
wget https://www.rocq.inria.fr/cluster-willow/amiech/howto100m/s3d_howto100m.pth
wget https://www.rocq.inria.fr/cluster-willow/amiech/howto100m/s3d_dict.npy

# Dry run to get config files
python locallaunch.py projects/retri/videoclip.yaml --dryrun

# install videoclip
mkdir -p runs/retri/videoclip
cd runs/retri/videoclip
wget https://dl.fbaipublicfiles.com/MMPT/retri/videoclip/checkpoint_best.pt
```
