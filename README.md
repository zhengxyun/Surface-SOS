# Surface-SOS

This is repo implementation of Surface-SOS, a Self-Supervised Object Segmentation via Neural Surface Representation.

<div>
<img src="./static/videos/daily_dayin_res.gif" height="280"/>
</div>

### [Project page](https://zhengxyun.github.io/Surface-SOS/) · [Paper](https://ieeexplore.ieee.org/abstract/document/10471326) 

## Environments

To utilize multiresolution hash encoding or fully fused networks provided by tiny-cuda-nn, you should have least an RTX 2080Ti, see https://github.com/NVlabs/tiny-cuda-nn#requirements for more details.

```
conda create -n sos python==3.8
```

Install PyTorch>=1.10 here based the package management tool you used and your cuda version. For example：
```
pip install torch==1.12.0 torchvision==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116
```

Install tiny-cuda-nn PyTorch extension: 
```
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

The following dependencies are required:
```
pip install -r requirements.txt
```

## Data Preparation
### Download Prepared Data
We provide a data sample for scene "Dayin" and "teddy bear" (with multi-view images, camera poses, and masks).

Then files according to the following directory structure:

```
├── data
│   ├── COLMAP-neus
│   │   └── daily_dayin 
│   │       └── images 
│   │       └── sparse
│   │       └── masks
│   │   └── tum_teddy 
|   |   └── ...

```


## Training

After preparing datasets, users can train a Surface-SOS by the following command:

```
## train on COLMAP data without mask
python launch.py --config ./configs/neus-colmap.yaml --gpu 0 --train dataset.scene=daily_dayin tag=colmap_womsk

## train on COLMAP data with mask
python launch.py --config ./configs/neus-colmap.yaml --gpu 1 --train dataset.scene=daily_dayin dataset.apply_mask=true tag=colmap_wmsk

```

## Citation

If you find this repo is helpful, please cite:

```

@article{zheng2024surface_sos,
  title={Surface-SOS: Self-Supervised Object Segmentation via Neural Surface Representation},
  author={Zheng, Xiaoyun and Liao, Liwei and Jiao, Jianbo and Gao, Feng and Wang, Ronggang},
  journal={IEEE Transaction on Image Processing},
  year={2024}
}

```


The website template is based on [nerfies](https://github.com/nerfies/nerfies.github.io).
