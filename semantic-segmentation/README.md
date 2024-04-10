## EHVT on Domain Generalized Semantic Segmentation

This is the implementation of EHVT on domain generalized semantic segmentation. 

### Dependencies:
We use python 3.8.18, and pytorch 1.12.0 with cuda 11.3. 
```shell
conda create -n ehvt-seg python=3.8
conda activate ehvt-seg
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
conda install scipy
conda install tqdm
conda install scikit-image
pip install tensorboardX
pip install thop
pip install kmeans1d
imageio_download_bin freeimage
```

### Datasets:
We trained our model with the source domain [GTAV](https://download.visinf.tu-darmstadt.de/data/from_games/). Then we evaluated the model on [Cityscapes](https://www.cityscapes-dataset.com/), [BDD-100K](https://bdd-data.berkeley.edu/), and [Mapillary Vistas](https://www.mapillary.com/dataset/vistas?pKey=2ix3yvnjy9fwqdzwum3t9g&lat=20&lng=0&z=1.5).

Following [RobustNet](https://github.com/shachoi/RobustNet), we adopt Class uniform sampling proposed in [this paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhu_Improving_Semantic_Segmentation_via_Video_Propagation_and_Label_Relaxation_CVPR_2019_paper.pdf) to handle class imbalance problems. 

1. We used [GTAV_Split](https://download.visinf.tu-darmstadt.de/data/from_games/code/read_mapping.zip) to split GTAV dataset into train/val/test set.

```
gtav
 └ images
 └ labels
```

2. For [Cityscapes](https://www.cityscapes-dataset.com/), download "leftImg8bit_trainvaltest.zip" and "gtFine_trainvaltest.zip". Unzip the files and make the directory structures as follows.
```
cityscapes
  └ leftImg8bit
    └ train
    └ val
    └ test
  └ gtFine
    └ train
    └ val
    └ test
```
3. For [BDD-100K](https://bdd-data.berkeley.edu/), download "10K Images" and "Segmentation". Unzip the files and make the directory structures as follows.
```
bdd100k
 └ images/10k
   └ train
   └ val
   └ test
 └ labels/semseg/masks
   └ train
   └ val
```
4. For [Mapillary Vistas](https://www.mapillary.com/dataset/vistas?pKey=2ix3yvnjy9fwqdzwum3t9g&lat=20&lng=0&z=1.5), download the full dataset. Unzip the files and make the directory structures as follows.
```
mapillary
 └ training
   └ images
   └ labels
 └ validation
   └ images
   └ labels
 └ test
   └ images
   └ labels
```

### Run
You should modify the path in **"<path_to_EHVT>/config.py"** according to your dataset path.
```
#Cityscapes Dir Location
__C.DATASET.CITYSCAPES_DIR = <YOUR_CITYSCAPES_PATH>
#Mapillary Dataset Dir Location
__C.DATASET.MAPILLARY_DIR = <YOUR_MAPILLARY_PATH>
#GTAV Dataset Dir Location
__C.DATASET.GTAV_DIR = <YOUR_GTAV_PATH>
#BDD-100K Dataset Dir Location
__C.DATASET.BDD_DIR = <YOUR_BDD_PATH>
```
#### Environment Inference and Environment Invariant Learning
The code of Environment Inference is mainly in file EIL.py. The environment inference in EIL is conducted every 5 epochs to acquire the environment label for each training sample.
The Environment Invariant Learning is integrated into the training process of the EHVT network.

#### Training
You can train EHVT with following commands.
```
bash scripts/res50_deepv3_gtav_ehvt.sh # Train: GTAV, Test: Cityscapes, BDD100K, Mapillary / ResNet50 + DeepV3 + EHVT
bash scripts/res50_ibnnet_gtav_ehvt.sh # Train: GTAV, Test: Cityscapes, BDD100K, Mapillary / ResNet50 + IBNNet + EHVT
bash scripts/res50_robust_gtav_ehvt.sh # Train: GTAV, Test: Cityscapes, BDD100K, Mapillary / ResNet50 + RobustNet + EHVT
```

#### Evaluation:
In the training stage, our code will automatically test the performance of four realistic datasets (Cityscapes, BDD100K, Mapillary) after each epoch.

#### Pretrained ResNet18 Model:
[Google Drive](https://drive.google.com/drive/folders/14puLJ5ccffh8Bc1P2hhPoK73gSC1TImf?usp=sharing)

### Citation

If you find our work useful in your research, please consider citing our paper:

```bibtex
@inproceedings{chang2023domain,
  title={Domain Generalized Stereo Matching via Hierarchical Visual Transformation},
  author={Chang, Tianyu and Yang, Xun and Zhang, Tianzhu and Wang, Meng},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={9559--9568},
  year={2023}
}

### Acknowledgements

This project is based on the following open-source projects. We thank their
authors for making the source code publically available.

* [RobustNet](https://github.com/shachoi/RobustNet)
* [IRMBed](https://github.com/IRMBed/IRMBed)
  

