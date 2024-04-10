## EHVT on Domain Generalized Stereo Matching 

### Dependencies:
- Python 3.9
- PyTorch 1.13.0
- torchvision 0.14.0
- matplotlib 
- opencv-python 
- pillow 

### Datasets:
- [SceneFlow](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)
- [KITTI stereo 2015](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo)
- [KITTI stereo 2012](http://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=stereo)
- [Middlebury v3](https://vision.middlebury.edu/stereo/submit3/)
- [ETH3D](https://www.eth3d.net/datasets#low-res-two-view)

### Run

#### Environment Inference and Environment Invariant Learning
The code of Environment Inference is mainly in file EIL.py. The environment inference in EIL is conducted every 5 epochs to acquire the environment label for each training sample.
The Environment Invariant Learning is integrated into the training process of the EHVT network.

#### Training:
```bash
python main.py --data_path (your SceneFlow data folder) --savemodel (the path of the saved models) --logfile (the path of the log file) --res18 (the path of the Pretrained ResNet18 Model)
```

#### Evaluation:
In the training stage, our code will automatically test the performance of four realistic datasets (Middlebury, ETH3D, KITTI2015, KITTI2012) after each epoch.

You only need to change the corresponding paths of the testing datasets in main.py.

If you want to evaluate the specific model on a certain realistic dataset:
```bash
python submission.py --loadmodel (your testing model) --test_name (the name of testing dataset) --logfile (the path of the log file) 
```

#### Pretrained ResNet18 Model:
[Google Drive](https://drive.google.com/drive/folders/14puLJ5ccffh8Bc1P2hhPoK73gSC1TImf?usp=sharing)

### Acknowledgements

This project is based on the following open-source projects. We thank their
authors for making the source code publically available.

* [PSMNet](https://github.com/JiaRenChang/PSMNet)
* [IRMBed](https://github.com/IRMBed/IRMBed)


