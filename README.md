Representation Based Regression for Object Distance Estimation
=============================

This repository includes the implentation of the methods in [Representation Based Regression for Object Distance Estimation](https://arxiv.org/abs/2106.14208).

Software environment:
```
1. Python with the following version and libraries.
python == 3.7.7
tensorflow == 2.1.0
Numpy == 1.18.5
Matplotlib == 3.2.2
SciPy == 1.5.0
scikit-learn == 0.24.1
argparse == 1.1
OpenCV == 4.2.0
pandas == 1.0.4
tqdm == 4.46.1
```
```
2. MATLAB -> MATLAB R2019a.
```

Content:
- [Citation](#Citation)
- [Getting started with the KITTI Dataset](#Getting-started-with-the-KITTI-Dataset)
- [Feature Extraction](#Feature-Extraction)
- [Distance Estimation via Representation based Classification](#Distance-Estimation-via-Representation-based-Classification)
    - [Sparse Representation based Classification (SRC)](#Sparse-Representation-based-Classification-SRC)
    - [Collaborative Representation based Classification (CRC)](#Collaborative-Representation-based-Classification-CRC)
- [Distance Estimation using Representation based Regression (RbR)](#Distance-Estimation-using-Representation-based-Regression-RbR)
    - [Convolutional Support Estimator Network (CSEN)](#Convolutional-Support-Estimator-Network-CSEN)
    - [Compressive Learning CSEN (CL-CSEN)](#Compressive-Learning-CSEN-CL-CSEN)
- [Distance Estimation using Support Vector Regressor (SVR)](#Distance-Estimation-using-Support-Vector-Regressor-SVR)
- [References](#References)

## Citation

If you use method(s) provided in this repository, please cite the following paper:

```
@misc{ahishali2021representation,
      title={Representation Based Regression for Object Distance Estimation}, 
      author={Mete Ahishali and Mehmet Yamac and Serkan Kiranyaz and Moncef Gabbouj},
      year={2021},
      eprint={2106.14208},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Getting started with the KITTI Dataset

The left color images of object dataset and the corresponding training labels can be obtained from [3D Object Detection Evaluation 2017](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d).

```
unzip data_object_image_2.zip
unzip data_object_label_2.zip
```

After unzipping, move them under ```kitti-data/``` folder and using ```generate-csv.py``` script generate a csv file for the KITTI annotations:

```
python kitti-data/generate-csv.py \
    --input=kitti-data/training/label_2/ \
    --output=kitti-data/annotations.csv
```

## Feature Extraction

Features are extracted using ```feature_extraction.py``` script. The supported models are ```DenseNet121```, ```VGG19```, and ```ResNet50``` (DenseNet-121 [1], VGG19 [2], ResNet-50 [3]):

```
python feature_extraction.py --model VGG19 
```

Next, the features are further processed and ordered using ```processFeatures.m```. In the script, please also set the proper model name ```param.modelName``` to either ```DenseNet121```, ```VGG19```, or ```ResNet50``` and ```param.DicDesign``` to ```2D``` or ```1D``` corresponding to the dictionary designs used in CSEN and CL-CSEN approaches. This procedure is only needed for the CSEN, CL-CSEN, and SVR approaches. If you are only interested in running SRC and CRC methods, you may proceed to the related sections: [SRC](#Sparse-Representation-based-Classification-SRC) and [CRC](#Collaborative-Representation-based-Classification-CRC). Note that the script of ```processFeatures.m``` produces the predicted distances using the CRC-light model that is discussed in the paper.

## Distance Estimation via Representation based Classification

We formulate the distance estimation task as a representation based classification problem by estimating the quantized distance values. For example, for the objects between [0.5, 60.5] meters away from the camera, we estimate a quantized distance level from 60 different distance levels ranging between [1, 60] with 1-meter senstivity.

### Sparse Representation based Classification (SRC)

There are implemented 8 different SRC algorithms for the distance estimation task including ADMM [4], Dalm [5], OMP [5], Homotopy [6], GPSR [7], L1LS [8], ℓ1-magic [9], and Palm [5]. You may run all of them in once as follows:
```
cd src/
run main.m
```
Alternatively or preferably (e.g., you may choose a specific SRC method since the required run-time is huge for running all SRC methods in once), the selected ones can be defined in the script:
```
l1method={'solve_ADMM','solve_dalm','solve_OMP','solve_homotopy','solveGPSR_BCm', 'solve_L1LS','solve_l1magic','solve_PALM'}; %'solve_PALM' is very slow
```
Similarly, please also set the proper model name ```param.modelName``` to either ```DenseNet121```, ```VGG19```, or ```ResNet50```.

### Collaborative Representation based Classification (CRC)
Distance estimation using the CRC method [10] can be run as follows:
```
cd crc\
run main.m
```
The CRC-light model can be run by setting ```CRC_light = 1``` in the script. Please change the model name ```param.modelName``` to ```DenseNet121```, ```VGG19```, or ```ResNet50``` to try different features.

## Distance Estimation using Representation based Regression (RbR)

Contrary to previous methos, it is possible to directly estimate the object distance information without the quantization step using CSEN and CL-CSEN approaches. As CSEN and CL-CSEN approaches still utilize the representative dictionary, we introduce the term <em>Representation based Regression (RbR)</em> for the proposed framework.

### Convolutional Support Estimator Network (CSEN)
The CSEN implementation is run as follows:
```
python regressor_main.py --method CSEN --feature_type DenseNet121
```
Note that similarly, the feature type can be set to ```DenseNet121```, ```VGG19```, or ```ResNet50```. If you like, only testing can be performed using the provided weights:
```
python regressor_main.py --method CSEN --feature_type DenseNet121 --weights True
```
### Compressive Learning CSEN (CL-CSEN)

The CL-CSEN implementation is run as follows:
```
python regressor_main.py --method CL-CSEN --feature_type DenseNet121
```
The parameter ```--feature_type``` can be set to ```DenseNet121```, ```VGG19```, or ```ResNet50```.

Testing of CL-CSEN with the provided weights:
```
python regressor_main.py --method CL-CSEN --feature_type DenseNet121 --weights True
```

## Distance Estimation using Support Vector Regressor (SVR)

The SVR method is implemented as a competing regressor. We use the Nystroem method for the kernel approximation since it is unfeasible to compute exact kernel mapping with the given high-dimensional dataset. Hyperparameter search is applied with grid search and then the performance is computed with the found optimal SVR parameters:
```
python regressor_main.py --method SVR --feature_type DenseNet121
```
The parameter ```--feature_type``` can be set to ```DenseNet121```, ```VGG19```, or ```ResNet50```.

## References
[1] G. Huang, Z. Liu, L. Van Der Maaten, and K. Q. Weinberger, "Densely connected convolutional networks," *in Proc. IEEE Conf. Comput. Vision and Pattern Recognit. (CVPR)*, 2017, pp. 4700–4708. \
[2] K. Simonyan and A. Zisserman, "Very deep convolutional networks for large-scale image recognition," *arXiv preprint arXiv:1409.1556*, 2014. \
[3] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," *in Proc. IEEE Conf. Comput. Vision and Pattern Recognit. (CVPR)*, 2016, pp. 770–778. \
[4] S. Boyd, N. Parikh, E. Chu, B. Peleato, J. Eckstein et al., "Distributed optimization and statistical learning via the alternating direction method of multipliers," *Found. Trends Mach. Learn.*, vol. 3, no. 1, 2011. \
[5] A. Y. Yang, Z. Zhou, A. G. Balasubramanian, S. S. Sastry, and Y. Ma, "Fast l1-minimization algorithms for robust face recognition," *IEEE Trans. Image Process.*, vol. 22, no. 8, pp. 3234–3246, 2013. \
[6] D. M. Malioutov, M. Cetin, and A. S. Willsky, "Homotopy continuation for sparse signal representation," *in Proc. IEEE Int. Conf. Acoust., Speech, and Signal Process. (ICASSP)*, vol. 5, 2005, pp. 733–736. \
[7] M. A. Figueiredo, R. D. Nowak, and S. J. Wright, "Gradient projection for sparse reconstruction: Application to compressed sensing and other inverse problems," *IEEE J. Sel. Topics Signal Process.*, vol. 1, no. 4, pp. 586–597, 2007. \
[8] K. Koh, S.-J. Kim, and S. Boyd, "An interior-point method for large-scale l1-regularized logistic regression," *J. Mach. Learn. Res.*, vol. 8, pp. 1519–1555, 2007. \
[9] E. Candes and J. Romberg, "l1-magic: Recovery of sparse signals via convex programming," *Caltech, Tech. Rep.*, 2005. [Online]. Available: https://statweb.stanford.edu/∼candes/software/l1magic/downloads/l1magic.pdf \
[10] L. Zhang, M. Yang, and X. Feng, "Sparse representation or collaborative representation: Which helps face recognition?" *in Proc. IEEE Int. Conf. Comput. Vision (ICCV)*, 2011, pp. 471–478.