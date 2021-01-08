# Example usage of Convolutional Support Estimator Network (CSEN) regressor for the distance estimation task over KITTI dataset:

Software environment, utilized libraries with versions:\
```
python == 3.7.7
tensorflow==2.1.0
pandas == 1.0.4
tqdm
OpenCV == 4.2.0
Numpy == 1.18.1
Pillow (PIL) == 1.1.7
SciPy == 1.5.2
Scikit-Image == 0.16.2
```

# KITTI dataset preprocessing.

1. First generate csv file for the KITTI annotations:
    ```
    python kitti-data/generate-csv.py --input=kitti-data/original_data/train_annots/ --output=kitti-data/annotations.csv
    ```

    You could download the kitti dataset from here: https://tuni-my.sharepoint.com/:f:/g/personal/mete_ahishali_tuni_fi/ErxxrsqyqyVKuFmxJhZfmIwBQ3tE7r_e_1aqjmD9QX2tjA?e=JT92JH

2. Rough dense depth-maps are computed using Struct2Depth pre-trained network.\
    V. Casser, S. Pirk, R. Mahjourian, A. Angelova, Depth Prediction Without the Sensors: Leveraging Structure for Unsupervised Learning from Monocular Videos, AAAI Conference on Artificial Intelligence, 2019 https://arxiv.org/pdf/1811.06152.pdf

    Code is released as a part of Tensorflow models:\
    https://github.com/tensorflow/models/tree/archive/research/struct2depth

    We are using the Tensorflow model, trained on the Cityscapes dataset, to compute depth maps of the Kitti dataset. The model can be downloaded from here:\
    https://tuni-my.sharepoint.com/:f:/g/personal/mete_ahishali_tuni_fi/EriDZHnpwb9Eo3lWmbNYVtUB7Bx9LkAJdjxgYueb7k0yjw?e=DawMzT

    You could set the input and output directories and give the model path in the script:

    ```
    input_dir="kitti-data/original_data/train_images"
    output_dir="kitti-data/extracted_depth_masks/"
    model_checkpoint="struct2depth_model_cityscapes/model-154688"
    ```

    Example code execution:

    ```
    python inference.py \
        --logtostderr \
        --file_extension png \
        --depth \
        --egomotion true \
        --input_dir $input_dir \
        --output_dir $output_dir \
        --model_ckpt $model_checkpoint
    ```

    Note: If you want to skip this step, I have included the extracted depth maps in the provided link of OneDrive folder in the 1st step.\
    (https://tuni-my.sharepoint.com/:f:/g/personal/mete_ahishali_tuni_fi/ErxxrsqyqyVKuFmxJhZfmIwBQ3tE7r_e_1aqjmD9QX2tjA?e=JT92JH)

3. Crop object features from the computed dense depth masks in the previous step.

    ```
    cd csen-regressor/

    python kitti_preprocess/cropper_mask_kitti.py
    ```

    Parameters in the script:
    ```
    Sizee = 64 # Feature size, e.g., 32 x 32, 1024-D features.\
    Occlusion:\
    Integer (0,1,2,3) indicating occlusion state:\
        0 = fully visible, 1 = partly occluded\
        2 = largely occluded, 3 = unknown.\
    Default selection: 0 and 1 cases.
    ```

    The resulted features are saved in XD_features_mask_kitti.mat file where X is the specified number of features per object sample.

# Distance Estimation
4. Run crc.m MATLAB script (written on MATLAB R2019a) for Collaborative Representation based Classification (CRC) algorithm.\
    There is Pre-processing step: Quantization and Sample Selection.\
    The outputs are recorded in crc_output folder.

    Parameters in the script:
    ```
    param.dictionary_size = 20; % Samples per class in the dictionary.
    These are the train/test proportations.
    param.train_size = 2;
    param.test_size = 2;
    MR = 0.5; % Measurement rate.
    measurement_type = 'eigenface'; % Gauss, eigenface, or None.
    projection_matrix = 'l2_norm'; % minimum_norm or l2_norm.
    ```

5. The Convolutional Support Estimator Network (CSEN) implementation for the distance estimation.
    ```
    python csen_main.py
    ```

    Parameters in the script:
    ```
    data = '4096D_crc_output/' 
    MR = '0.5' # Measurement rate.
    weights = False # If the weights are already available.
    modelType = 'CSEN' # CSEN or MLP.
    ```

    The training histories, predictions, and performance results are stored in results directory for each fold.

    modelType could be change to MLP to test MLP performance as the competing method using the calculated features.