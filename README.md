# Traffic sign recognition
The task is to recognize and classify road signs from a general image.
Road signs used in post-Soviet countries were chosen as the target type.

More than 95% of road signs in post-Soviet countries are identical, so you can use this model for any of these countries 
(Armenia, Azerbaijan, Belarus, Estonia, Georgia, Kazakhstan, Kyrgyzstan, Latvia, Lithuania, Moldova, Russia, Tajikistan, Turkmenistan, Ukraine and Uzbekistan)

**More details in [Wikipedia](https://en.wikipedia.org/wiki/Traffic_signs_in_post-Soviet_states)**
![](readme_images/Traffic_signs_in_post_Soviet_states.png)

## Data
The prepared dataset contains 200 classes, 117.000 images and was also publicly available on Kaggle:
https://www.kaggle.com/datasets/mikhailkosov/traffic-signs-in-post-soviet-states-200-classes

The imbalance of classes is justified by the complexity of the structure and the variety of forms of road signs, on the one hand,
and vice versa - simple and uniform form on the other hand.

**More details in [EDA.ipynb](EDA.ipynb)**

![](readme_images/Class_numbers.png)

## Evaluation Metrics
For training, validation and testing, the dataset was divided in a stratified way (70/15/15): [Training_model/split_dfs]()
* pre-trained weights: [Training_model/trained_models_tf/](Training_model/trained_models_tf/best_weights.hdf5);
* chart metrics: [Training_model/res_metrics/]()

![](readme_images/metrics_36M_64x64_80ep_log.png)

## Project Structure
* описание продукта, который решает задачу;
* скрипты для получения данных и ссылка на данные с разметкой;
* пайплайны ML экспериментов с инструкциями воспроизведения (работа с данными, обучение, валидация, визуализация графиков/дашбордов);
* скрипты продукта с инструкциями полного запуска;

- **EDA.ipynb** Jupyter notebook analyzing the dataset.
- **preparation_balanced_df.py**: Python file for preparation training datasets.
- **model_training.py**: Python file for model training.
- **model_inference.py**: Python file for model inference.
- **balanced_dfs**: Folder with balanced dataframes *(after executing preparation_balanced_df.py)*.
- **pretrained_models**: Folder with pretrained models *(after executing model_training.py)*.
- **plot_training_metrics**: Folder with charts of accuracy and loss metrics *(after executing model_training.py)*.
- **test_images**: Folder with 50 satellite images *(for testing on model_inference.py)*.
- **segmented_images**: Folder with segmented images *(after executing model_inference.py)*.
- **requirements.txt**: List of required Python modules


## How to Run
* unzip balanced_dfs/train_ship_segmentations_v2.zip
1) **preparation_balanced_df.py** based on *balanced_dfs/train_ship_segmentations_v2.csv*
prepares two additional dataframes balanced by the number of masks in each image:
    - *balanced_dfs/balanced_train_df_1.csv* - 100% images with masks;
    - *balanced_dfs/balanced_train_df_2.csv* - 50% of images with masks and 50% without masks.
2) **model_training.py** trains the model in two iterations (on two balanced data sets)
and save (depending on the size of the images supplied to the model input: 768 or 384) the trained models (at each training iteration):
   - *pretrained_models/trained_local/384_best_weight_model_1.h5* - after the first iteration of training;
   - *pretrained_models/trained_local/384_best_weight_model_2.h5* - after the second iteration of training.
   It also saves graphs of the evolution of the Dice and Loss metrics at the training epochs:
   - *plot_training_metrics/plot-1.png* - after the first iteration of training;
   - *plot_training_metrics/plot-2.png* - after the second iteration of training.
3) **model_inference.py** uses two functions to test the model on new images: 
   - *def segmentation_random_images* displays and saves 5 random images to a folder: *segmented_images/segmented_5_images.png*.
   - *def segmentation_load_image* displays and saves 1 selected image to a folder: *segmented_images/segmented_loaded_image.png*, - but for it to work, you need to set the path (**image_path**) to this image in the file.


## Architecture
1) U-net model with additional BatchNormalization in each layer.
2) Optimizer - RMSprop(start_lr=1e-3) with reduceLROnPlat(factor=0.5, patience=3).
3) Loss function based on IoU metric.


### Segmentation - Detectron2
Для сегментации торожных знаков с изображений общего плана использовалась модель [detectron2 от facebookresearch](https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md)
[detectron2/LVISv0.5-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x/](https://dl.fbaipublicfiles.com/detectron2/LVISv0.5-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x/144219108/model_final_5e3439.pkl)
![](readme_images/Segmented_Image.png)


## Environment
* описание окружения (requirements/Docker/etc) с инструкциями установки ;
For training on a video card, I used [tensorflow-gpu==2.10.0](https://www.tensorflow.org/install/source_windows). To install this library, you must use the following auxiliary tools:
   * [Bazel 5.1.1](https://github.com/bazelbuild/bazel/releases?q=5.1.1&expanded=true)
   * [cuDNN](https://developer.nvidia.com/rdp/cudnn-archive)
   * [CUDA](https://developer.nvidia.com/cuda-toolkit-archive)
   
![](readme_images/enviroment.png)

pip install "git+https://github.com/facebookresearch/detectron2.git"
("LVISv0.5-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x.yaml")

