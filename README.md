# Traffic sign recognation - project in progress
https://en.wikipedia.org/wiki/Vienna_Convention_on_Road_Signs_and_Signals
![](readme_images/Vienna_Convention.png)

## Data
Датасэт для обучения был создан на основе:
https://www.kaggle.com/datasets/daniildeltsov/traffic-signs-gtsrb-plus-162-custom-classes
При этом исходный датасэт был существенным образом преобразован ~ 20-30% (некоторые классы объеденялись, другие наоборот - разъеденялись,
часть изображений удалялась, а так же данные дополнялись из других источников). 


https://en.wikipedia.org/wiki/Traffic_signs_in_post-Soviet_states
![](readme_images/Traffic_signs_in_post_Soviet_states.png)

### Segmentation - Detectron2
https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md
pip install 'git+https://github.com/facebookresearch/detectron2.git'
![](readme_images/Segmented_Image.png)

("LVISv0.5-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x.yaml")

## Training and evaluation
![](readme_images/metrics_36M_64x64_80ep_log.png)

* описание задачи;
* описание продукта, который решает задачу;
* описание окружения (requirements/Docker/etc) с инструкциями установки ;
* скрипты для получения данных и ссылка на данные с разметкой;
* пайплайны ML экспериментов с инструкциями воспроизведения (работа с данными, обучение, валидация, визуализация графиков/дашбордов);
* скрипты продукта с инструкциями полного запуска;
* ссылка на веса моделей, которые используются в проде продукта;
* демки: схемы, картинки, гифки;
* лицензия;
* всё, что считаете полезным для вас/других.