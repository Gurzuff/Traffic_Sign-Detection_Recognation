import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from Training_model.mymodel import MyModel_32, MyModel_64, MyModel_48

def final_image(img_segmented, name_img, y_pred, y_prob):
    # Создаем фигуру (лист) с пропорцией изображений: 4 - большое, 1 - малые
    fig = plt.figure(figsize=(18, 6))
    gs = GridSpec(1, 9, width_ratios=np.array([4] + [1] * 8))

    # Сегментирование знаков на общем изображении
    ax_large = plt.subplot(gs[0])
    ax_large.imshow(img_segmented, cmap='viridis')
    ax_large.set_title(f'Image {name_img} {img_segmented.size}')
    ax_large.axis('off')

    # Создаем вложенные GridSpec для правой части
    gs_right = GridSpecFromSubplotSpec(2, len(y_pred), gs[1:], width_ratios=[1] * len(y_pred))

    # Draw images from folder "segmented_sign" in the first row
    signs_path = os.path.join(PATH_signs, name_img)
    signs_segmented = os.listdir(signs_path)
    for road_sign, sign_prob, i in zip(signs_segmented, y_prob, range(len(y_pred))):
        sign_path = os.path.join(signs_path, road_sign)
        road_sign = Image.open(sign_path)
        ax_small = plt.subplot(gs_right[i])
        ax_small.imshow(road_sign, cmap='viridis')
        ax_small.set_title(f'Segmented sign {i+1}')
        ax_small.axis('off')

    # Draw images from folder "PATH_meta" in the second row
    low_quality_img = Image.open('Readme_images/Low_quality.png')
    for sign_class, sign_prob, i in zip(y_pred, y_prob, range(len(y_pred), 2 * len(y_pred))):
        sign_name = df_classes.sign_name[df_classes.sign_class == sign_class].values
        ax_small = plt.subplot(gs_right[i])
        if sign_prob > 0.5:
            meta_path = os.path.join(PATH_meta, f'{sign_class}.png')
            meta_img = Image.open(meta_path)
            # ax_small = plt.subplot(gs_right[i])
            ax_small.imshow(meta_img, cmap='viridis')
            # sign_prob = np.round(sign_prob, 3)
            ax_small.set_title(f'{sign_name} (prob.: {format(sign_prob*100, ".0f")}%)')   #
            ax_small.axis('off')
        else:

            # ax_small = plt.subplot(gs_right[i])
            ax_small.imshow(low_quality_img, cmap='viridis')
            ax_small.set_title(f'Low quality!')
            ax_small.axis('off')

    plt.tight_layout()
    plt.savefig(f'Inference_model/{name_weight}/{name_img}')


# Custom parameters
name_weight = 'model_43M_32x32_9966'   # 'model_43M_64x64_9984'
PATH_root = 'E:\DataSets\Traffic Sign - Detection&Recognation\Traffic_Sign - 200 classes'
PATH_meta = os.path.join(PATH_root, 'Meta')
# Paths to the folders with segmented test images and road signs
PATH_test_images = 'Segmentation_image\segmented_images'
PATH_signs = 'Segmentation_image\segmented_sign'

# Set parameters for model
KERNEL = int(name_weight[10:12])
input_shape = (None, KERNEL, KERNEL, 3)
CLASSES = len(os.listdir(PATH_meta))

# Load the model and pre-trained weights
model = MyModel_32(CLASSES, input_shape)
model.build(input_shape)
model.load_weights(f'Training_model/trained_models_tf/{name_weight}.hdf5')

# Load DF with full class names
df_classes = pd.read_csv('Class_sign_description/full_name_signs.csv')

def main():
    # Create a folder to save final classified road sign images
    if os.path.exists(f'Inference_model/{name_weight}'):
        pass
    else:
        os.makedirs(f'Inference_model/{name_weight}')

    # Go through all the test images
    for name_img in os.listdir(PATH_test_images):
        img_path = os.path.join(PATH_test_images, name_img)
        img_segmented = Image.open(img_path)

        # Prepare signs for inference
        signs_path = os.path.join(PATH_signs, name_img)
        signs_segmented = os.listdir(signs_path)
        batch_signs = []
        for segmented_sign in signs_segmented:
            sign_path = os.path.join(signs_path, segmented_sign)
            sign_img = Image.open(sign_path).resize((KERNEL, KERNEL))
            sign_np = np.array(sign_img)/255
            batch_signs.append(sign_np)
        batch_signs = np.array(batch_signs)

        # Make prediction
        predictions = model.predict(batch_signs)
        y_pred = np.argmax(predictions, axis=1)
        y_prob = predictions[np.arange(len(y_pred)), y_pred]
        print(name_img, 'y_pred:', y_pred, y_prob)

        # Creating final segmented image with classified road signs
        final_image(img_segmented, name_img, y_pred, y_prob)

if __name__ == "__main__":
    main()
