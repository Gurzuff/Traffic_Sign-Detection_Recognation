import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

from Training_model.classes.mymodel import MyModel_32, MyModel_48, MyModel_64


def final_image(PATH_labels: str,
                PATH_signs: str,
                img_segmented: Image.Image,
                name_img: str,
                y_pred: list[int],
                y_prob: list[float],
                threshold: float = 0.4) -> None:
    '''
    Creates a composite image displaying the original segmented image and segmented signs.
    Parameters:
    - PATH_labels: Path to the folder with road sign labels
    - PATH_signs: Path to the folder with segmented road signs
    - img_segmented: Original segmented image.
    - name_img: Image name identifier.
    - y_pred: Predicted classes for segmented signs.
    - y_prob: Predicted probabilities for segmented signs.
    - threshold: Probability threshold for displaying meta images.
    Saves the resulting composite image to the folder with "model_weights" names.
    '''
    # Create a figure with a specific layout for images (4 - big, 1 - small)
    fig = plt.figure(figsize=(18, 6))
    gs = GridSpec(1, 9, width_ratios=np.array([4] + [1] * 8))

    # Display the original segmented image
    ax_large = plt.subplot(gs[0])
    ax_large.imshow(img_segmented, cmap='viridis')
    ax_large.set_title(f'Segmented image {img_segmented.size}')
    ax_large.axis('off')

    # Create nested GridSpec for the right part
    gs_right = GridSpecFromSubplotSpec(2, len(y_pred), gs[1:], width_ratios=[1] * len(y_pred))

    # Display segmented signs in the first row
    signs_path = os.path.join(PATH_signs, name_img)
    signs_segmented = os.listdir(signs_path)
    for road_sign, sign_prob, i in zip(signs_segmented, y_prob, range(len(y_pred))):
        sign_path = os.path.join(signs_path, road_sign)
        road_sign = Image.open(sign_path)
        ax_small = plt.subplot(gs_right[i])
        ax_small.imshow(road_sign, cmap='viridis')
        ax_small.set_title(f'Segmented sign {i+1}')
        ax_small.axis('off')

    # Display meta images in the second row
    low_quality_img = Image.open('../readme_files/Low_quality.png')
    for sign_class, sign_prob, i in zip(y_pred, y_prob, range(len(y_pred), 2 * len(y_pred))):
        sign_name = df_classes.sign_name[df_classes.sign_class == sign_class].values
        ax_small = plt.subplot(gs_right[i])
        if sign_prob > threshold:
            meta_path = os.path.join(PATH_labels, f'{sign_class}.png')
            meta_img = Image.open(meta_path)
            ax_small.imshow(meta_img, cmap='viridis')
            ax_small.set_title(f'{sign_name} (prob.: {format(sign_prob*100, ".0f")}%)')   #
            ax_small.axis('off')
        else:
            ax_small.imshow(low_quality_img, cmap='viridis')
            ax_small.set_title(f'Low quality!')
            ax_small.axis('off')

    plt.tight_layout()
    plt.savefig(f'{name_weight}/{name_img}')

# CUSTOM PARAMETERS
CLASSES = 200

# Choose the weight of pre-trained models ('Training_model\models_weights')
name_weight = 'model_43M_32x32_9966'
KERNEL = int(name_weight[10:12])
input_shape = (None, KERNEL, KERNEL, 3)

# Choose the model (MyModel_32, MyModel_48, MyModel_64) and load pre-trained weights
model = MyModel_32(CLASSES, input_shape)
model.build(input_shape)
model.load_weights(f'../Training_model/models_weights/{name_weight}.hdf5')

# Paths to sign labels, segmented test images and segmented road signs
PATH_sign_labels = '../Classes_description/class_labels'
PATH_segmented_images = '../Data/segmented_images'
PATH_segmented_signs = f'../Data/segmented_signs/model_{KERNEL}'

# Load DF with full class names
df_classes = pd.read_csv('../Classes_description/sign_names.csv')

def main():
    # Create a folder to save final classified road sign images
    os.makedirs(name_weight, exist_ok=True)

    # Go through all the test images
    for name_img in os.listdir(PATH_segmented_images):
        img_path = os.path.join(PATH_segmented_images, name_img)
        img_segmented = Image.open(img_path)

        # Prepare batch of signs
        batch_signs = []
        signs_path = os.path.join(PATH_segmented_signs, name_img)
        for segmented_sign in os.listdir(signs_path):
            sign_path = os.path.join(signs_path, segmented_sign)
            sign_img = Image.open(sign_path).resize((KERNEL, KERNEL))
            sign_np = np.array(sign_img)/255
            batch_signs.append(sign_np)
        batch_signs = np.array(batch_signs)

        # Make prediction
        predictions = model.predict(batch_signs)
        y_pred = np.argmax(predictions, axis=1)
        y_prob = predictions[np.arange(len(y_pred)), y_pred]
        print(name_img, 'y_pred:', y_pred, 'y_pred:', y_prob)

        # Creating final segmented image with classified road signs
        final_image(PATH_sign_labels, PATH_segmented_signs, img_segmented, name_img, y_pred, y_prob)

if __name__ == "__main__":
    main()
