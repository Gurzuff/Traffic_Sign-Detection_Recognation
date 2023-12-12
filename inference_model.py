if __name__ == "__main__":
    import os
    import numpy as np
    from PIL import Image
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
    from Training_model.mymodel import MyModel

# Choose image
NAME_img = 'test8.PNG'   # test6.PNG, test1.JPG
ROOT_images = 'Segmentation_image\segmented_images'
img_path = os.path.join(ROOT_images, NAME_img)
img_segmented = Image.open(img_path)

#
ROOT_signs = 'Segmentation_image\segmented_sign'


# Dataset of images
PATH_root = 'E:\DataSets\Traffic Sign - Detection&Recognation\Traffic_Sign - 200 classes'
PATH_meta = os.path.join(PATH_root, 'Meta')
# Number of road sign classes
CLASSES = len(os.listdir(PATH_meta))
# Set parameters for shape of the resized image
KERNEL = 64
input_shape = (None, KERNEL, KERNEL, 3)


# Load the model and pre-trained weights
model = MyModel(CLASSES, input_shape)
model.build(input_shape)
name_weight = 'best_weights'
model.load_weights(f'Training_model/trained_models_tf/{name_weight}.hdf5')


# Prepare signs for inference
signs_path = os.path.join(ROOT_signs, NAME_img)
signs_segmented = os.listdir(signs_path)
infer_signs = []
for segmented_sign in signs_segmented:
    sign_path = os.path.join(signs_path, segmented_sign)
    sign_img = Image.open(sign_path)
    sign_np = np.array(sign_img)/255
    infer_signs.append(sign_np)

infer_signs = np.array(infer_signs)
predictions = model.predict(infer_signs)
y_pred = np.argmax(predictions, axis=1)
print('y_pred:', y_pred)


# Создаем фигуру (лист) с пропорцией изображений: 4 - большое, 1 - малые
fig = plt.figure(figsize=(18, 6))
gs = GridSpec(1, 9, width_ratios=np.array([4] + [1] * 8))

# Сегментирование знаков на общем изображении
ax_large = plt.subplot(gs[0])
ax_large.imshow(img_segmented, cmap='viridis')
ax_large.set_title('Large Image (256x256)')
ax_large.axis('off')

# Создаем вложенные GridSpec для правой части
gs_right = GridSpecFromSubplotSpec(2, len(y_pred), gs[1:], width_ratios=[1] * len(y_pred))

# Отрисовываем изображения из folder_1 в первом ряду
signs_path = os.path.join(ROOT_signs, NAME_img)
signs_segmented = os.listdir(signs_path)
for i, road_sign in enumerate(signs_segmented, start=0):
    sign_path = os.path.join(signs_path, road_sign)
    road_sign = Image.open(sign_path)
    ax_small = plt.subplot(gs_right[i])
    ax_small.imshow(road_sign, cmap='viridis')
    ax_small.set_title(f'Segmented sign {i}')
    ax_small.axis('off')

# Отрисовываем изображения из folder_2 во втором ряду
for i, sign_class in enumerate(y_pred, start=len(y_pred)):
    meta_path = os.path.join(PATH_meta, f'{sign_class}.png')
    meta_img = Image.open(meta_path)
    ax_small = plt.subplot(gs_right[i])
    ax_small.imshow(meta_img, cmap='viridis')
    ax_small.set_title(f'Predicted sign {i}')
    ax_small.axis('off')

plt.tight_layout()
plt.show()