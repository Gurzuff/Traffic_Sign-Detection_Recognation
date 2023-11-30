if __name__ == '__main__':
    import os
    import json
    import numpy as np
    import pandas as pd
    from tensorflow.keras.utils import load_img, img_to_array
    from tensorflow.keras.models import load_model

    # Load traffic sign data class and information
    df_sign_info = pd.read_csv('Directory_signs.csv')
    print(df_sign_info.head())

    # Load model madel classes by index
    class_map_root = '../../2. training_model/src/class_mapping.json'
    with open(class_map_root, 'r') as file:
        class_mapping = json.load(file)
    inv_class_mapping = {v: k for k, v in class_mapping.items()}

    # Load model
    model_root = '../../2. training_model/src/trained_models_tf'
    model_name = 'model_9M_32_10'
    model = load_model(os.path.join(model_root, model_name))


    kernel = 32
    target_size = (kernel, kernel)
    root_unknown_signs = '../../1. segmentation_image/src/3.segmented_sign'
    image_names = os.listdir(root_unknown_signs)
    for image in image_names:
        root_signs = os.path.join(root_unknown_signs, image)
        signs_names = os.listdir(root_signs)
        for sign_name in signs_names:
            root_sign = os.path.join(root_signs, sign_name)
            # img = Image.open(root_sign).resize(target_size)
            # img = np.array(img)/255
            img = load_img(root_sign, target_size=target_size)
            img = img_to_array(img)/255
            img = img.reshape((1, 32, 32, 3))
            # print(img.shape)
            pred_index = np.argmax(model.predict(img, verbose=0))
            pred_class = int(inv_class_mapping[pred_index])
            print(f'Image:{image}, {sign_name}: {pred_class} {df_sign_info.sign_group_name[df_sign_info.sign_class == str(pred_class)].values} {df_sign_info.sign_name[df_sign_info.sign_class == str(pred_class)].values}')  #{df_sign_info}



