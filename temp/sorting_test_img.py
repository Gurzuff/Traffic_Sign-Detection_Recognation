if __name__=='__main__':
    import os
    from PIL import Image
    import pandas as pd
    from tqdm import tqdm

'''
This file used for sorting images from the Test folder to sign classes folders.
'''

PATH_test = 'E:\DataSets\Traffic Sign - Detection&Recognation\Traffic_Sign - 205 classes (GTSRB+162 custom classes)\Test'
PATH_df = 'E:\DataSets\Traffic Sign - Detection&Recognation\Traffic_Sign - 205 classes (GTSRB+162 custom classes)\Test_labels.csv'

df = pd.read_csv(PATH_df)
df['Path'] = df['Path'].apply(lambda x: x[5:])
df['Path'], df['ClassId'] = df['ClassId'], df['Path']

test_dict = dict(df.values)

for img_name in tqdm(os.listdir(PATH_test)):
    img_class = str(test_dict[img_name])
    PATH_class = os.path.join(PATH_test, img_class)
    if os.path.exists(PATH_class):
        pass
    else:
        os.makedirs(PATH_class)
    PATH_img = os.path.join(PATH_test, img_name)
    img = Image.open(PATH_img)
    PATH_img_save = os.path.join(PATH_class, img_name)
    img.save(PATH_img_save)
