if __name__ == '__main__':
    import os
    import pickle
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    from Training_model.mymodel import MyModel

    from keras.preprocessing.image import ImageDataGenerator
    from sklearn.metrics import confusion_matrix, classification_report
    from sklearn.metrics import accuracy_score, f1_score

# Dataset of images
PATH_root = 'E:\DataSets\Traffic Sign - Detection&Recognation\Traffic_Sign - 200 classes'
PATH_train = os.path.join(PATH_root, 'Train')
# Number of road sign classes
CLASSES = len(os.listdir(PATH_train))
# Set parameters for shape of the resized image
KERNEL = 64
input_shape = (None, KERNEL, KERNEL, 3)

# Load test dataframe
test_data = pd.read_csv('Training_model\split_dfs\df_test.csv')

# Load test generator parameters
with open(r'Training_model\test_generator_params.pkl', 'rb') as file:
    test_generator_params = pickle.load(file)

# Create test generator
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_dataframe(**test_generator_params)

# Load the model and pre-trained weights
model = MyModel(CLASSES, input_shape)
model.build(input_shape)
name_weight = 'best_weights'
# root_weight = 'Training_model/trained_models_tf'
model.load_weights(f'Training_model/trained_models_tf/{name_weight}.hdf5')

# Evaluating metrics
y_true = test_data.class_id.values
print('y_true:', y_true)
predictions = model.predict(test_generator)
y_pred = np.argmax(predictions, axis=1)
print('y_pred:', y_pred)
scores = accuracy_score(y_true, y_pred)
print(classification_report(y_true, y_pred))
print(f"Accuracy: {round(scores * 100, 2)}%")

# Confusion matrix
c_m = confusion_matrix(y_true, y_pred)
class_totals = np.sum(c_m, axis=1)   # Total number of objects in each class
matrix_percent = (c_m.T / class_totals).T * 100   # Converting Matrix to percentage

# Create folder
if os.path.exists(f'Testing_model/res_conf_matrix/{name_weight}'):
    pass
else:
    os.makedirs(f'Testing_model/res_conf_matrix/{name_weight}')

# Build and save several smaller confusion matrix
cm_size = 50    # confusion matrix on 'cm_size' classes
unique_labels = np.unique(y_true)
for i in range(4):
    plt.figure(figsize=(18, 10))
    c_m_small = matrix_percent[cm_size*i:cm_size*(i+1), cm_size*i:cm_size*(i+1)]
    mask = c_m_small == 0.0    # mask for zero values
    sns.heatmap(c_m_small, annot=True, cbar=False, mask=mask,
                annot_kws={'size': 7}, fmt='.0f',
                cmap='summer_r',   # coolwarm
                xticklabels=unique_labels[cm_size * i:cm_size * (i + 1)],
                yticklabels=unique_labels[cm_size * i:cm_size * (i + 1)]
                )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix part_{i+1}')
    plt.savefig(f'Testing_model/res_conf_matrix/{name_weight}/Confusion Matrix part_{i+1}')
    i += cm_size
