import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score

from Training_model.classes.mymodel import MyModel_32, MyModel_48, MyModel_64

def save_conf_matrix(y_true, y_pred, name_weight):
    '''
    Generates and saves confusion matrices for the classification results.
    Parameters:
    - y_true: True class labels.
    - y_pred: Predicted class labels.
    - name_weight: Name identifier for the model.
    Saves four smaller confusion matrices for a more detailed analysis.
    '''
    # Create folder to save confusion matrix
    os.makedirs(f'{name_weight}', exist_ok=True)

    # Confusion matrix
    c_m = confusion_matrix(y_true, y_pred)
    # Total number of objects in each class
    class_totals = np.sum(c_m, axis=1)
    # Converting matrix to percentage
    matrix_percent = (c_m.T / class_totals).T * 100

    # Build and save 4 smaller confusion matrix (200/4 = 50 classes)
    c_m_size = 50
    unique_labels = np.unique(y_true)
    for i in range(4):
        plt.figure(figsize=(18, 10))
        # smaller confusion matrix (50 classes)
        c_m_small = matrix_percent[c_m_size*i:c_m_size*(i+1), c_m_size*i:c_m_size*(i+1)]
        # mask for zero values
        mask = c_m_small == 0.0
        sns.heatmap(c_m_small, annot=True, cbar=False, mask=mask,
                    annot_kws={'size': 7}, fmt='.0f',
                    cmap='summer_r',
                    xticklabels=unique_labels[c_m_size * i:c_m_size * (i + 1)],
                    yticklabels=unique_labels[c_m_size * i:c_m_size * (i + 1)]
                    )
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix part_{i+1}')
        plt.savefig(f'{name_weight}/Confusion Matrix part_{i+1}')
        i += c_m_size

# CUSTOM PARAMETERS
# Choose the weight of pre-trained models ('Training_model\models_weights')
name_weight = 'model_43M_64x64_9984'
KERNEL = int(name_weight[10:12])
input_shape = (None, KERNEL, KERNEL, 3)

# Path to dataset of images
PATH_data = 'E:\DataSets\Traffic_Sign - 200 classes\data'

# Number of road sign classes
CLASSES = 200

# Choose the model (MyModel_32, MyModel_48, MyModel_64) and load pre-trained weights
model = MyModel_64(CLASSES, input_shape)
model.build(input_shape)
model.load_weights(f'../Training_model/models_weights/{name_weight}.hdf5')

# Load test dataframe
test_data = pd.read_csv('../Training_model/split_dfs/df_test.csv')

# Load test generator parameters
with open(f'../Training_model/test_generator_params/test_generator_{KERNEL}.pkl', 'rb') as file:
    test_generator_params = pickle.load(file)

def main():
    # Create test generator
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_dataframe(**test_generator_params)

    # Evaluating accuracy
    y_true = test_data.class_id.values
    predictions = model.predict(test_generator)
    y_pred = np.argmax(predictions, axis=1)
    scores = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {round(scores * 100, 2)}%")

    # Weighted average via classification_report
    report = classification_report(y_true, y_pred, output_dict=True)
    weighted_avg = report['weighted avg']
    print(f"Weighted-average: "
          f"Precision={weighted_avg['precision']:.4f}, "
          f"Recall={weighted_avg['recall']:.4f}, "
          f"F1-Score={weighted_avg['f1-score']:.4f}")
    macro_avg = report['macro avg']
    print(f"Macro-average: "
          f"Precision={macro_avg['precision']:.4f}, "
          f"Recall={macro_avg['recall']:.4f}, "
          f"F1-Score={macro_avg['f1-score']:.4f}")

    # Create and save confusion matrix:
    save_conf_matrix(y_true, y_pred, name_weight)

if __name__ == '__main__':
    main()
