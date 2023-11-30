if __name__ == '__main__':
    import os
    import pickle
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, classification_report
    from sklearn.metrics import accuracy_score, f1_score
    from tensorflow.keras.models import load_model
    from keras.preprocessing.image import ImageDataGenerator

    # Load test generator parameters
    with open('2. training_model/test_generator_params.pkl', 'rb') as file:
        test_generator_params = pickle.load(file)

    # Create test generator
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_dataframe(**test_generator_params)

    # Load the model and make prediction
    name_model = 'model_9M_32_10'    # base_model_9M, model_9M_48_5
    root_model = '../../2. training_model/src/trained_models_tf'
    model = load_model(os.path.join(root_model, name_model))
    predictions = model.predict(test_generator)

    # Evaluating metrics
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes
    scores = accuracy_score(y_true, y_pred)
    print(classification_report(y_true, y_pred))
    print(f"Accuracy: {round(scores * 100, 2)}%")

    # Confusion matrix
    c_m = confusion_matrix(y_true, y_pred)
    class_totals = np.sum(c_m, axis=1)   # Total number of objects in each class
    matrix_percent = (c_m.T / class_totals).T * 100   # Converting Matrix to percentage

    # Create folder
    if os.path.exists(f'res_conf_matrix/{name_model}'):
        pass
    else:
        os.makedirs(f'res_conf_matrix/{name_model}')

    # Build and save several smaller confusion matrix
    cm_size = 52    # confusion matrix on 'cm_size' classes
    unique_labels = np.unique(test_generator.classes)
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
        plt.savefig(f'res_conf_matrix/{name_model}/Confusion Matrix part_{i+1}')
        i += cm_size
