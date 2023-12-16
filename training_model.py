import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, RMSprop
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

from Training_model.mymodel import MyModel_32, MyModel_48, MyModel_64

def split_df(PATH_data, SEED):
    '''
    Splits a dataset into training, validation, and test sets based on image paths and class IDs.
    Parameters:
    - PATH_data: Path to the dataset.
    - SEED: Seed for reproducibility.
    Returns:
    - train_data: DataFrame for training set.
    - valid_data: DataFrame for validation set.
    - test_data: DataFrame for test set.
    '''
    # create a dataframe based on the folder structure of the images dataset
    images_path_list = []
    for root, dirs, imgs in os.walk(PATH_data):
        for subdir in dirs:
            PATH_subdir = os.path.join(root, subdir)
            for img in os.listdir(PATH_subdir):
                PATH_image = os.path.join(subdir, img)
                images_path_list.append((PATH_image, int(subdir)))
    df_road_sign = pd.DataFrame(images_path_list, columns=['img_path', 'class_id'])
    print(f'Total number of images in dataset: {df_road_sign.shape[0]}')

    # Split data: train, valid, test
    train_data, temp_data = train_test_split(df_road_sign, test_size=0.3, shuffle=True, random_state=SEED,
                                             stratify=df_road_sign['class_id'])
    valid_data, test_data = train_test_split(temp_data, test_size=0.5, shuffle=True, random_state=SEED,
                                             stratify=temp_data['class_id'])
    print(f'train_data={train_data.shape[0]} images, '
          f'valid_data={valid_data.shape[0]} images, '
          f'test_data={test_data.shape[0]} images')

    # Save DFs
    train_data.to_csv('Training_model/split_dfs/df_train.csv', header=True, index=None)
    valid_data.to_csv('Training_model/split_dfs/df_valid.csv', header=True, index=None)
    test_data.to_csv('Training_model/split_dfs/df_test.csv', header=True, index=None)
    df_road_sign.to_csv('Training_model/split_dfs/df_road_sign.csv', header=True, index=None)

    return train_data, valid_data, test_data

def data_generators(train_data, valid_data, test_data, PATH_train, BATCH, target_size, SEED):
    '''
    Creates data generators for training, validation, and testing with augmentation.
    Parameters:
    - train_data: DataFrame for training set.
    - valid_data: DataFrame for validation set.
    - test_data: DataFrame for test set.
    - PATH_train: Path to the training data directory.
    - BATCH: Batch size for the generators.
    - target_size: Tuple specifying the target size of the images.
    - SEED: Seed for reproducibility.
    Returns:
    - train_generator: Data generator for training set with augmentation.
    - valid_generator: Data generator for validation set without augmentation.
    - test_generator: Data generator for test set without augmentation.
    '''
    # Augmentation
    aug_datagen = ImageDataGenerator(rescale=1. / 255,
                                     rotation_range=15,
                                     zoom_range=0.05,
                                     channel_shift_range=0.05,
                                     brightness_range=[0.9, 1.1]
                                     )
    # Train generator
    train_generator = aug_datagen.flow_from_dataframe(dataframe=train_data,
                                                      directory=PATH_train,
                                                      x_col='img_path',
                                                      y_col='class_id',
                                                      batch_size=BATCH,
                                                      target_size=target_size,
                                                      class_mode='raw',
                                                      shuffle=True,
                                                      seed=SEED
                                                      )
    # Validation generator
    valid_datagen = ImageDataGenerator(rescale=1. / 255)
    valid_generator = valid_datagen.flow_from_dataframe(dataframe=valid_data,
                                                        directory=PATH_train,
                                                        x_col='img_path',
                                                        y_col='class_id',
                                                        batch_size=BATCH,
                                                        target_size=target_size,
                                                        class_mode='raw',
                                                        shuffle=True,
                                                        seed=SEED
                                                        )
    # Test generator
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_generator_params = {'dataframe': test_data,
                             'directory': PATH_train,
                             'x_col': 'img_path',
                             'y_col': 'class_id',
                             'batch_size': BATCH,
                             'target_size': target_size,
                             'class_mode': 'raw',
                             'shuffle': False,
                             'seed': SEED}
    test_generator = test_datagen.flow_from_dataframe(**test_generator_params)
    # Save Test generator parameters
    with open(f'Testing_model/test_generator_params/test_generator_{target_size[0]}.pkl', 'wb') as file:
        pickle.dump(test_generator_params, file)

    return train_generator, valid_generator, test_generator

def metrics_chart(model, name_weight, test_generator, history, EPOCHS):
    '''
    Plots accuracy and loss metrics for a trained model.
    Parameters:
    - model: Trained model.
    - name_weight: Name of the trained model's weight file.
    - test_generator: Data generator for the test set.
    - history: Training history containing accuracy and loss values.
    - EPOCHS: Number of training epochs.
    Displays and saves plots for accuracy and loss metrics.
    '''
    # Evaluate the model on the test data
    model.load_weights(f'Training_model/models_weights/{name_weight}.hdf5')
    test_results = model.evaluate(test_generator)
    print('Model evaluate: [Loss, Accuracy] =', test_results)

    # Build plot accuracy and loss by history
    acc, val_acc = history.history['accuracy'], history.history['val_accuracy']
    loss, val_loss = history.history['loss'], history.history['val_loss']
    test_loss, test_accuracy = test_results[0], test_results[1]
    epochs = range(1, len(acc) + 1)

    # Plot accuracy - general scale
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    axs[0].plot(epochs, acc, 'b', label='Train acc')
    axs[0].plot(epochs, val_acc, 'g', label='Valid acc')
    axs[0].scatter(len(epochs) + 1, test_accuracy, c='r', marker='x', label='Test acc')
    axs[0].text(len(epochs) + 1, test_accuracy, f'{test_accuracy:.4f}', color='r', ha='left', va='bottom')
    axs[0].set_title('Train, Valid, Test: Accuracy (general scale)')
    axs[0].legend()

    # Plot accuracy - scale: [0.95, 1.00]
    axs[1].plot(epochs, acc, 'b', label='Train acc')
    axs[1].plot(epochs, val_acc, 'g', label='Valid acc')
    axs[1].scatter(len(epochs) + 1, test_accuracy, c='r', marker='x', label='Test acc')
    axs[1].text(len(epochs) + 1, test_accuracy, f'{test_accuracy:.4f}', color='r', ha='left', va='bottom')
    axs[1].set_title('Train, Valid, Test: Accuracy (scale: [0.95, 1.00]')
    axs[1].legend()
    axs[1].set_ylim(0.95, 1.0)

    # Plot loss - log scale
    axs[2].plot(epochs, loss, 'b', label='Train loss')
    axs[2].plot(epochs, val_loss, 'g', label='Valid loss')
    axs[2].scatter(len(epochs) + 1, test_loss, c='r', marker='x', label='Test loss')
    axs[2].set_title('Train, Valid, Test: CategoryCrossEntropy')
    axs[2].legend()
    axs[2].set_yscale('log')

    # Save the figure
    name_chart = name_weight[6:15]
    plt.savefig(f'Training_model/res_metrics/metrics_{name_chart}_{EPOCHS}.png')
    plt.show()

class MyEarlyStop(Callback):
    '''
    Custom callback to stop training if validation accuracy exceeds a set threshold.
    Parameters:
    - threshold: Threshold for validation accuracy to trigger early stopping.
    '''
    def __init__(self, threshold):
        super(MyEarlyStop, self).__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs={}):
        if logs.get('val_accuracy') is not None and logs.get('val_accuracy') >= self.threshold:
            print(f'\nReached {self.threshold * 100}% accuracy!')
            self.model.stop_training = True

def callback_list(weight_path, val_loss="val_loss", mode_min='min', val_accur='val_accuracy', mode_max='max', threshold=0.999):
    '''
    Creates a list of callbacks for model training.
    Parameters:
    - weight_path: Path to save model weights.
    - val_loss: Metric to monitor for early stopping based on minimum.
    - mode_min: Mode for monitoring minimum value ('min' by default).
    - val_accur: Metric to monitor for early stopping based on maximum.
    - mode_max: Mode for monitoring maximum value ('max' by default).
    - threshold: Accuracy threshold to trigger custom early stopping.
    Returns:
    - callbacks_list: List of callbacks.
    '''
    # Reduce LR
    reduceLROnPlat = ReduceLROnPlateau(monitor=val_accur, mode=mode_max,
                                       factor=0.5, patience=3, verbose=1,
                                       cooldown=0, min_lr=1e-8)
    # Save best weights
    checkpoint = ModelCheckpoint(weight_path, monitor=val_accur, mode=mode_max,
                                 verbose=1, save_best_only=True, save_weights_only=True)
    # EarlyStopping
    earlystop = EarlyStopping(monitor=val_loss, mode=mode_min, verbose=2, patience=15)
    # MyEarlyStopping when accuracy reaches "threshold"
    my_earlystop = MyEarlyStop(threshold=threshold)

    callbacks_list = [checkpoint, reduceLROnPlat, earlystop, my_earlystop]
    return callbacks_list

# Check GPU available
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# CUSTOM PARAMETERS
SEED = 33
EPOCHS = 99
KERNEL = 64
# Dict: {KERNEL: [BATCH, STEPS]}
KERNEL_dict = {32: [256, 64], 48: [128, 128], 64: [64, 128]}
BATCH = KERNEL_dict[KERNEL][0]
STEPS = KERNEL_dict[KERNEL][1]
print(f'SHAPE: {KERNEL}x{KERNEL}, BATCH: {BATCH}, STEPS: {STEPS}')
target_size = (KERNEL, KERNEL)
input_shape = (None, KERNEL, KERNEL, 3)
# Dataset of images
PATH_root = 'E:\DataSets\Traffic Sign - Detection&Recognation\Traffic_Sign - 200 classes'
PATH_train = os.path.join(PATH_root, 'Train')
# Number of road sign classes
CLASSES = len(os.listdir(PATH_train))
print('Number classes:', CLASSES)
# Set model and main parameters
model = MyModel_32(CLASSES, input_shape)
model.build(input_shape)

def main():
    # Number of parameters
    total_params = 0
    for layer in model.layers:
        total_params += layer.count_params()
    total_params_M = round(total_params/1e6)
    print(f"Total number of parameters: {total_params_M} millions")

    # Create dataframes and data_generators
    train_data, valid_data, test_data = split_df(PATH_train, SEED)
    train_generator, valid_generator, test_generator = data_generators(train_data, valid_data, test_data,
                                                                       PATH_train, BATCH, target_size, SEED
                                                                       )
    # Set a path to save best weights and create Callbacks list
    name_weight = f'model_{total_params_M}M_{KERNEL}x{KERNEL}'
    weight_path = f'Training_model/models_weights/{name_weight}.hdf5'
    callbacks_list = callback_list(weight_path)

    # Model compiling
    LR = 1e-4
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=RMSprop(learning_rate=LR),
                  metrics=['accuracy']
                  )
    # Model training
    history = model.fit(train_generator,
                        epochs=EPOCHS,
                        steps_per_epoch=STEPS,
                        validation_data=valid_generator,
                        validation_steps=min(STEPS//2, valid_data.shape[0]//BATCH),
                        callbacks=callbacks_list,
                        verbose=1
                        )
    # Build metrics plot
    metrics_chart(model, name_weight, test_generator, history, EPOCHS)

if __name__ == '__main__':
    main()
