if __name__ == '__main__':
    import os
    import pickle
    import pandas as pd
    import matplotlib.pyplot as plt

    from Training_model.mymodel import MyModel

    import tensorflow as tf
    from keras.preprocessing.image import ImageDataGenerator
    from keras.optimizers import Adam, RMSprop
    from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    from sklearn.model_selection import train_test_split

# Dataset of images
PATH_root = 'E:\DataSets\Traffic Sign - Detection&Recognation\Traffic_Sign - 200 classes'
PATH_train = os.path.join(PATH_root, 'Train')
# Instances of signs
PATH_meta = os.path.join(PATH_root, 'Meta')

SEED = 33
BATCH = 192
# Number of road sign classes
CLASSES = len(os.listdir(PATH_train))
print('Number classes:', CLASSES)
# Set parameters for shape of the resized image
KERNEL = 64
target_size = (KERNEL, KERNEL)
input_shape = (None, KERNEL, KERNEL, 3)

# Check GPU available
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# DF [image_paths, class_id]
images_path_list = []
for root, dirs, imgs in os.walk(PATH_train):
    for subdir in dirs:
        PATH_subdir = os.path.join(root, subdir)
        for img in os.listdir(PATH_subdir):
            PATH_image = os.path.join(subdir, img)
            images_path_list.append((PATH_image, int(subdir)))
df_road_sign = pd.DataFrame(images_path_list, columns=['img_path', 'class_id'])

# Split data: train, valid, test
train_data, temp_data = train_test_split(df_road_sign, test_size=0.3, random_state=SEED,
                                         stratify=df_road_sign['class_id'], shuffle=True)
valid_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=SEED,
                                         stratify=temp_data['class_id'], shuffle=True)
print(f'train_data={train_data.shape[0]} img, valid_data={valid_data.shape[0]} img, test_data={test_data.shape[0]} img')

# Save DFs
train_data.to_csv('Training_model/split_dfs/df_train.csv', header=True, index=None)
valid_data.to_csv('Training_model/split_dfs/df_valid.csv', header=True, index=None)
test_data.to_csv('Training_model/split_dfs/df_test.csv', header=True, index=None)
df_road_sign.to_csv('Training_model/split_dfs/df_road_sign.csv', header=True, index=None)

# Augmentation
aug_datagen = ImageDataGenerator(rescale=1./255,
                                 rotation_range=15,
                                 zoom_range=0.05,
                                 channel_shift_range=0.05,
                                 brightness_range=[0.9, 1.1])

# Train generator
train_generator = aug_datagen.flow_from_dataframe(dataframe=train_data,
                                                  directory=PATH_train,
                                                  x_col='img_path',
                                                  y_col='class_id',
                                                  batch_size=BATCH,
                                                  target_size=target_size,
                                                  class_mode='raw',
                                                  shuffle=True,
                                                  seed=SEED)

# Validation generator
valid_datagen = ImageDataGenerator(rescale=1./255)
valid_generator = valid_datagen.flow_from_dataframe(dataframe=valid_data,
                                                    directory=PATH_train,
                                                    x_col='img_path',
                                                    y_col='class_id',
                                                    batch_size=BATCH,
                                                    target_size=target_size,
                                                    class_mode='raw',
                                                    shuffle=True,
                                                    seed=SEED)

# Test generator
test_datagen = ImageDataGenerator(rescale=1./255)
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
with open('Training_model/test_generator_params.pkl', 'wb') as file:
    pickle.dump(test_generator_params, file)

# Model
model = MyModel(CLASSES, input_shape)
model.build(input_shape)

# Number of parameters
total_params = 0
for layer in model.layers:
    total_params += layer.count_params()
total_params_M = round(total_params/1e6)
print(f"Total number of parameters: {total_params_M} millions")

# Calback: Reduce LR
reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', mode='min',     # 'val_accuracy', 'max',
                                   factor=0.5, patience=3, verbose=1,  # 0.7, 2
                                   cooldown=0, min_lr=1e-8)

# Calback: Best weights
weight_path = "Training_model/trained_models_tf/best_weights.hdf5"
checkpoint = ModelCheckpoint(weight_path, monitor='val_accuracy', mode='max',
                             verbose=1, save_best_only=True, save_weights_only=True)

# Calback: EarlyStopping
earlystop = EarlyStopping(monitor="val_loss", mode="min", verbose=2, patience=12)

callbacks_list = [checkpoint, reduceLROnPlat, earlystop]

# Model compiling
LR = 1e-4
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=RMSprop(learning_rate=LR),
              metrics=['accuracy'])

# Model training
EPOCHS = 80
STEPS_train = 64
STEPS_valid = 32
history = model.fit(train_generator,
                    epochs=EPOCHS,
                    steps_per_epoch=STEPS_train,
                    validation_data=valid_generator,
                    validation_steps=STEPS_valid,   # (71.9 steps - max)
                    callbacks=callbacks_list,
                    verbose=1)

# Evaluate the model on the test data
test_results = model.evaluate(test_generator)
print('Model evaluate: [Loss, Accuracy] =', test_results)

# Save model with best weights
name_model = f'model_{total_params_M}M_{KERNEL}x{KERNEL}_{EPOCHS}ep'
model.load_weights(f'Training_model/trained_models_tf/{name_model}.hdf5')
# model.save(f'Training_model/trained_models_tf/{name_model}', save_format='tf')

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
plt.savefig(f'Training_model/res_metrics_png/metrics_{total_params_M}M_{KERNEL}x{KERNEL}_{EPOCHS}ep_log.png')
plt.show()
