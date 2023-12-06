if __name__ == '__main__':
    import os
    import pickle
    import pandas as pd
    import matplotlib.pyplot as plt

    import tensorflow as tf
    from keras import backend as K
    from keras.preprocessing.image import ImageDataGenerator
    from keras.optimizers import Adam, RMSprop
    from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense
    from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    from sklearn.model_selection import train_test_split

    class MyModel(tf.keras.Model):
        def __init__(self, classes, input_shape):
            super(MyModel, self).__init__()

            # first block-layers
            self.conv1 = Conv2D(64, 3, 1, activation='relu', padding='same', input_shape=input_shape)
            self.bn1 = BatchNormalization()
            self.conv2 = Conv2D(128, 3, 1, activation='relu', padding='same')
            self.bn2 = BatchNormalization()
            self.maxpool1 = MaxPooling2D()
            self.dropout1 = Dropout(0.25)
            # second block-layers
            self.conv3 = Conv2D(128, 3, 1, activation='relu', padding='same')
            self.bn3 = BatchNormalization()
            self.conv4 = Conv2D(256, 3, 1, activation='relu', padding='same')
            self.bn4 = BatchNormalization()
            self.maxpool2 = MaxPooling2D()
            self.dropout2 = Dropout(0.25)
            # head layers
            self.flatten = Flatten()
            self.dense1 = Dense(512, activation='relu')
            self.bn5 = BatchNormalization()
            self.dropout3 = Dropout(0.4)
            self.dense2 = Dense(classes, activation='softmax')

        def call(self, inputs, training=None, mask=None):
            x = inputs
            # first block-layers
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.maxpool1(x)
            x = self.dropout1(x)
            # second block-layers
            x = self.conv3(x)
            x = self.bn3(x)
            x = self.conv4(x)
            x = self.bn4(x)
            x = self.maxpool2(x)
            x = self.dropout2(x)
            # head layers
            x = self.flatten(x)
            x = self.dense1(x)
            x = self.bn5(x)
            x = self.dropout3(x)
            x = self.dense2(x)
            return x

PATH_root = 'E:\DataSets\Traffic Sign - Detection&Recognation\Traffic_Sign - 200 classes'
PATH_train = os.path.join(PATH_root, 'Train')
PATH_meta = os.path.join(PATH_root, 'Meta')

SEED = 33
BATCH = 256
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
                                         stratify=df_road_sign['class_id'], shuffle=True
                                         )
valid_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=SEED,
                                         stratify=temp_data['class_id'], shuffle=True
                                         )
print(f'train_data={train_data.shape[0]} img, valid_data={valid_data.shape[0]} img, test_data={test_data.shape[0]} img')
# Save DFs
train_data.to_csv('2. training_model/split_dfs/df_train.csv', header=True, index=None)
valid_data.to_csv('2. training_model/split_dfs/df_valid.csv', header=True, index=None)
test_data.to_csv('2. training_model/split_dfs/df_test.csv', header=True, index=None)
df_road_sign.to_csv('2. training_model/split_dfs/df_road_sign.csv', header=True, index=None)

# Augmentation
aug_datagen = ImageDataGenerator(rescale=1./255,
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

# Save Test generator parameters
test_generator_params = {'dataframe': test_data,
                         'directory': PATH_train,
                         'x_col': 'img_path',
                         'y_col': 'class_id',
                         'batch_size': BATCH,
                         'target_size': target_size,
                         'class_mode': 'raw',
                         'shuffle': False,
                         'seed': SEED}
# Save
with open('2. training_model/test_generator_params.pkl', 'wb') as file:
    pickle.dump(test_generator_params, file)

# Model
model_9M = MyModel(CLASSES, input_shape)
model_9M.build(input_shape)
# print(model_9M_new.summary())

# Calback: Reduce LR
reduceLROnPlat = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5,
                                   patience=2, verbose=1, mode='max',
                                   cooldown=0, min_lr=1e-8)   # min_delta=0.0001,
# Calback: EarlyStopping
earlystop = EarlyStopping(monitor="val_loss", mode="min", verbose=2, patience=12)
# Calback: Best weights
weight_path = "2. training_model/trained_models_tf/best_weights.hdf5"
checkpoint = ModelCheckpoint(weight_path, monitor='val_accuracy', mode='max',
                             verbose=1, save_best_only=True, save_weights_only=True)
callbacks_list = [checkpoint, reduceLROnPlat, earlystop]

# Model compiling
LR = 1e-4
model_9M.compile(loss="sparse_categorical_crossentropy",
                 optimizer=RMSprop(learning_rate=LR),   # RMSprop, Adam
                 metrics=['accuracy', 'categorical_accuracy']
                 )
# Model training
EPOCHS = 12
STEPS_train = 64
STEPS_valid = 32
history = model_9M.fit(train_generator,
                       epochs=EPOCHS,
                       steps_per_epoch=STEPS_train,
                       validation_data=valid_generator,
                       validation_steps=STEPS_valid,   # (71.9 steps - max)
                       callbacks=callbacks_list,
                       verbose=1
                       )
# print('Model evaluate: [Loss, Accuracy] =', model_9M_new.evaluate(test_generator))

# Model saving
name_model = f'model_9M_{KERNEL}_{EPOCHS}'
model_9M.save(f'2. training_model/trained_models_tf/{name_model}', save_format='tf')

# Build plot accuracy and loss by history
acc, val_acc = history.history['accuracy'], history.history['val_accuracy']
# f1, val_f1 = history.history['f1_score'], history.history['val_f1_score']
loss, val_loss = history.history['loss'], history.history['val_loss']
epochs = range(1, len(acc) + 1)

# Plot accuracy and f1_score
fig, axs = plt.subplots(1, 2, figsize=(15, 5))
axs[0].plot(epochs, acc, 'b', label='Training acc')
axs[0].plot(epochs, val_acc, 'g', label='Validation acc')
# axs[0].plot(epochs, f1, 'b', linestyle='--', label='Training f1')
# axs[0].plot(epochs, val_f1, 'g', linestyle='--', label='Validation f1')
axs[0].set_title('Train, Valid: Accuracy, F1_score')
axs[0].legend()

# Plot loss
axs[1].plot(epochs, loss, 'b', label='Training loss')
axs[1].plot(epochs, val_loss, 'g', label='Validation loss')
axs[1].set_title('Train, Valid: CategoryCrossEntropy')
axs[1].legend()

# Save the figure
plt.savefig(f'2. training_model/res_metrics_png/metrics_{EPOCHS}epochs_{KERNEL}size.png')
plt.show()

