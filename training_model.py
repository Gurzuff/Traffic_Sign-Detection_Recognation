if __name__ == '__main__':
    import os
    import json
    import pickle
    import pandas as pd
    import matplotlib.pyplot as plt

    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split

    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense

    SEED = 33
    epochs = 10
    batch = 256
    kernel = 32
    classes = 206
    learning_rate = 3e-4
    target_size = (kernel, kernel)
    input_shape = (None, kernel, kernel, 3)
    url_train = r'E:\DataSets\Traffic Sign\Traffic_Sign - 205 classes (GTSRB+162 custom classes)\Train'
    url_test = r'E:\DataSets\Traffic Sign\Traffic_Sign - 205 classes (GTSRB+162 custom classes)\Test'
    url_test_labels = r'E:\DataSets\Traffic Sign\Traffic_Sign - 205 classes (GTSRB+162 custom classes)\Test_labels.csv'
    test_labels_df = pd.read_csv(url_test_labels)

    # Check GPU available
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    # Turn-on all processor cores
    # os.environ["TF_NUM_INTRAOP_THREADS"] = "0"

    # Train generator preparation
    train_datagen = ImageDataGenerator(rescale=1./255)   #
    train_generator = train_datagen.flow_from_directory(url_train,
                                                        target_size=target_size,
                                                        batch_size=batch,
                                                        class_mode='sparse'
                                                        )
    # Dictionary of mapping indexes and class names from the generator
    class_mapping = train_generator.class_indices
    with open('2. training_model/class_mapping.json', 'w') as f:
        json.dump(class_mapping, f)


    # Split test data on validation and test
    test_labels_df['Path'] = test_labels_df['Path'].apply(lambda x: x[5:])  # remove part of path: 'Test/'
    test_labels_df['ClassId'] = test_labels_df['ClassId'].astype('str')
    valid_df, test_df = train_test_split(test_labels_df,
                                         test_size=0.5,
                                         stratify=test_labels_df['ClassId'],
                                         random_state=SEED
                                         )
    # Validation generator preparation
    valid_datagen = ImageDataGenerator(rescale=1./255)   #
    valid_generator = valid_datagen.flow_from_dataframe(dataframe=valid_df,
                                                        directory=url_test,
                                                        x_col='Path',
                                                        y_col='ClassId',
                                                        batch_size=batch,
                                                        target_size=target_size,
                                                        class_mode='sparse',
                                                        shuffle=False
                                                        )
    # Save test generator parameters
    test_generator_params = {'dataframe': test_df,
                             'directory': url_test,
                             'x_col': 'Path',
                             'y_col': 'ClassId',
                             'batch_size': batch,
                             'target_size': target_size,
                             'class_mode': 'sparse',
                             'shuffle': False
                             }
    with open('2. training_model/test_generator_params.pkl', 'wb') as file:
        pickle.dump(test_generator_params, file)

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
            self.dropout3 = Dropout(0.5)
            self.dense2 = Dense(classes, activation='softmax')

        def call(self, inputs, training=None, mask=None):
            x = inputs
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.maxpool1(x)
            x = self.dropout1(x)

            x = self.conv3(x)
            x = self.bn3(x)
            x = self.conv4(x)
            x = self.bn4(x)
            x = self.maxpool2(x)
            x = self.dropout2(x)

            x = self.flatten(x)
            x = self.dense1(x)
            x = self.bn5(x)
            x = self.dropout3(x)
            x = self.dense2(x)
            return x

    model_9M = MyModel(classes, input_shape)
    model_9M.build(input_shape)
    # print(model.summary())

    model_9M.compile(loss="sparse_categorical_crossentropy",
                     optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                     metrics=['accuracy']
                     )

    history = model_9M.fit(train_generator,
                           validation_data=valid_generator,
                           epochs=epochs,
                           verbose=1
                           )
    # Save model
    model_9M.save(f'trained_models_tf/model_9M_{target_size[0]}_{epochs}', save_format='tf')

    # Build plot accuracy and loss by history
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, acc, 'b', label='Training acc')
    plt.plot(epochs, val_acc, 'g', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.savefig(f'res_metrics_png/accuracy_{len(acc)}_epochs.png')

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'g', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig(f'res_metrics_png/loss_{len(acc)}_epochs.png')
    plt.show()
