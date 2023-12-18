import tensorflow as tf
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense

#  6Conv_2MaxPool: 32 -> 16 -> 8; 2xDense(512, 512) - 43M parameters
class MyModel_32(tf.keras.Model):
    def __init__(self, classes, input_shape):
        super(MyModel_32, self).__init__()
        # first block-layers
        self.conv1 = Conv2D(128, 3, 1, activation='relu', padding='same', input_shape=input_shape)
        self.bn1 = BatchNormalization()
        self.conv2 = Conv2D(256, 3, 1, activation='relu', padding='same')
        self.bn2 = BatchNormalization()
        self.maxpool1 = MaxPooling2D()
        self.dropout1 = Dropout(0.25)
        # second block-layers
        self.conv3 = Conv2D(256, 3, 1, activation='relu', padding='same')
        self.bn3 = BatchNormalization()
        self.conv4 = Conv2D(512, 3, 1, activation='relu', padding='same')
        self.bn4 = BatchNormalization()
        self.maxpool2 = MaxPooling2D()
        self.dropout2 = Dropout(0.25)
        # third block-layers
        self.conv5 = Conv2D(512, 3, 1, activation='relu', padding='same')
        self.bn5 = BatchNormalization()
        self.conv6 = Conv2D(1024, 3, 1, activation='relu', padding='same')
        self.bn6 = BatchNormalization()
        # self.maxpool3 = MaxPooling2D()
        self.dropout3 = Dropout(0.25)
        # head layers
        self.flatten = Flatten()
        self.dense1 = Dense(512, activation='relu')
        self.bn7 = BatchNormalization()
        self.dropout4 = Dropout(0.25)
        self.dense2 = Dense(512, activation='relu')
        self.bn8 = BatchNormalization()
        self.dropout5 = Dropout(0.35)
        self.dense3 = Dense(classes, activation='softmax')

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
        # third block-layers
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.conv6(x)
        x = self.bn6(x)
        # x = self.maxpool3(x)
        x = self.dropout3(x)
        # head layers
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.bn7(x)
        x = self.dropout4(x)
        x = self.dense2(x)
        x = self.bn8(x)
        x = self.dropout5(x)
        x = self.dense3(x)
        return x


#  6Conv_2MaxPool: 48 -> 24 -> 12: 2xDense(512, 512) - 85M parameters
class MyModel_48(tf.keras.Model):
    def __init__(self, classes, input_shape):
        super(MyModel_48, self).__init__()
        # first block-layers
        self.conv1 = Conv2D(128, 3, 1, activation='relu', padding='same', input_shape=input_shape)
        self.bn1 = BatchNormalization()
        self.conv2 = Conv2D(256, 3, 1, activation='relu', padding='same')
        self.bn2 = BatchNormalization()
        self.maxpool1 = MaxPooling2D()
        self.dropout1 = Dropout(0.25)
        # second block-layers
        self.conv3 = Conv2D(256, 3, 1, activation='relu', padding='same')
        self.bn3 = BatchNormalization()
        self.conv4 = Conv2D(512, 3, 1, activation='relu', padding='same')
        self.bn4 = BatchNormalization()
        self.maxpool2 = MaxPooling2D()
        self.dropout2 = Dropout(0.25)
        # third block-layers
        self.conv5 = Conv2D(512, 3, 1, activation='relu', padding='same')
        self.bn5 = BatchNormalization()
        self.conv6 = Conv2D(1024, 3, 1, activation='relu', padding='same')
        self.bn6 = BatchNormalization()
        # self.maxpool3 = MaxPooling2D()
        self.dropout3 = Dropout(0.25)
        # head layers
        self.flatten = Flatten()
        self.dense1 = Dense(512, activation='relu')
        self.bn7 = BatchNormalization()
        self.dropout4 = Dropout(0.25)
        self.dense2 = Dense(512, activation='relu')
        self.bn8 = BatchNormalization()
        self.dropout5 = Dropout(0.5)
        self.dense3 = Dense(classes, activation='softmax')

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
        # third block-layers
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.conv6(x)
        x = self.bn6(x)
        # x = self.maxpool3(x)
        x = self.dropout3(x)
        # head layers
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.bn7(x)
        x = self.dropout4(x)
        x = self.dense2(x)
        x = self.bn8(x)
        x = self.dropout5(x)
        x = self.dense3(x)
        return x


#  6Conv_3MaxPool: 64 -> 32 -> 16 -> 8: 2xDense(1024, 1024) - 43M parameters
class MyModel_64(tf.keras.Model):
    def __init__(self, classes, input_shape):
        super(MyModel_64, self).__init__()
        # first block-layers
        self.conv1 = Conv2D(128, 3, 1, activation='relu', padding='same', input_shape=input_shape)
        self.bn1 = BatchNormalization()
        self.conv2 = Conv2D(256, 3, 1, activation='relu', padding='same')
        self.bn2 = BatchNormalization()
        self.maxpool1 = MaxPooling2D()
        self.dropout1 = Dropout(0.25)
        # second block-layers
        self.conv3 = Conv2D(256, 3, 1, activation='relu', padding='same')
        self.bn3 = BatchNormalization()
        self.conv4 = Conv2D(512, 3, 1, activation='relu', padding='same')
        self.bn4 = BatchNormalization()
        self.maxpool2 = MaxPooling2D()
        self.dropout2 = Dropout(0.25)
        # third block-layers
        self.conv5 = Conv2D(512, 3, 1, activation='relu', padding='same')
        self.bn5 = BatchNormalization()
        self.conv6 = Conv2D(1024, 3, 1, activation='relu', padding='same')
        self.bn6 = BatchNormalization()
        self.maxpool3 = MaxPooling2D()
        self.dropout3 = Dropout(0.25)
        # head layers
        self.flatten = Flatten()
        self.dense1 = Dense(512, activation='relu')
        self.bn7 = BatchNormalization()
        self.dropout4 = Dropout(0.3)
        self.dense2 = Dense(1024, activation='relu')
        self.bn8 = BatchNormalization()
        self.dropout5 = Dropout(0.35)
        self.dense3 = Dense(classes, activation='softmax')

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
        # third block-layers
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.maxpool3(x)
        x = self.dropout3(x)
        # head layers
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.bn7(x)
        x = self.dropout4(x)
        x = self.dense2(x)
        x = self.bn8(x)
        x = self.dropout5(x)
        x = self.dense3(x)
        return x

