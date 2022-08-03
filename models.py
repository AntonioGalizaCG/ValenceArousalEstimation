import tensorflow

from tensorflow.keras.models import Model, Sequential

from tensorflow.keras.layers import (BatchNormalization,
  									 Conv2D,
									 MaxPool2D,
									 Activation,
									 Dropout,
									 Dense,
									 Flatten,
									 Input,
									 add,
									 concatenate,
									 average,
                                     GRU)
from residual_block import make_basic_block_layer, make_bottleneck_layer

def vgg16(height, width, depth):
	inputShape = (height, width, depth)

	inputs = Input(shape=inputShape)
	x = inputs
	x = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="same")(x)
	x = Activation("relu")(x)

	x = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="same")(x)
	x = Activation("relu")(x)

	x = MaxPool2D(pool_size=(2, 2), strides=(2,2))(x)

	x = Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding="same")(x)
	x = Activation("relu")(x)

	x = Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding="same")(x)
	x = Activation("relu")(x)

	x = MaxPool2D(pool_size=(2, 2), strides=(2,2))(x)

	x = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding="same")(x)
	x = Activation("relu")(x)

	x = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding="same")(x)
	x = Activation("relu")(x)

	x = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding="same")(x)
	x = Activation("relu")(x)

	x = MaxPool2D(pool_size=(2, 2), strides=(2,2))(x)

	x = Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding="same")(x)
	x = Activation("relu")(x)

	x = Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding="same")(x)
	x = Activation("relu")(x)

	x = Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding="same")(x)
	x = Activation("relu")(x)

	x = MaxPool2D(pool_size=(2, 2), strides=(2,2))(x)

	x = Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding="same")(x)
	x = Activation("relu")(x)

	x = Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding="same")(x)
	x = Activation("relu")(x)

	x = Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding="same")(x)
	x = Activation("relu")(x)

	x = MaxPool2D(pool_size=(2, 2), strides=(2,2))(x)

	x = Flatten()(x)
	x = Dense(4096)(x)
	x = Activation("relu")(x)
	x = Dropout(.5)(x)
	x = Dense(2048)(x)
	x = Activation("relu")(x)
	x = Dropout(.5)(x)
	x = Dense(1)(x)
	x = Activation("linear")(x)

	# gru1 = GRU(128, return_state=True)(x1)
	# gru2 = GRU(128, return_state=True)(x2)
	# gru3 = GRU(128, return_state=True)(x)
    #
	# combinedInput = concatenate([gru1.output, gru2.output, gru3.output])
	# xfinal = Dense(1, activation="linear")(combinedInput)

	model = Model(inputs, x)
	# return the CNN
	return model

# def create_cnn(height, width, depth):
# 	inputShape = (height, width, depth)
#
# 	inputs = Input(shape=inputShape)
# 	x = inputs
# 	x = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="same")(x)
# 	x = Activation("relu")(x)
#
# 	x = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="same")(x)
# 	x = Activation("relu")(x)
#
# 	x = MaxPool2D(pool_size=(2, 2), strides=(2,2))(x)
#
# 	x = Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding="same")(x)
# 	x = Activation("relu")(x)
#
# 	x = Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding="same")(x)
# 	x = Activation("relu")(x)
#
# 	x = MaxPool2D(pool_size=(2, 2), strides=(2,2))(x)
#
# 	x = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding="same")(x)
# 	x = Activation("relu")(x)
#
# 	x = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding="same")(x)
# 	x = Activation("relu")(x)
#
# 	x = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding="same")(x)
# 	x = Activation("relu")(x)
#
# 	x = MaxPool2D(pool_size=(2, 2), strides=(2,2))(x)
#
# 	x = Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding="same")(x)
# 	x = Activation("relu")(x)
#
# 	x = Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding="same")(x)
# 	x = Activation("relu")(x)
#
# 	x = Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding="same")(x)
# 	x = Activation("relu")(x)
#
# 	x = MaxPool2D(pool_size=(2, 2), strides=(2,2))(x)
#
# 	x1 = Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding="same")(x)
# 	x1 = Activation("relu")(x1)
# 	print("Sheipe:",x1.shape)
#
#     x = Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding="same")(x1)
#     x = Activation("relu")(x)
#
#     x = Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding="same")(x)
#     x = Activation("relu")(x)
#
#     x2 = MaxPool2D(pool_size=(2, 2), strides=(2,2))(x)
#
#     x = Flatten()(x2)
#     x = Dense(4096)(x)
#     x = Activation("linear")(x)
#     x = Dense(2)(x)
#     x = Activation("linear")(x)

	# gru1 = GRU(128, return_state=True)(x1)
	# gru2 = GRU(128, return_state=True)(x2)
	# gru3 = GRU(128, return_state=True)(x)
    #
	# combinedInput = concatenate([gru1.output, gru2.output, gru3.output])
	# xfinal = Dense(1, activation="linear")(combinedInput)

	model = Model(inputs, xfinal)
	# return the CNN
	return model

class ResNetTypeI(tensorflow.keras.Model):
    def __init__(self, layer_params):
        super(ResNetTypeI, self).__init__()

        self.conv1 = tensorflow.keras.layers.Conv2D(filters=64,
                                            kernel_size=(7, 7),
                                            strides=2,
                                            padding="same")
        self.bn1 = tensorflow.keras.layers.BatchNormalization()
        self.pool1 = tensorflow.keras.layers.MaxPool2D(pool_size=(3, 3),
                                               strides=2,
                                               padding="same")

        self.layer1 = make_basic_block_layer(filter_num=64,
                                             blocks=layer_params[0])
        self.layer2 = make_basic_block_layer(filter_num=128,
                                             blocks=layer_params[1],
                                             stride=2)
        self.layer3 = make_basic_block_layer(filter_num=256,
                                             blocks=layer_params[2],
                                             stride=2)
        self.layer4 = make_basic_block_layer(filter_num=512,
                                             blocks=layer_params[3],
                                             stride=2)

        self.avgpool = tensorflow.keras.layers.GlobalAveragePooling2D()
        self.conn1 = tensorflow.keras.layers.Dense(units=4096, activation=tensorflow.keras.activations.relu)
        self.conn2 = tensorflow.keras.layers.Dense(units=2048, activation=tensorflow.keras.activations.relu)
        self.drop = Dropout(.5)
        self.flat = Flatten()
        self.fc = tensorflow.keras.layers.Dense(units=1, activation=tensorflow.keras.activations.linear)

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tensorflow.nn.relu(x)
        x = self.pool1(x)
        x = self.layer1(x, training=training)
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        x = self.layer4(x, training=training)
        x = self.avgpool(x)
        x = self.flat(x)
        x = self.conn1(x)
        x = self.drop(x)
        x = self.conn2(x)
        x = self.drop(x)
        # x = Flatten()(x)
        # x = Dense(4096)(x)
        # x = Activation("relu")(x)
        # x = Dropout(.5)(x)
        # x = Dense(2048)(x)
        # x = Activation("relu")(x)
        # x = Dropout(.5)(x)
        # x = Dense(1)(x)
        output = self.fc(x)

        return output


class ResNetTypeII(tensorflow.keras.Model):
    def __init__(self, layer_params):
        super(ResNetTypeII, self).__init__()
        self.conv1 = tensorflow.keras.layers.Conv2D(filters=64,
                                            kernel_size=(7, 7),
                                            strides=2,
                                            padding="same")
        self.bn1 = tensorflow.keras.layers.BatchNormalization()
        self.pool1 = tensorflow.keras.layers.MaxPool2D(pool_size=(3, 3),
                                               strides=2,
                                               padding="same")

        self.layer1 = make_bottleneck_layer(filter_num=64,
                                            blocks=layer_params[0])
        self.layer2 = make_bottleneck_layer(filter_num=128,
                                            blocks=layer_params[1],
                                            stride=2)
        self.layer3 = make_bottleneck_layer(filter_num=256,
                                            blocks=layer_params[2],
                                            stride=2)
        self.layer4 = make_bottleneck_layer(filter_num=512,
                                            blocks=layer_params[3],
                                            stride=2)

        self.avgpool = tensorflow.keras.layers.GlobalAveragePooling2D()
        self.fc = tensorflow.keras.layers.Dense(units=2, activation=tensorflow.keras.activations.linear)

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tensorflow.nn.relu(x)
        x = self.pool1(x)
        x = self.layer1(x, training=training)
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        x = self.layer4(x, training=training)
        x = self.avgpool(x)
        output = self.fc(x)

        return output


def resnet_18():
    return ResNetTypeI(layer_params=[2, 2, 2, 2])


def resnet_34():
    return ResNetTypeI(layer_params=[3, 4, 6, 3])


def resnet_50():
    return ResNetTypeII(layer_params=[3, 4, 6, 3])


def resnet_101():
    return ResNetTypeII(layer_params=[3, 4, 23, 3])


def resnet_152():
    return ResNetTypeII(layer_params=[3, 8, 36, 3])
