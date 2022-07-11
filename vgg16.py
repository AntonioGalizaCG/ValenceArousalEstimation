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

	#x = Flatten()(x2)
	x = Dense(4096)(x)
	x = Activation("relu")(x)
	x = Dropout(.5)(x)
	x = Dense(2048)(x)
	x = Activation("relu")(x)
	x = Dropout(.5)(x)
	x = Dense(2)(x)
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
