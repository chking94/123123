import tensorflow as tf
from tensorflow.keras.layers import LeakyReLU, ReLU, Conv2D, MaxPooling2D, BatchNormalization, Conv2DTranspose, UpSampling2D, concatenate
from tensorflow.keras.layers import Dense, Input, Activation, Flatten
from tensorflow.keras.layers import BatchNormalization,Add,Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import to_categorical

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

def convnet(input_size = (28,28,1)):
    inp = Input(input_size)
    conv1 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inp)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    flatten = Flatten()(pool1)
    dense1 = Dense(100, activation = 'relu')(flatten)
    dense2 = Dense(10, activation = 'softmax')(dense1)
    model = Model(inputs = inp, outputs = dense2)
    return model

model = convnet()
model.summary()
model.compile(optimizer = Adam(lr = 1e-3), loss = 'categorical_crossentropy', metrics = ['accuracy'])
history = model.fit(x_train, y_train, epochs = 10)
evalu = model.evaluate(x_test, y_test)
