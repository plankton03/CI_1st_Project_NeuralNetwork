from layers.fullyconnected import FC
from layers.convolution2d import Conv2D
from layers.maxpooling2d import MaxPool2D
from activations import ReLU, LinearActivation, Sigmoid
from losses.binarycrossentropy import BinaryCrossEntropy
from losses.meansquarederror import MeanSquaredError
from optimizers.gradientdescent import GD
from optimizers.adam import Adam
from model import Model
from PIL import Image
import numpy as np
import os



#loading images
def load_images(directory):
    images = []
    for file in os.listdir(directory):
        path = os.path.join(directory, file)
        image = Image.open(path)
        images.append(image)
    return images

def test_accuracy(prediction):
    y_pred = []
    correct_predictions =0
    for i in range(len(y_test)):
        if prediction[0][i] > 0.55 and y_test[i] > 0.55:
            correct_predictions +=1;
        if prediction[0][i] < 0.45 and y_test[i] < 0.45:
            correct_predictions += 1;
    print(correct_predictions / len(y_test) * 100, 'percent accuracy')

images_2 = load_images('datasets/MNIST/2')
images_5 = load_images('datasets/MNIST/5')



imageset = []
labels = []
data_size = 50
for i in range(data_size):
    imageset.append(np.array(images_2[i], dtype=np.uint8))
    labels.append(1)
    imageset.append(np.array(images_5[i], dtype=np.uint8))
    labels.append(0)
imageset = np.array(imageset)
labels = np.array(labels)


indices = np.arange(len(imageset))
np.random.seed(30)
np.random.shuffle(indices)

x_train = imageset[indices[int(0.2*len(imageset)):]]
x_test = imageset[indices[:int(0.2*len(imageset))]]
y_train = labels[indices[int(0.2*len(imageset)):]]
y_test = labels[indices[:int(0.2*len(imageset))]]


x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

x_train = np.divide(x_train, 255)
x_test = np.divide(x_test, 255)
# layer1 = Conv2D(1, 5, "l1")
# activation1 = LinearActivation()
# layer2 = MaxPool2D()
# activation2 = LinearActivation()
# layer3 = Conv2D(5, 5, "l3")
# activation3 = LinearActivation()
# layer4 = MaxPool2D()
# activation4 = LinearActivation()
# layer5= FC(3920, 1, "l5")
# activation5 = Sigmoid()
# epochs = 10
# arch = {"l1" : layer1, "a1" : activation1, "l2" : layer2, "a2" : activation2, "l3" : layer3, "a3" : activation3,  "l4" : layer4, "a4" : activation4,  "l5" : layer5, "a5" : activation5}
# layers_list = {"l1" : layer1, "l3" : layer3, "l5" : layer5}
# loos_function = MeanSquaredError()
# optimizer = Adam(layers_list)
# model = Model(arch, loos_function, optimizer)
# #
models_arch = {
    "CONV1": Conv2D(1, 2, name="CONV1", kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), initialize_method="random"),
    "RELU1": ReLU(),
    "MAXPOOL1": MaxPool2D(kernel_size=(2, 2), stride=(2, 2)),
    "CONV2": Conv2D(2, 4, name="CONV2", kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
    "RELU2": ReLU(),
    "MAXPOOL2": MaxPool2D(kernel_size=(2, 2), stride=(2, 2)),
    "FC1": FC(196, 32, "FC1"),
    "RELU3": ReLU(),
    "FC2": FC(32, 1, "FC2"),
    "SIGMOMID2": Sigmoid(),
}

model = Model(models_arch, criterion=MeanSquaredError(), optimizer=Adam(models_arch))

model.train(x_train, y_train.T, epochs=10)

prediction = model.predict(x_test)

test_accuracy(prediction)
