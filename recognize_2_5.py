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
data_size = 100
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

models_arch = {
    "CONV1": Conv2D(1, 4, name="CONV1"),
    "SIGMOMID1": Sigmoid(),
    "MAXPOOL1": MaxPool2D(),
    "SIGMOMID2": Sigmoid(),
    "CONV2": Conv2D(4, 4, name="CONV2"),
    "SIGMOMID3": Sigmoid(),
    "FC1": FC(3600, 1, "FC1"),
    "RELU2": ReLU(),
}

model = Model(models_arch, criterion=MeanSquaredError(), optimizer=Adam(models_arch))

model.train(x_train, y_train.T, epochs=10)

prediction = model.predict(x_test)

test_accuracy(prediction)
