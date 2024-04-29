from prepare_data import *
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from nets.MLP import mlp
from nets.conv import conv
from random import randint
from PIL import Image
from keras.utils import to_categorical

# define some constants
N_ANIMALS = 10
ANIMALS = {
    0: "cat",
    1: "elephant",
    2: "bear",
    3: "bird",
    4: "crab",
    5: "fish",
    6: "giraffe",
    7: "lion",
    8: "rabbit",
    9: "snake",
}

# number of samples to take in each class
N = 5000

# some other constants
N_EPOCHS = 30

# data files in the same order as defined in ANIMALS
files = [
    "cat.npy",
    "elephant.npy",
    "bear.npy",
    "bird.npy",
    "crab.npy",
    "fish.npy",
    "giraffe.npy",
    "lion.npy",
    "rabbit.npy",
    "snake.npy",
]

# images need to be 28x28 for training with a ConvNet
animals = load("data/", files, reshaped=True)

# images need to be flattened for training with an MLP
# animals = load("data/", files, reshaped=False)


# limit no of samples in each class to N
animals = set_limit(animals, N)

# normalize the values
animals = np.array(list(map(normalize, animals)))

# define the labels
labels = make_labels(N_ANIMALS, N)


# prepare the data

x_train, x_test, y_train, y_test = train_test_split(animals, labels, test_size=0.05)

# one hot encoding
Y_train = to_categorical(y_train, N_ANIMALS)
Y_test = to_categorical(y_test, N_ANIMALS)

# use our custom designed ConvNet model
model = conv(classes=N_ANIMALS, input_shape=(28, 28, 1))

# use our custom designed MLP model
# model = mlp(classes=N_ANIMALS)


model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])


print("Training commenced")

model.fit(
    np.array(x_train), np.array(Y_train), batch_size=32, epochs=N_EPOCHS, verbose=1
)

print("Training complete")

print("Evaluating model")
preds = model.predict(np.array(x_test))

score = 0
for i in range(len(preds)):
    if np.argmax(preds[i]) == y_test[i]:
        score += 1

print("Accuracy: ", ((score + 0.0) / len(preds)) * 100)

model_name = input(">Enter name to save trained model: ")
model.save(model_name + ".keras")
print("Model saved")


def visualize_and_predict():
    "selects a random test case and shows the object, the prediction and the expected result"
    n = randint(0, len(x_test))

    visualize(denormalize(np.reshape(x_test[n], (28, 28))))
    pred = ANIMALS[np.argmax(model.predict(np.array([x_test[n]])))]
    actual = ANIMALS[y_test[n]]
    print("Actual:", actual)
    print("Predicted:", pred)


print("Testing model")
visualize_and_predict()
