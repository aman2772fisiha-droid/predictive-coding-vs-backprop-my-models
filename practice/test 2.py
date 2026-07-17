from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape)  # (60000, 28, 28)
print(y_train.shape)  # (60000,)

from torchvision import datasets

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True
)