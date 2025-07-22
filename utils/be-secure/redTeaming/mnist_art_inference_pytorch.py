import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
from art.estimators.classification import PyTorchClassifier
from art.attacks.inference.membership_inference import MembershipInferenceBlackBox
from art.utils import to_categorical

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(9216, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

def get_mnist_data(n_samples=100, train=True):
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST('.', train=train, download=True, transform=transform)
    x = np.array([dataset[i][0].numpy() for i in range(n_samples)])
    y = np.array([dataset[i][1] for i in range(n_samples)])
    return x, y

def main():
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    x_train, y_train = get_mnist_data(100, train=True)
    x_test, y_test = get_mnist_data(100, train=False)
    x_train = x_train.astype(np.float32)
    y_train = y_train.astype(np.int64)
    x_test = x_test.astype(np.float32)
    y_test = y_test.astype(np.int64)

    classifier = PyTorchClassifier(
        model=model,
        loss=criterion,
        optimizer=optimizer,
        input_shape=(1, 28, 28),
        nb_classes=10,
        clip_values=(0.0, 1.0),
    )

    # Train the model briefly for demonstration
    y_train_cat = to_categorical(y_train, 10)
    classifier.fit(x_train, y_train_cat, batch_size=32, nb_epochs=1)

    # Membership inference attack
    attack = MembershipInferenceBlackBox(classifier)
    attack.fit(x_train, y_train)
    preds_train = attack.infer(x_train, y_train)
    preds_test = attack.infer(x_test, y_test)
    print(f"Train set membership inference accuracy: {np.mean(preds_train):.4f}")
    print(f"Test set membership inference accuracy: {np.mean(preds_test):.4f}")

if __name__ == "__main__":
    main()