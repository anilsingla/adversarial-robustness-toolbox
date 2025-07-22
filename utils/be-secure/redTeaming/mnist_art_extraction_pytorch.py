import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
from art.estimators.classification import PyTorchClassifier
from art.attacks.extraction import KnockoffNets
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

def get_mnist_data(n_samples=100):
    transform = transforms.Compose([transforms.ToTensor()])
    testset = datasets.MNIST('.', train=False, download=True, transform=transform)
    x = np.array([testset[i][0].numpy() for i in range(n_samples)])
    y = np.array([testset[i][1] for i in range(n_samples)])
    return x, y

def main():
    # Victim model
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    x_test, y_test = get_mnist_data(100)
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

    # Substitute model (same architecture for simplicity)
    substitute = SimpleCNN().to(device)
    substitute_classifier = PyTorchClassifier(
        model=substitute,
        loss=criterion,
        optimizer=optim.Adam(substitute.parameters(), lr=0.001),
        input_shape=(1, 28, 28),
        nb_classes=10,
        clip_values=(0.0, 1.0),
    )

    # KnockoffNets extraction attack
    attack = KnockoffNets(
        classifier=classifier,
        batch_size_fit=16,
        batch_size_query=16,
        nb_epochs=1,
        nb_stolen=50,
        substitute_classifier=substitute_classifier,
        use_probability=True,
    )

    attack.extract(x_test, y=None)
    preds = substitute_classifier.predict(x_test)
    acc = np.mean(np.argmax(preds, axis=1) == y_test)
    print(f"Accuracy of stolen model on test data: {acc:.4f}")

if __name__ == "__main__":
    main()