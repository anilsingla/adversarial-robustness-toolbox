import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
from art.estimators.classification import PyTorchClassifier
from art.attacks.poisoning import PoisoningAttackBackdoor, BackdoorAttack
from art.utils import to_categorical

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Simple PyTorch CNN for MNIST
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
    trainset = datasets.MNIST('.', train=True, download=True, transform=transform)
    testset = datasets.MNIST('.', train=False, download=True, transform=transform)
    x_train = np.array([trainset[i][0].numpy() for i in range(n_samples)])
    y_train = np.array([trainset[i][1] for i in range(n_samples)])
    x_test = np.array([testset[i][0].numpy() for i in range(n_samples)])
    y_test = np.array([testset[i][1] for i in range(n_samples)])
    return x_train, y_train, x_test, y_test

def main():
    # 1. Load model
    model = SimpleCNN().to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 2. Prepare data
    x_train, y_train, x_test, y_test = get_mnist_data(200)
    x_train = x_train.astype(np.float32)
    y_train = y_train.astype(np.int64)
    x_test = x_test.astype(np.float32)
    y_test = y_test.astype(np.int64)

    # 3. ART classifier
    classifier = PyTorchClassifier(
        model=model,
        loss=criterion,
        optimizer=optimizer,
        input_shape=(1, 28, 28),
        nb_classes=10,
        clip_values=(0.0, 1.0),
    )

    # 4. Poisoning attack: Backdoor
    # Define a simple backdoor pattern (e.g., a white square in the corner)
    def add_backdoor(x):
        x_bd = np.copy(x)
        x_bd[:, :, 24:28, 24:28] = 1.0
        return x_bd

    # Poison a fraction of the training data
    poison_fraction = 0.2
    n_poison = int(poison_fraction * x_train.shape[0])
    x_poison = add_backdoor(x_train[:n_poison])
    y_poison = np.full(n_poison, 0)  # Target label for backdoor (e.g., class 0)

    # Combine clean and poisoned data
    x_train_poisoned = np.concatenate([x_train, x_poison])
    y_train_poisoned = np.concatenate([y_train, y_poison])

    # 5. Retrain model on poisoned data
    y_train_poisoned_cat = to_categorical(y_train_poisoned, 10)
    for epoch in range(3):
        classifier.fit(x_train_poisoned, y_train_poisoned_cat, batch_size=32, nb_epochs=1)
        print(f"Epoch {epoch+1} completed.")

    # 6. Evaluate on clean test data
    preds_clean = classifier.predict(x_test)
    acc_clean = np.mean(np.argmax(preds_clean, axis=1) == y_test)
    print(f"Accuracy on clean test data: {acc_clean:.4f}")

    # 7. Evaluate on backdoored test data
    x_test_bd = add_backdoor(x_test)
    y_test_bd = np.full_like(y_test, 0)  # Target label for backdoor
    preds_bd = classifier.predict(x_test_bd)
    acc_bd = np.mean(np.argmax(preds_bd, axis=1) == y_test_bd)
    print(f"Attack success rate (backdoor): {acc_bd:.4f}")

if __name__ == "__main__":
    main()