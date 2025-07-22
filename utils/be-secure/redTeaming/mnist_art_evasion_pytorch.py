import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from art.estimators.classification import PyTorchClassifier

# Import all potential attacks from ART
from art.attacks.evasion import (
    FastGradientMethod, BasicIterativeMethod, ProjectedGradientDescent,
    DeepFool, CarliniL2Method, ElasticNet, SaliencyMapMethod, UniversalPerturbation,
    HopSkipJump, AutoAttack, SquareAttack, VirtualAdversarialMethod, BoundaryAttack
)

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
    testset = datasets.MNIST('.', train=False, download=True, transform=transform)
    x = np.array([testset[i][0].numpy() for i in range(n_samples)])
    y = np.array([testset[i][1] for i in range(n_samples)])
    return x, y

def main():
    # 1. Load a pretrained or freshly trained model for simplicity
    model = SimpleCNN().to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 2. Prepare data
    x_test, y_test = get_mnist_data(100)
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
    
    # 4. List of all attacks you want to try (add/remove as needed)
    attacks = [
        ("FGSM", FastGradientMethod(classifier, eps=0.2)),
        ("BIM", BasicIterativeMethod(classifier, eps=0.2, max_iter=10)),
        ("PGD", ProjectedGradientDescent(classifier, eps=0.2, max_iter=10)),
        ("DeepFool", DeepFool(classifier, max_iter=10)),
        ("C&W L2", CarliniL2Method(classifier, max_iter=10, confidence=0.0)),
        ("ElasticNet", ElasticNet(classifier, max_iter=10, confidence=0.0, beta=0.01)),
        ("JSMA", SaliencyMapMethod(classifier, theta=0.1, gamma=0.1)),
        ("UniversalPerturbation", UniversalPerturbation(classifier, attacker=FastGradientMethod(classifier, eps=0.2))),
        ("HopSkipJump", HopSkipJump(classifier, max_iter=10)),
        ("AutoAttack", AutoAttack(classifier, eps=0.2)),
        ("SquareAttack", SquareAttack(classifier, eps=0.2, max_iter=100)),
        ("VAT", VirtualAdversarialMethod(classifier, eps=0.2, max_iter=10)),
        ("BoundaryAttack", BoundaryAttack(classifier, max_iter=10)),
    ]
    
    # 5. Evaluate on clean data
    preds = classifier.predict(x_test)
    acc = np.mean(np.argmax(preds, axis=1) == y_test)
    print(f"Clean accuracy: {acc:.4f}\n")

    # 6. Run each attack
    for name, attack in attacks:
        try:
            print(f"Running {name}...")
            # Some attacks may not require y (e.g., DeepFool), but most do
            if name in ("JSMA",):  # Targeted attacks require target labels; skipping for simplicity
                print(f"Skipping {name} (requires target labels by default).")
                continue
            x_adv = attack.generate(x=x_test, y=y_test)
            preds_adv = classifier.predict(x_adv)
            acc_adv = np.mean(np.argmax(preds_adv, axis=1) == y_test)
            print(f"{name} adversarial accuracy: {acc_adv:.4f}")
        except Exception as ex:
            print(f"{name} failed: {ex}")
        print("-" * 40)

if __name__ == "__main__":
    main()
