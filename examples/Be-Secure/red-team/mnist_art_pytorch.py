import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod

transform = transforms.Compose([transforms.ToTensor()])
trainset = datasets.MNIST('./data', train=True, download=True, transform=transform)
testset = datasets.MNIST('./data', train=False, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False)

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(28*28, 10)
    def forward(self, x):
        x = x.view(-1, 28*28)
        return self.fc(x)

model = SimpleNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

model.train()
for epoch in range(1):
    for images, labels in trainloader:
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

classifier = PyTorchClassifier(
    model=model,
    loss=criterion,
    optimizer=optimizer,
    input_shape=(1, 28, 28),
    nb_classes=10,
    clip_values=(0.0, 1.0)
)

model.eval()
x_test = []
y_test = []
for images, labels in testloader:
    x_test.append(images)
    y_test.append(labels)
x_test = torch.cat(x_test, 0).numpy()
y_test = torch.cat(y_test, 0).numpy()

preds = classifier.predict(x_test)
acc = (preds.argmax(axis=1) == y_test).mean()
print(f"Clean accuracy: {acc:.4f}")

attack = FastGradientMethod(estimator=classifier, eps=0.2)
x_test_adv = attack.generate(x=x_test)
preds_adv = classifier.predict(x_test_adv)
acc_adv = (preds_adv.argmax(axis=1) == y_test).mean()
print(f"Adversarial accuracy: {acc_adv:.4f}")
