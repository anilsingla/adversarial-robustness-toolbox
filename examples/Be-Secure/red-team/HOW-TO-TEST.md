# MNIST Adversarial Robustness Demos (PyTorch & TensorFlow)

This folder provides working examples for testing adversarial robustness of simple MNIST models using the Adversarial Robustness Toolbox (ART).

## Files

- `mnist_art_pytorch.py`: PyTorch example for MNIST with ART.
- `mnist_art_tensorflow.py`: TensorFlow example for MNIST with ART.

## Steps to Run

1. **Install dependencies**:
   ```bash
   pip install torch torchvision tensorflow adversarial-robustness-toolbox

   2. python mnist_art_pytorch.py

   3. python mnist_art_tensorflow.py

   ## Notes
   Both scripts use only one training epoch for demo speed. Increase epochs for better accuracy.
You can adjust the eps parameter in FastGradientMethod for different adversarial strengths.
PyTorch uses integer labels and manual training, TensorFlow uses Keras and one-hot labels.
Input shape for PyTorch is (1, 28, 28), for TensorFlow is (28, 28, 1).
