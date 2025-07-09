Absolutely! Here’s a single, copy-paste ready markdown block for your `HOW-TO-TEST.md` file, with all steps, prerequisites, installation, run commands, and notes clearly organized and formatted.

---

```markdown
# HOW-TO-TEST.md

## MNIST Adversarial Robustness Demos (PyTorch & TensorFlow)

This guide provides steps to test adversarial robustness of simple MNIST models using the Adversarial Robustness Toolbox (ART).

---

### Pre-requisites

- **Python 3.7 or higher** is required.
- Ensure `pip` is up to date:
  ```bash
  python -m pip install --upgrade pip
  ```
- It is recommended to use a [virtual environment](https://docs.python.org/3/tutorial/venv.html) to avoid dependency conflicts.

---

### ART Installation

Install ART, PyTorch, TensorFlow, and torchvision by running:

```bash
pip install adversarial-robustness-toolbox torch torchvision tensorflow
```

> **GPU Users:**  
> If you have a GPU and want to use it with TensorFlow or PyTorch, follow the official [PyTorch](https://pytorch.org/get-started/locally/) and [TensorFlow](https://www.tensorflow.org/install) instructions for installing the GPU-enabled versions.

---

### How to Run the Examples

#### Run the PyTorch example

This will train and evaluate a simple PyTorch model, then test it with adversarial examples:

```bash
python mnist_art_pytorch.py
```

#### Run the TensorFlow example

Similarly, this will train and evaluate a TensorFlow model and test it with adversarial examples:

```bash
python mnist_art_tensorflow.py
```

---

### Notes

- Both scripts use only one training epoch for demonstration. **Increase epochs in the script for better model accuracy.**
- The `eps` parameter in `FastGradientMethod` controls attack strength—raise it for stronger attacks.
- PyTorch expects input shape `(1, 28, 28)` (channels first). TensorFlow expects `(28, 28, 1)` (channels last).
- PyTorch uses integer labels and a manual training loop. TensorFlow uses Keras and one-hot encoded labels.

---

After following these steps, both scripts will print clean and adversarial accuracy, allowing you to observe the effect of adversarial attacks on simple MNIST models.
```

---

You can copy and paste this entire block directly into your `HOW-TO-TEST.md` file.  
If you need further customization or have additional requirements, let me know!
