import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from art.estimators.classification import TensorFlowV2Classifier
from art.attacks.evasion import FastGradientMethod

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train.astype(np.float32) / 255.0, x_test.astype(np.float32) / 255.0
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
y_train_cat = tf.keras.utils.to_categorical(y_train, 10)
y_test_cat = tf.keras.utils.to_categorical(y_test, 10)

model = tf.keras.Sequential([
    layers.Flatten(input_shape=(28, 28, 1)),
    layers.Dense(10)
])

loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

model.compile(optimizer=optimizer, loss=loss_object, metrics=['accuracy'])
model.fit(x_train, y_train_cat, batch_size=128, epochs=1, verbose=2)

classifier = TensorFlowV2Classifier(
    model=model,
    loss_object=loss_object,
    nb_classes=10,
    input_shape=(28, 28, 1),
    clip_values=(0.0, 1.0),
    channels_first=False
)

preds = classifier.predict(x_test)
acc = (np.argmax(preds, axis=1) == y_test).mean()
print(f"Clean accuracy: {acc:.4f}")

attack = FastGradientMethod(estimator=classifier, eps=0.2)
x_test_adv = attack.generate(x=x_test)
preds_adv = classifier.predict(x_test_adv)
acc_adv = (np.argmax(preds_adv, axis=1) == y_test).mean()
print(f"Adversarial accuracy: {acc_adv:.4f}")
