import os

# MUST BE SET BEFORE IMPORTING TENSORFLOW
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


def main():
    base_dir = os.path.dirname(__file__)
    x_path = os.path.join(base_dir, 'X_data.npy')
    y_path = os.path.join(base_dir, 'y_data.npy')

    if not os.path.exists(x_path) or not os.path.exists(y_path):
        raise FileNotFoundError("Dataset files not found. Run generate_dataset.py first.")

    X = np.load(x_path)
    y = np.load(y_path)

    # 80/20 Stratified Train-Validation Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 2-hidden-layer MLP (32 & 16 units, ReLU, 3-class Softmax)
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(6,)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print("Training Keras MLP Model (Legacy Keras mode)...")
    model.fit(
        X_train, y_train,
        epochs=40,
        batch_size=16,
        validation_data=(X_val, y_val),
        verbose=1
    )

    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\nFinal Validation Accuracy: {val_acc * 100:.2f}%")

    if val_acc < 0.88:
        print("WARNING: Validation accuracy is below the mandatory 88% threshold.")
    else:
        print("SUCCESS: Validation accuracy meets the LogiEdge requirement (>88%).")

    y_pred = np.argmax(model.predict(X_val), axis=1)
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_val, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=['Normal (0)', 'Warning (1)', 'Critical (2)']))

    # Save as legacy .h5 model file
    model_path = os.path.join(base_dir, 'baseline_model.h5')
    model.save(model_path)
    print(f"\nBaseline model saved successfully to '{model_path}'")


if __name__ == "__main__":
    main()
