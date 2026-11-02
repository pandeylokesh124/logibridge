import os
import warnings

# MUST BE SET BEFORE IMPORTING TENSORFLOW
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

warnings.filterwarnings('ignore')

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot


def representative_dataset():
    data_path = os.path.join(os.path.dirname(__file__), '../training/X_data.npy')
    calibration_data = np.load(data_path)[:200]
    for data in calibration_data:
        yield [data.astype(np.float32).reshape(1, 6)]


def main():
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, '../training/baseline_model.h5')
    x_data_path = os.path.join(base_dir, '../training/X_data.npy')
    y_data_path = os.path.join(base_dir, '../training/y_data.npy')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Cannot find {model_path}. Please run train_model.py first.")

    print("1. Loading Baseline Model (M1)...")
    model = tf.keras.models.load_model(model_path)

    # 2. M2: Full INT8 PTQ
    print("2. Quantizing M2 (Full INT8 PTQ)...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_quant_model = converter.convert()
    m2_path = os.path.join(base_dir, 'model_m2_ptq.tflite')
    with open(m2_path, 'wb') as f:
        f.write(tflite_quant_model)
    print(f"-> M2 saved successfully to '{m2_path}'")

    # 3. M3: 35% Structured Pruning + INT8 PTQ
    print("3. Applying M3 (35% Structured Pruning + INT8 PTQ)...")
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.0,
            final_sparsity=0.35,
            begin_step=0,
            end_step=1000
        )
    }

    pruned_model = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)
    pruned_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    X_train = np.load(x_data_path)
    y_train = np.load(y_data_path)

    callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]
    pruned_model.fit(X_train, y_train, batch_size=16, epochs=2, callbacks=callbacks, verbose=0)

    # Strip pruning wrappers to get clean Keras architecture
    pruned_model_stripped = tfmot.sparsity.keras.strip_pruning(pruned_model)

    # Convert pruned model to INT8 TFLite
    converter_m3 = tf.lite.TFLiteConverter.from_keras_model(pruned_model_stripped)
    converter_m3.optimizations = [tf.lite.Optimize.DEFAULT]
    converter_m3.representative_dataset = representative_dataset
    converter_m3.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter_m3.inference_input_type = tf.int8
    converter_m3.inference_output_type = tf.int8

    tflite_pruned_quant_model = converter_m3.convert()
    m3_path = os.path.join(base_dir, 'model_m3_pruned_ptq.tflite')
    with open(m3_path, 'wb') as f:
        f.write(tflite_pruned_quant_model)
    print(f"-> M3 saved successfully to '{m3_path}'")
    print("\nSUCCESS: All model optimization variants (M2 & M3) generated successfully!")


if __name__ == "__main__":
    main()
