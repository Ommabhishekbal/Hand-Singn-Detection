import os
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

def build_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def train():
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(base_dir, 'data', 'combined_dataset')
        model_dir = os.path.join(base_dir, 'model')
        logs_dir = os.path.join(base_dir, 'logs')

        dataset_path = os.path.join(data_dir, 'augmented_dataset.pkl')
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"❌ Dataset not found at {dataset_path}")

        print(f"\n📥 Loading dataset from: {dataset_path}")
        df = pd.read_pickle(dataset_path)
        print(f"✅ Loaded {len(df)} samples")

        if 'landmarks' not in df.columns or 'label' not in df.columns:
            raise ValueError("❌ DataFrame missing 'landmarks' or 'label' columns")

        # Features and labels
        X = np.stack(df['landmarks'].values)
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(df['label'])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            stratify=y,
            random_state=42
        )

        print(f"🔹 Training samples: {len(X_train)}")
        print(f"🔹 Validation samples: {len(X_test)}")
        print(f"🔹 Classes: {list(label_encoder.classes_)}")

        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)

        # Callbacks
        callbacks = [
            EarlyStopping(patience=5, monitor='val_loss', restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, min_lr=1e-6, verbose=1),
            ModelCheckpoint(
                filepath=os.path.join(model_dir, 'best_model.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            TensorBoard(log_dir=os.path.join(logs_dir, 'fit', datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
        ]

        model = build_model(input_shape=(X.shape[1],), num_classes=len(label_encoder.classes_))
        model.summary()

        print("\n🚀 Starting training...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=40,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )

        # Save final model and labels
        final_model_path = os.path.join(model_dir, 'isl_model.h5')
        model.save(final_model_path)
        np.save(os.path.join(model_dir, 'label_classes.npy'), label_encoder.classes_)
        print(f"\n✅ Final model saved to: {final_model_path}")

        # Evaluation
        print("\n📊 Evaluating on validation set...")
        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        print(f"📈 Final Accuracy: {acc*100:.2f}%")

        y_pred = model.predict(X_test, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        print("\n📋 Classification Report:")
        print(classification_report(y_test, y_pred_classes, target_names=label_encoder.classes_))

    except Exception as e:
        print(f"\n❌ Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    print("🧠 ASL Model Training Script")
    print("=" * 50)
    train()
