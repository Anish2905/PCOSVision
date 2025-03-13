import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from PIL import Image  # Added for better image handling

# Set random seed for reproducibility
np.random.seed(42)

def load_data(data_dir, img_size=(128, 128)):
    """
    Load images and their corresponding labels from the dataset directory.
    Resizes and normalizes images for training.
    """
    images = []
    labels = []
    class_names = sorted(os.listdir(data_dir))  # Ensure consistent class ordering

    for label, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            if not img_name.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            
            img = cv2.imread(img_path)
            if img is None:
                continue  # Skip invalid images

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            img = cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)  # Resize with better quality
            img = img / 255.0  # Normalize pixel values (0 to 1)

            images.append(img)
            labels.append(label)

    images = np.array(images, dtype="float32")
    labels = to_categorical(labels, num_classes=len(class_names))

    return images, labels, class_names

def plot_additional_metrics(y_true, y_pred_probs, class_names):
    """
    Plot additional evaluation metrics such as ROC curve and Precision-Recall curve.
    """
    y_true = np.argmax(y_true, axis=1)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_pred_probs[:, 1])
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker='.', label='Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.show()
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_probs[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()

def build_model(input_shape, num_classes):
    """
    Build a CNN model.
    """
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2,2)),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D((2,2)),
        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def plot_history(history):
    """Plot training & validation accuracy and loss."""
    plt.figure(figsize=(12,5))
    
    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    
    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def evaluate_model(model, X_test, y_test, class_names):
    """
    Evaluate the model on test data and print classification metrics.
    """
    preds = model.predict(X_test)
    y_pred = np.argmax(preds, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Confusion Matrix with Heatmap
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.show()
    
    # Plot additional metrics
    plot_additional_metrics(y_test, preds, class_names)

def main():
    """
    Main function to handle data loading, model training, evaluation, and saving.
    """
    data_dir = "dataset_extracted/PCOS"  # Path to dataset
    img_size = (128, 128)
    epochs = 25
    batch_size = 32
    
    # Load the dataset
    if not os.path.exists(data_dir):
        print(f"Error: Dataset directory '{data_dir}' does not exist.")
        return
    
    X, y, class_names = load_data(data_dir, img_size)
    
    if X is None or len(X) == 0:
        print("Exiting: No images found in the dataset.")
        return
    
    print("Dataset loaded. Total samples:", len(X))
    
    # Split into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Build the model
    input_shape = (img_size[0], img_size[1], 3)
    num_classes = len(class_names)
    model = build_model(input_shape, num_classes)
    model.summary()
    
    # Set callbacks
    checkpoint = ModelCheckpoint("best_model.h5", monitor='val_accuracy', save_best_only=True, verbose=1)
    early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
    
    # Train the model
    history = model.fit(X_train, y_train, validation_split=0.2, epochs=epochs, batch_size=batch_size, 
                        callbacks=[checkpoint, early_stop])
    
    # Plot training history
    plot_history(history)
    
    # Evaluate the model
    evaluate_model(model, X_test, y_test, class_names)
    
    # Save the final model
    model.save("final_model.h5")
    print("Model saved as final_model.h5")

if __name__ == "__main__":
    main()
