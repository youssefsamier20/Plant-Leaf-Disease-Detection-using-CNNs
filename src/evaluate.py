import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

sys.path.append("src")

from dataset import test_generator, class_labels

model_path = 'best_model.h5'
print(f"⏳ Loading model from {model_path}...")

try:
    model = tf.keras.models.load_model(model_path)
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    sys.exit()

print("\n📊 Evaluating on Test Set...")
test_loss, test_acc = model.evaluate(test_generator)
print(f"🔹 Final Test Loss: {test_loss:.4f}")
print(f"🔹 Final Test Accuracy: {test_acc:.4f}")

print("\n🔮 Generating Predictions...")
test_generator.reset()

predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_generator.classes
class_labels_list = list(test_generator.class_indices.keys())

print("\n📝 Detailed Classification Report:")
print(classification_report(true_classes, predicted_classes, target_names=class_labels_list))

print("\n🟦 Drawing Confusion Matrix...")
cm = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels_list, yticklabels=class_labels_list)
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.show()