
import numpy as np
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# Step 1: Load Handwritten Digits Dataset
digits = load_digits()

X, y = digits.data, digits.target

print(
    'Dataset shape:', X.shape,
    '| Classes:', np.unique(y)
)


# Step 2: Split Dataset into Training and Testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)


# Step 3: Standardize Pixel Values
scaler = StandardScaler()

X_train_s = scaler.fit_transform(X_train)

X_test_s = scaler.transform(X_test)


# Step 4: Create SVM Model
svm_model = SVC(
    kernel='rbf',
    C=10,
    gamma='scale'
)


# Step 5: Train the Model
svm_model.fit(X_train_s, y_train)


# Step 6: Make Predictions
y_pred = svm_model.predict(X_test_s)


# Step 7: Calculate Accuracy
acc = accuracy_score(y_test, y_pred)

print(f'\nAccuracy: {acc:.4f}')


# Step 8: Display Classification Report
print(
    '\nClassification Report:\n',
    classification_report(y_test, y_pred)
)


# Step 9: Generate Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=range(10)
)

disp.plot(
    cmap='Blues',
    values_format='d'
)

plt.title('Confusion Matrix - SVM on Handwritten Digits')

plt.tight_layout()

plt.savefig('svm_digits_confusion.png', dpi=120)

plt.close()


# Step 10: Display Sample Digit Predictions
fig, axes = plt.subplots(
    2,
    5,
    figsize=(10, 4)
)

for ax, image, true_label, pred_label in zip(
    axes.ravel(),
    X_test[:10].reshape(-1, 8, 8),
    y_test[:10],
    y_pred[:10]
):

    ax.imshow(image, cmap='gray')

    ax.set_title(
        f'True: {true_label} Pred: {pred_label}',
        fontsize=9
    )

    ax.axis('off')


# Step 11: Save Sample Predictions
plt.tight_layout()

plt.savefig('svm_digits_samples.png', dpi=120)

plt.close()


# Step 12: Project Completion
print('\nSVM Handwritten Digit Recognition Completed!')

print('Generated files:')
print('1. svm_digits_confusion.png')
print('2. svm_digits_samples.png')
