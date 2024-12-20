# Import modules
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Preprocessing Data
# Load images and split into training and testing sets
def prepare_data(data_dir, img_size=(224, 224), test_split=0.2):
    data, labels = [], []
    class_names = os.listdir(data_dir)

    for idx, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        if os.path.isdir(class_dir):
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, img_size)
                    data.append(img.flatten())
                    labels.append(idx)

    data = np.array(data) / 255.0  # Normalize pixel values to [0, 1]
    labels = np.array(labels)

    # Return data, labels, and class names
    return data, labels, class_names

# Build Classification Algorithm
def Knn(X_train, y_train, X_test):
    # Classification with KNN
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    return knn.predict(X_test)

def Svm(X_train, y_train, X_test):
    # Classification with SVM
    svm = SVC(kernel='linear')
    svm.fit(X_train, y_train)
    return svm.predict(X_test)

def Random_forest(X_train, y_train, X_test):
    # Classification with Random Forest
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X_train, y_train)
    return rf.predict(X_test)

# Function to display confusion matrix
def display_confusion_matrix(y_test, y_pred, title):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(title)
    plt.show()

if __name__ == '__main__':
    # Load data
    X, y, class_names = prepare_data('d:/PROJECT MK KB/Tika/Dataset')  # Adjust path as needed

    # Split dataset into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Classification with KNN
    y_pred_knn = Knn(X_train, y_train, X_test)
    print("KNN Classification Report:")
    print(classification_report(y_test, y_pred_knn))
    display_confusion_matrix(y_test, y_pred_knn, "KNN Confusion Matrix")

    # Classification with SVM
    y_pred_svm = Svm(X_train, y_train, X_test)
    print("SVM Classification Report:")
    print(classification_report(y_test, y_pred_svm))
    display_confusion_matrix(y_test, y_pred_svm, "SVM Confusion Matrix")

    # Classification with Random Forest
    y_pred_rf = Random_forest(X_train, y_train, X_test)
    print("Random Forest Classification Report:")
    print(classification_report(y_test, y_pred_rf))
    display_confusion_matrix(y_test, y_pred_rf, "Random Forest Confusion Matrix")