import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import balanced_accuracy_score

cancer = load_breast_cancer()
X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
y = cancer.target

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

stand = StandardScaler()
X_train_s = stand.fit_transform(X_train)
X_test_s = stand.transform(X_test)

C = [0.00001,0.01,0.1,1,10,100,100000]
lr_accur = []
svm_accur = []

for i in C:
    lr = LogisticRegression(C=i, penalty='l2', max_iter=10000, random_state=42)
    lr.fit(X_train_s, y_train)
    lr_pred = lr.predict(X_test_s)
    lr_accur.append(balanced_accuracy_score(y_test, lr_pred))

    svm = LinearSVC(C=i, penalty='l2', max_iter=10000, random_state=42)
    svm.fit(X_train_s, y_train)
    svm_pred = svm.predict(X_test_s)
    svm_accur.append(balanced_accuracy_score(y_test, svm_pred))


plt.figure(figsize=(10, 6))
plt.semilogx(C, lr_accur, marker='o')
plt.semilogx(C, svm_accur, marker='s')
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.show()


best_id = np.argmax(lr_accur)
best_C = C[best_id]
best_acc = lr_accur[best_id]
print(f"LR")
print(f"accuracy: {best_acc}")
print(f"C: {best_C}")

best_id = np.argmax(svm_accur)
best_C = C[best_id]
best_acc = svm_accur[best_id]

print("SVM")
print(f"accuracy: {best_acc}")
print(f"C: {best_C}")



