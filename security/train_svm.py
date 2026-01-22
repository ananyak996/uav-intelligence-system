import pandas as pd

df=pd.read_csv('uuav_behavior_dataset.csv')
#Feature-Label separation
from sklearn.model_selection import train_test_split

X = df[['energy_consumption',
        'mobility_pattern',
        'packet_transmission',
        'link_stability']]

y = df['behavior_label']

#Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y      #ensures that the proportion of different classes in the original dataset is preserved
)

#Feature Scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)  #learns mean & variance from training data
X_test_scaled = scaler.transform(X_test)    #applies the same transformation to test data





#SVM Model
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

svm = SVC(kernel='rbf', random_state=42)

svm.fit(X_train_scaled, y_train)

y_pred_svm = svm.predict(X_test_scaled)

accuracy_svm = accuracy_score(y_test, y_pred_svm)
cm_svm = confusion_matrix(y_test, y_pred_svm)

print("SVM Accuracy:", accuracy_svm)
print("SVM Confusion Matrix:\n", cm_svm)
