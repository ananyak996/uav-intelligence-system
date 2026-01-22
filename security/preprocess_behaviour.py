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
