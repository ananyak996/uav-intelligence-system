#EDA on UAV Behavior Dataset
import pandas as pd

df=pd.read_csv('uuav_behavior_dataset.csv')
print(df.head())
print(df.info())
print(df.describe())
print(df['behavior_label'].value_counts())      #count of each class
print(df['behavior_label'].value_counts(normalize=True))   #percentage of each class


#Visualizing the distribution of key features
import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
plt.scatter(
    df[df['behavior_label'] == 0]['energy_consumption'],
    df[df['behavior_label'] == 0]['mobility_pattern'],
    label='Cooperative',
    alpha=0.6,     #opacity
)
plt.scatter(
    df[df['behavior_label'] == 1]['energy_consumption'],
    df[df['behavior_label'] == 1]['mobility_pattern'],
    label='Malicious',
    alpha=0.6   
)
plt.xlabel('Energy Consumption')
plt.ylabel('Mobility Pattern')
plt.legend()
plt.title('Mobility vs Energy by UAV Behavior')
plt.show()



plt.figure(figsize=(6,4))
plt.scatter(
    df[df['behavior_label'] == 0]['packet_transmission'],
    df[df['behavior_label'] == 0]['link_stability'],
    label='Cooperative',
    alpha=0.6
)
plt.scatter(
    df[df['behavior_label'] == 1]['packet_transmission'],
    df[df['behavior_label'] == 1]['link_stability'],
    label='Malicious',
    alpha=0.6
)

plt.xlabel('Packet Transmission (%)')
plt.ylabel('Link Stability')
plt.legend()
plt.title('Communication Features by UAV Behavior')
plt.show()


