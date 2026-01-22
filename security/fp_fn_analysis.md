# UAV Security Module – Behavioral Classification Analysis


## Exploratory Data Analysis (EDA)
Initial visual analysis revealed partial overlap between cooperative and malicious UAV behavior.  
Malicious UAVs showed higher variance in mobility and energy consumption




## Model Comparison
Three classifiers were evaluated:

### Logistic Regression
- Accuracy: 78%
- Observed high false-negative rate


### K-Nearest Neighbors (k = 5)
- Accuracy: 90.5%
- Significant reduction in false negatives

### Support Vector Machine (RBF Kernel)
- Accuracy: 95%
- False Positives: 6
- False Negatives: 4

SVM demonstrated the best balance between detection accuracy and security reliability.



## Security Risk Analysis
In UAV security, false negatives pose a greater risk than false positives, because missing a malicious UAV can lead to severe safety and mission failures.

## Final Model Selection
Based on accuracy, robustness, and security considerations, the Support Vector Machine (RBF kernel) was selected as the final classifier for the UAV Security Module.
