import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import euclidean_distances


data = [6, 7, 8, 9, 10, 11, 12]
mean_val = np.mean(data)
print("Mean ", mean_val)


variance_val = np.var(data)
std_dev_val = np.std(data)
print("Variance ", variance_val)
print("Standard Deviation ", std_dev_val)


dataset = [[25, 20000],
           [30, 40000],
           [35, 80000]]
scaler = StandardScaler()
scaled_data = scaler.fit_transform(dataset)
print("Scaled Dataset ", scaled_data)


point1 = [[25, 20000]]
point2 = [[35, 80000]]
dist_before = euclidean_distances(point1, point2)[0][0]
scaled_points = scaler.transform([point1[0], point2[0]])
dist_after = euclidean_distances([scaled_points[0]], [scaled_points[1]])[0][0]
print("Distance before scaling ", dist_before)
print("Distance after scaling ", dist_after)


actual = [1,1,1,1,0,0,0,0]
predicted = [1,1,0,1,0,1,0,0]

TP = sum((a==1 and p==1) for a,p in zip(actual, predicted))
TN = sum((a==0 and p==0) for a,p in zip(actual, predicted))
FP = sum((a==0 and p==1) for a,p in zip(actual, predicted))
FN = sum((a==1 and p==0) for a,p in zip(actual, predicted))

print("TP:", TP, "TN:", TN, "FP:", FP, "FN:", FN)

report = classification_report(actual, predicted)
print("Classification Report ", report)