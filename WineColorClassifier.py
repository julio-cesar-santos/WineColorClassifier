import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

#Loading the dataset
file_path = "../machine learning/wine_dataset.csv"
data = pd.read_csv(file_path)

label_encoder = LabelEncoder() 
#Converting using the LabelEncoder the "style" column to numerical values
data["style"] = label_encoder.fit_transform(data["style"])

X = data.drop(columns="style", axis=1)
y = data["style"]

#Splitting the dataset into training 80% and testing 20% subsets
#'Stratify =y' to both class distribution sets to be same
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#Using the entropy criterion because it yielded the best results during testing
#Using random_state = 42 ensures that the test produces the same result every time
model = ExtraTreesClassifier(criterion="entropy", random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

#Results
accuracy = accuracy_score(y_test, y_pred)
print(f"The test has {accuracy * 100:.2f}% accuracy")
