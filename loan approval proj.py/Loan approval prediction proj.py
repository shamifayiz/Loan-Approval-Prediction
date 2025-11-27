#Loan Approval Prediction Using SVM

#Predict whether a loan application will be approved (1) or rejected (0) based on applicant features.
#1. Features in a Typical Loan Dataset
#Feature	        Description
#ApplicantIncome	Monthly income of the applicant
#CoapplicantIncome	Income of co-applicant
#LoanAmount	        Amount of loan requested
#Loan_Amount_Term	Term of loan (in months)
#Credit_History	    1 = good, 0 = bad
#Gender, Married, Education, Self_Employed->Categorical variables
#Property_Area	    Urban/Rural/Semiurban
#Loan_Status	    Target (Y = Approved, N = Rejected)


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler,OneHotEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


# ✅ Step 1: Load data
data = pd.read_csv(r"Loan Prediction Dataset - RAW.csv")
print(data.head())

# ✅ Step 2: Drop unnecessary columns
data.drop("Loan_ID", axis=1, inplace=True)
print(data)
print(data.isnull().sum())

print(data)

print(data.describe())
print(data.info())


# ✅ Step 3: Rename all column headers
data.columns = data.columns.str.strip().str.lower().str.replace(" ", "_")
print(data.columns)

# ✅ Step 4: Handle missing values
cat_cols = data.select_dtypes(include="object").columns
num_cols = data.select_dtypes(include=np.number).columns

for col in cat_cols:
    data[col].fillna(data[col].mode()[0], inplace=True)

for col in num_cols:
    data[col].fillna(data[col].median(), inplace=True)

print(data.isnull().sum())

print(data)

#Exploratory Data Analystics(Different kind of realtion from dataset helps our further process)
print(data["loan_status"].value_counts())
print(data.columns)

#Visulazation

import matplotlib.pyplot as plt

# Convert Y/N to 1/0
data["loan_status_num"] = data["loan_status"].map({'Y': 1, 'N': 0})

plt.figure(figsize=(10, 5))
scatter = plt.scatter(
    data["credit_history"],
    data["applicantincome"],
    c=data["loan_status_num"],
    cmap="coolwarm",
    s=60,
    edgecolor='k'
)
plt.colorbar(scatter, label="Loan Status (0 = Not Approved, 1 = Approved)")
plt.xlabel("Credit History")
plt.ylabel("Applicant Income")
plt.title("Loan Approval Prediction (SVM Classifier)")
plt.show()

#Categorical features:barplot(Gender)
sns.countplot(x="gender",data=data)
plt.show()

# Categorical feature: Married status
sns.countplot(x='married', data=data)
plt.show()

#Numerical features:(histogram)
sns.histplot(data["applicantincome"],bins=5,kde=True)
plt.show()

sns.histplot(data["coapplicantincome"],bins=5,kde=True)
plt.show()

#Boxplot(shows distribution across categories and categorical and numeric)
sns.boxplot(x='loan_status', y='applicantincome', data=data)
plt.title('loan status vs applicant income')
plt.show()

#Encoding variable
le = LabelEncoder()
for col in cat_cols:
    data[col] = le.fit_transform(data[col])
    
#Correlation Heatmap
import seaborn as sns
import matplotlib.pyplot as plt

numeric_data = data.select_dtypes(include=["float64", "int64"])
corr = numeric_data.corr()['loan_status'].sort_values(ascending=False)
print(corr)

plt.figure(figsize=(10,5))
sns.heatmap(numeric_data.corr(),annot=True,cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

print(data)

# Features (X) and target (y)
X = data.drop(['loan_status'], axis=1)
y = data['loan_status']

# Split train/test data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

#Features scaling
from sklearn.preprocessing import StandardScaler
import numpy as np
scalar=StandardScaler()
X_train_scaled=scalar.fit_transform(X_train) # first computes the mean and standard deviation from the training set (fit), then applies scaling (transform).
X_test_scaled=scalar.transform(X_test)  #This ensures there’s no data leakage from the test set into the training process.
X_train_scaled
print("Mean after scaling:", np.mean(X_train_scaled, axis=0))
print("Std after scaling:", np.std(X_train_scaled, axis=0))

#Evalutition metrics
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# ----------------------------
# Logistic Regression
# ----------------------------
log_model = LogisticRegression(max_iter=700, random_state=42)
log_model.fit(X_train_scaled, y_train)
log_pred = log_model.predict(X_test_scaled)   # <-- fixed .predict()

# ----------------------------
# Support Vector Machine (SVM)
# ----------------------------
svm_model = SVC(kernel="rbf", probability=True, random_state=42)  # <-- fixed parameter names
svm_model.fit(X_train_scaled, y_train)
svm_pred = svm_model.predict(X_test_scaled)   # <-- fixed .predict()

# ----------------------------
# Random Forest Classifier
# ----------------------------
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train_scaled, y_train)
rf_pred = rf_model.predict(X_test_scaled)

# ----------------------------
# Evaluation of all models
# ----------------------------
models = {
    "Logistic Regression": (y_test, log_pred),
    "Support Vector Machine": (y_test, svm_pred),
    "Random Forest": (y_test, rf_pred)
}

for model_name, (y_true, y_pred) in models.items():
    acc = accuracy_score(y_true, y_pred)
    print(f"{model_name} Accuracy: {acc:.4f}")
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("Classification Report:\n", classification_report(y_true, y_pred))
    print("-" * 60) #is just a simple formatting trick used to make console output easier to read."–" * 60 creates a string with 60 dashes:
    


# Create dictionary of accuracies
accuracy_results = {
    "Logistic Regression": accuracy_score(y_test, log_pred),
    "Support Vector Machine": accuracy_score(y_test, svm_pred),
    "Random Forest": accuracy_score(y_test, rf_pred)
}

# Best model selection
best_model = max(accuracy_results, key=accuracy_results.get)
print("Best Model:", best_model)
print("Best Model Accuracy:", accuracy_results[best_model])
