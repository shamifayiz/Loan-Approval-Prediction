📌 Project Overview
This project predicts whether a loan application will be approved (Y) or rejected (N) based on applicant financial and demographic information.
It uses Machine Learning models such as:
-Logistic Regression
-Support Vector Machine (SVM)
-Random Forest Classifier

The dataset used is:
        Loan Prediction Dataset - RAW.csv

├── Loan Approval Prediction Proj.py     # Main project code
├── Loan Prediction Dataset - RAW.csv     # Training dataset
└── README.md


📊 Dataset Description

| Feature           | Description                 |
| ----------------- | --------------------------- |
| ApplicantIncome   | Monthly income of applicant |
| CoapplicantIncome | Income of co-applicant      |
| LoanAmount        | Amount of loan              |
| Loan_Amount_Term  | Loan term (months)          |
| Credit_History    | 1 = Good, 0 = Bad           |
| Gender            | Male/Female                 |
| Married           | Yes/No                      |
| Education         | Graduate/Not Graduate       |
| Self_Employed     | Yes/No                      |
| Property_Area     | Urban/Semiurban/Rural       |
| Loan_Status       | **Target variable (Y/N)**   |


🧪 Steps Performed in the Code
✔ 1. Load Dataset

Using pandas.read_csv() to load raw CSV.

✔ 2. Clean Column Names

Lowercasing, trimming spaces, replacing spaces with underscores.

✔ 3. Handle Missing Values

Categorical → Mode

Numerical → Median

✔ 4. Exploratory Data Analysis (EDA)

Visualization includes:

Countplots for categorical data

Histograms for numeric features

Boxplots

Scatter plots (colored by loan status)

Correlation heatmap

✔ 5. Label Encoding

Converts text features into numeric values.

✔ 6. Train–Test Split

Splits data into 70% train and 30% test.

✔ 7. Feature Scaling

StandardScaler ensures all features have similar scale.

✔ 8. Model Training

Trains:

LogisticRegression

SVC with RBF kernel

RandomForestClassifier

✔ 9. Model Evaluation

Accuracy, Confusion Matrix, Classification Report are printed.

✔ 10. Best Model Selection

Automatically selects the model with highest accuracy.

📝 How to Run This Project
1. Install Dependencies

Run:

pip install pandas numpy scikit-learn matplotlib seaborn

2. Place Dataset in the Project Folder

Ensure:

Loan Prediction Dataset - RAW.csv


is in the same directory as your Python script.

3. Run the Python Script
python "Loan Approval Prediction Proj.py"

📈 Output Examples

You will get:

Cleaned dataset preview

Missing value summary

Correlation heatmap

Model accuracies

Confusion matrices

Classification reports

Best model name and accuracy

Example output:

Logistic Regression Accuracy: 0.78
Support Vector Machine Accuracy: 0.82
Random Forest Accuracy: 0.80

Best Model: Support Vector Machine
Best Model Accuracy: 0.82

🧠 Technologies Used

Python 3

Pandas

NumPy

Scikit-Learn

Matplotlib

Seaborn

👩‍💻 Author

Shamima
Loan Approval Prediction ML Project
