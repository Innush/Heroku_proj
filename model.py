import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

Bank_data = pd.read_csv('Bank_Personal_Loan_Modelling.csv')
X = Bank_data[['Age', 'Experience', 'Income', 'ZIP Code', 'Family', 'CCAvg', 'Education']]

y = Bank_data['Personal Loan']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

lm = LogisticRegression()
lm.fit(X_train, y_train)

pickle.dump(lm, open('model.pkl', 'wb'))
