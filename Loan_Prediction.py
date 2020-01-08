#Importing libraries

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix

#Importing Dataset

training_set = pd.read_csv('Loan_Prediction_Train.csv')
test_set = pd.read_csv('Loan_Prediction_Test.csv')
test_set['Loan_Status'] = 'Y'
sample_submission = test_set[['Loan_ID','Loan_Status']]
dataset = pd.concat([training_set, test_set])

#Data Exploration
dataset.describe()


#Checking for missing values
dataset.apply(lambda x: sum(x.isnull()),axis=0) 
dataset.info()

#Dropping ID variable
dataset.drop(columns = ['Loan_ID'], inplace = True)


cat_var = ['Gender', 'Married', 'Dependents', 'Education',
           'Self_Employed', 'Credit_History', 'Property_Area',
           'Loan_Status']

for var in cat_var:
    print(dataset[var].value_counts())


#Imputing missing  values with mode
for var in cat_var:
    dataset[var].fillna(dataset[var].mode()[0], inplace = True)
  

#Checking data distribution
table = dataset.pivot_table(values='LoanAmount', index='Self_Employed' ,
                       columns='Education', aggfunc=np.median)

def fage(x):
    return table.loc[x['Self_Employed'],x['Education']]

dataset['LoanAmount'].fillna(dataset[dataset['LoanAmount'].isnull()].apply(
        fage, axis=1), inplace=True)


plt.hist(dataset['Loan_Amount_Term'])

# This graph clearly suggests that most of the loans taken out are year long so it'll be sensible to impute them with 365.
dataset['Loan_Amount_Term'].fillna(360, inplace = True)


dataset['Dependents'].unique()
dataset.loc[dataset['Dependents'] == '3+', 'Dependents'] = 4

#Rechecking for missing values
dataset.apply(lambda x: sum(x.isnull()), axis=0) 

dataset['Dependents'] = dataset['Dependents'].astype(int)

#Encoding categorical variables
le = LabelEncoder()
for i in cat_var:
    dataset[i] = le.fit_transform(dataset[i])

#Feature engineering
dataset['Total_Income'] = dataset['ApplicantIncome'] + dataset['CoapplicantIncome']
dataset['Total_Income_Log'] = np.log(dataset['ApplicantIncome'] + dataset['CoapplicantIncome'])
dataset['DebtIncomeRatio'] = dataset['LoanAmount']/dataset['Total_Income']

#Splitting the dataset back into train and test
training_set = dataset[:614]    
test_set = dataset[614:]

X_train = training_set.drop(columns = ['Loan_Status'])
y_train = training_set['Loan_Status']

X_test = test_set.drop(columns = ['Loan_Status'])

classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy',
                                    random_state = 0)

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

pd.DataFrame(classifier.feature_importances_,
                                   index = X_train.columns,
                                    columns=['importance']).sort_values('importance',  
                                            ascending=False)

top_features=  ['Credit_History', 'DebtIncomeRatio',
                'ApplicantIncome', 'LoanAmount',
                'Total_Income']

classifier.fit(X_train[top_features], y_train)

y_pred = classifier.predict(X_test[top_features])


#Grid Search to find the best parameters
parameters = [{'n_estimators': [10,50,100,500] ,
               'max_depth': np.linspace(1,32,32), 
               'criterion': ['gini']
                },
                {'n_estimators': [10,50,100,500] ,
               'max_depth': np.linspace(1,32,32), 
               'criterion': ['entropy']
                }]


grid_search = GridSearchCV(estimator= classifier, 
                           param_grid = parameters,
                           scoring= 'accuracy',
                           cv = 30,
                           n_jobs = -1)

grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_

#Final model 
classifier = RandomForestClassifier(n_estimators = 50, criterion = 'entropy',
                                    random_state = 0, max_depth= 9.0)

classifier.fit(X_train[top_features], y_train)
y_pred = classifier.predict(X_test[top_features])


sample_submission['Loan_Status'] = y_pred
sample_submission['Loan_Status'] = np.where(sample_submission['Loan_Status']==0, 'N', 'Y')
sample_submission.to_csv('sample_submission.csv', sep = ',', index = False)
