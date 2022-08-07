# -*- coding: utf-8 -*-
"""
Created on Sun Aug  7 18:33:13 2022

@author: vsund
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('eda_data.csv')
df.head()

#choose relevant columns
df.columns

df_model = df[['avg_salary','Rating','Size','Type of ownership', 'Industry', 'Sector', 'Revenue', 'num_comp','hourly', 'employer_provided','job_state', 'same_state', 'age', 'python_yn',
'spark', 'aws', 'excel', 'job_simp', 'seniority', 'desc_len']]
#get dummies(one hot encoding)

df_dum = pd.get_dummies(df_model)
#train test split

from sklearn.model_selection import train_test_split

X = df_dum.drop('avg_salary',axis=1)
y = df['avg_salary'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#multiple linear regression
import statsmodels.api as sm

X_sm = sm.add_constant(X)
model = sm.OLS(y, X_sm)
model.fit().summary()

from sklearn.linear_model import LinearRegression,Lasso
lm = LinearRegression()
lm.fit(X_train,y_train)

from sklearn.model_selection import cross_val_score
cross_val_score(lm, X_train,y_train,scoring='neg_mean_absolute_error',cv=3).mean()
#lasso regerssion
lm_l = Lasso(alpha=.13)
lm_l.fit(X_train,y_train)
cross_val_score(lm_l, X_train,y_train,scoring='neg_mean_absolute_error',cv=3).mean()

alpha = []
error = []

for i in range(1,100):
    alpha.append(i/100)
    lml = Lasso(alpha=(i/100))
    error.append(cross_val_score(lml, X_train,y_train,scoring='neg_mean_absolute_error',cv=3).mean())
    
plt.plot(alpha,error)
#random forest
from sklearn.ensemble import RandomForestRegressor

rf =RandomForestRegressor(n_estimators=170,criterion='absolute_error',max_features='auto')
rf.fit(X_train,y_train)
cross_val_score(rf, X_train,y_train,scoring='neg_mean_absolute_error',cv=3).mean()
#DecisionTreeRegressor
from sklearn.tree import DecisionTreeRegressor

dtr = DecisionTreeRegressor()
cross_val_score(dtr, X_train,y_train,scoring='neg_mean_absolute_error',cv=3).mean()

#xgboost
pip install xgboost
from xgboost import XGBRegressor

xg = XGBRegressor()
xg.fit(X_train,y_train)
cross_val_score(xg, X_train,y_train,scoring='neg_mean_absolute_error',cv=3).mean()

#tunning models Gridcv
from sklearn.model_selection import GridSearchCV

#params = {'n_estimators':range(170),'criterion':('mae'),'max_features':('auto','sqrt','log2')}
gv = GridSearchCV(rf, param_grid=params,scoring='neg_mean_absolute_error',cv=3)
gv.fit(X_train,y_train)

gv.best_score_
#test ensembles

tpred_lm = lm.predict(X_test)
tpred_lml = lm_l.predict(X_test)
tpred_rf = rf.predict(X_test)
tpred_xg = xg.predict(X_test)

from sklearn.metrics import mean_absolute_error

mean_absolute_error(y_test, tpred_lm)
mean_absolute_error(y_test, tpred_lml)
mean_absolute_error(y_test, tpred_rf)
mean_absolute_error(y_test, tpred_xg)

mean_absolute_error(y_test, (tpred_xg+tpred_rf+tpred_lm)/3)