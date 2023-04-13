#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#cancer = pd.read_csv(r"C:\Users\hp\Downloads\Cancer\data.csv") -- method 1 and method 2
cancer = pd.read_csv("dataset location")
cancer.info()


cancer.drop("Unnamed: 32", axis=1, inplace=True)

cancer

cancer.drop("id", axis=1, inplace=True)

cancer

get_ipython().run_line_magic('matplotlib', 'inline')
cancer.plot(x='diagnosis', kind='bar');


sns.set(style="whitegrid")
plt.figure(figsize=(6,4))
total = float(len(cancer))
ax = sns.countplot(x="diagnosis", hue="diagnosis", palette='bwr', data=cancer)
plt.title('Classification of Tumor', fontsize=16)
for p in ax.patches:
    percentage = '{:.1f}%'.format(100 * p.get_height()/total)
    x = p.get_x() + p.get_width()
    y = p.get_height()
    ax.annotate(percentage, (x, y),ha='center')
plt.show()


ax = sns.countplot(x='diagnosis', hue="diagnosis", palette='bwr', data=cancer)

for i in ax.containers:
    ax.bar_label(i)


plt.figure(figsize=(22,20))
sns.heatmap(cancer.corr(), annot=True, data=cancer, cmap='Blues')


cancer.columns


mean_col = ['diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean']

se_col = ['diagnosis', 'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
       'fractal_dimension_se']

worst_col = ['diagnosis', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave points_worst',
       'symmetry_worst', 'fractal_dimension_worst']


sns.pairplot(cancer[mean_col], hue='diagnosis', palette='Blues');

sns.pairplot(cancer[se_col], hue='diagnosis', palette='Greens');

sns.pairplot(cancer[worst_col], hue='diagnosis', palette='Oranges');


from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
import plotly

scale = RobustScaler()
X_train = scale.fit_transform(X_train)
X_test = scale.transform(X_test)


pca = PCA()
pca.fit(X_train)
exp_var_cumul = np.cumsum(pca.explained_variance_ratio_)

fig = px.line(x=np.arange(1,exp_var_cumul.shape[0]+1), y=exp_var_cumul, markers=True, labels={'x':'# of components', 'y':'Cumulative Explained Variance'})

#fig.add_shape(type='line', line=dict(dash='dash'),x0=0, x1=30, y0=0.95, y1=0.95)

fig.show()

cancer['diagnosis'] = cancer['diagnosis'].map({'B':0, 'M':1})

cancer['diagnosis'].value_counts()


from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(cancer.drop('diagnosis', axis=1),
                                                    cancer['diagnosis'], test_size=0.2)

print(X_train.shape)
print(X_test.shape)


from sklearn.preprocessing import StandardScaler

sts = StandardScaler()
X_train = sts.fit_transform(X_train)
X_test = sts.fit_transform(X_test)


from sklearn.linear_model import LinearRegression
import numpy as np

lgr = LinearRegression()
lgr.fit(X_train, Y_train)

lgr.score(X_test, Y_test)


from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor()
rfr.fit(X_train, Y_train)
rfr_p = rfr.score(X_test, Y_test)
print('Model Prediction is:', rfr_p)


from sklearn.svm import SVC

svc_m = SVC()
svc_m.fit(X_train, Y_train)
svc_p = svc_m.score(X_test, Y_test)
print('Model Prediction is:', svc_p)


from sklearn.linear_model import LogisticRegression

lg = LogisticRegression()
lg.fit(X_train, Y_train)
lg_p = lg.score(X_test, Y_test)
print('Model Prediction is:', lg_p)


from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()
rfc.fit(X_train, Y_train)
rfc_p = rfc.score(X_test, Y_test)
print('Model Prediction is:', rfc_p)


plt.figure(figsize=(8,6))
plt.title('Accuracy for each model', fontsize=16)
model_acc = [rfr_p, svc_p, rfc_p, lg_p]
model_name = ['RandomForestRegresser', 'SVM', 'LogisticRegression', 'RandomForestClassifier']
ax = sns.barplot(x= model_name, y=model_acc, palette='bwr');

for p in ax.patches:
    percentage = '{:.1f}%'.format(100 * p.get_height())
    x = p.get_x() + p.get_width()
    y = p.get_height()
    ax.annotate(percentage, (x, y),ha='center')
plt.show()
