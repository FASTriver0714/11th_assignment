# 라이브러리 및 데이터 불러오기

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from sklearn.datasets import load_wine

from sklearn.model_selection import train_test_split, GridSearchCV

import matplotlib.pyplot as plt

wine = load_wine()

# feature로 사용할 데이터에서는 'target' 컬럼을 drop합니다.
# target은 'target' 컬럼만을 대상으로 합니다.
# X, y 데이터를 test size는 0.2, random_state 값은 42로 하여 train 데이터와 test 데이터로 분할합니다.
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = wine.target

''' 코드 작성 바랍니다 '''

X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = wine.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

####### A 작업자 작업 수행 #######

''' 코드 작성 바랍니다 '''

# DecisionTreeClassifier 모델링
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

dt_param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [2, 5],
    'min_samples_split': [2, 10],
    'min_samples_leaf': [1, 2, 4]
}

dt_model = DecisionTreeClassifier(random_state=42)
dt_grid_search = GridSearchCV(dt_model, dt_param_grid, cv=5, scoring='accuracy')
dt_grid_search.fit(X_train, y_train)

dt_best_model = dt_grid_search.best_estimator_
dt_pred = dt_best_model.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_pred)


# Feature Importance 시각화
plt.figure(figsize=(12, 6))
feature_importance = dt_best_model.feature_importances_
plt.bar(wine.feature_names, feature_importance)
plt.title('Feature Importance')
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

####### B 작업자 작업 수행 #######

from xgboost import XGBClassifier

xgb_param_grid = {
    'max_depth': [3, 5, 7, 9, 15],
    'learning_rate': [0.1, 0.01, 0.001],
    'n_estimators': [50, 100, 200, 300]
}

xgb_model = XGBClassifier(random_state=42)
xgb_grid_search = GridSearchCV(xgb_model, xgb_param_grid, cv=5, scoring='accuracy')
xgb_grid_search.fit(X_train, y_train)

xgb_best_model = xgb_grid_search.best_estimator_
xgb_pred = xgb_best_model.predict(X_test)
xgb_accuracy = accuracy_score(y_test, xgb_pred)

print(f"XGB 최적 파라미터: {xgb_grid_search.best_params_}")
print(f"XGB 정확도: {xgb_accuracy:.4f}")

# Feature Importance 시각화
plt.figure(figsize=(12, 6))
feature_importance_xgb = xgb_best_model.feature_importances_
plt.bar(wine.feature_names, feature_importance_xgb)
plt.title('Feature Importance')
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
