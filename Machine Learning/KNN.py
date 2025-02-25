import pandas as pd
from sklearn.metrics import classification_report , roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
pd.set_option('display.max_columns', None)


### 1) Exploratory Data Analysis (EDA)

df = pd.read_csv("C:/Users/duran/PycharmProjects/Mıuul/Machine Learning/datasets/diabetes.csv")

df.head()
df.shape
df.describe().T
df["Outcome"].value_counts()

### 2) Data Preprocessing & Feature Engineering (Veri Ön İşleme)

y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

X_scaled = StandardScaler().fit_transform(X)

X = pd.DataFrame(X_scaled,columns = X.columns)

### 3) Modeling & Prediction

knn_model = KNeighborsClassifier()._fit(X,y)

random_user = X.sample(1, random_state=45)

knn_model.predict(random_user)


### 4) Model Evaluation (Model Başarı DEğerlendirme)

# Confusion matrix için y_pred
y_pred = knn_model.predict(X)

# AUC için y_prob
y_prob = knn_model.predict_proba(X)[:,1]

print(classification_report(y, y_pred))
#accuracy 0.83
#f1-score 0.74

# AUC
roc_auc_score(y,y_prob)
#0.90

# şu ana kadar modeli öğrendiği veriler üzerinden test ettik
# veriyi ayıralım

cv_result = cross_validate(knn_model, X, y, cv=5,scoring=['accuracy','f1','roc_auc']) # veriyi beşe böl 1 iyle test et

cv_result['test_accuracy'].mean()
#accuracy 0.73

cv_result['test_f1'].mean()
#f1-score 0.59

cv_result['test_roc_auc'].mean()
# AUC 0.78

# sonuçları nasıl iyileştirebiliriz

### 5) Hyperparameter Optimization

knn_model = KNeighborsClassifier()
knn_model.get_params()

knn_params = {"n_neighbors": range(2,50)}

knn_gs_best = GridSearchCV(knn_model ,
                           knn_params,
                           cv = 5,
                           n_jobs = -1, verbose=1).fit(X,y)

knn_gs_best.best_params_
# {'n_neighbors': 17}

### 6) Final Model

knn_final = knn_model.set_params(**knn_gs_best.best_params_).fit(X,y)

cv_result = cross_validate(knn_final,
                           X,
                           y,
                           cv=5,scoring=['accuracy','f1','roc_auc'])

cv_result['test_accuracy'].mean()
#accuracy 0.76

cv_result['test_f1'].mean()
#f1-score 0.61

cv_result['test_roc_auc'].mean()
# AUC 0.81

# test etme
random_user = X.sample(1)

knn_final.predict(random_user)
