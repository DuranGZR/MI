#####################################################################
# Diabetes Prediction with Logistic Regression
#####################################################################

# İş problemi:

# Özellikleri belirtildiğinde kişilerin diyabet hastası olup olamdığını
# olmadıklarını tahmmin eden bir makine öğrenmesi modeli geliştire bilir misiniz?

# Değişkenler
# Pregnancies : Hamilelik sayısı
# Glucose : Glikoz
# BloodPressure : Kan basıncı
# SkinThickness : Cilt kalınlığı
# Insulin : İnsülin
# BMI : Bedwen kitle indeksi
# DiabetesPedigreeFunction : Soyumuzdaki kişilere göre diyabet olma ihtimalimizi hesaplayan bir fonksiyon
# Age : Yaş
# Outcome : Kişinin diyabet olup olmadığı bilgisi. Hastalığa sahip (1) ya da değil (0)


# 1. Exploratory Data Analysis
# 2. Data Preprocessing
# 3. Model & Prediction
# 4. Model Evaulation
# 5. Model Validation : Holdout
# 6. Model Validation : 10-Fold Cross Validation
# 7. Prediction for A New Observation


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from astropy.units.quantity_helper.function_helpers import block

from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import roc_curve, roc_auc_score




pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

# Gerekli Fonks.

def outlier_treshholds(dataframe, col_name, q1 = 0.05, q3 = 0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    inter_quantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * inter_quantile_range
    low_limit = quartile1 - 1.5 * inter_quantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_treshholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

#############################
# Exploratory Data Analysis
#############################

df = pd.read_csv(f"C:/Users/duran/PycharmProjects/Mıuul/Machine Learning/datasets/diabetes.csv")
df.head()
df.shape

##################
# Target Analizi
##################

df["Outcome"].value_counts()

sns.countplot(x="Outcome", data=df)
plt.show()

# yüzde hesabı
100 * df["Outcome"].value_counts() / len(df)

########################
# Feature'ların Analizi
########################

df.describe().T

df["BloodPressure"].hist(bins=20)
plt.xlabel("BloodPressure")
plt.show()

df["Glucose"].hist(bins=20)
plt.xlabel("Glucose")
plt.show()

# Hepsini gösteröek için fonks.
def plot_numerical_col(dataframe, numerical_col):
    dataframe[numerical_col].hist(bins=20)
    plt.xlabel(numerical_col)
    plt.show(block=True)

for col in df.columns:
    plot_numerical_col(df, col)

# Bağımlı değişkeni istemiyorsak:
cols = [col for col in df.columns if "Outcome" not in col]

for col in cols:
    plot_numerical_col(df, col)

##############################
# Target vs Features
#############################

df.groupby("Outcome").agg({"Pregnancies": "mean"})

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end= "\n\n\n")

for col in cols:
    target_summary_with_num(df, "Outcome", col)


##################################################
# Data Preprocessing : Veri Ön İşleme
##################################################

df.shape
df.head()

# Eksik değer var mı
df.isnull().sum()

df.describe().T

for col in cols:
    print( col, check_outlier(df, col))

for col in cols:
    df[col] = RobustScaler().fit_transform(df[[col]])

df.head()

################################
# Model & Prediction
################################

# bağımlı değişken tanımla
y = df["Outcome"]

# bağımsız değişkenleri tanımla
X = df.drop(["Outcome"], axis=1)

# Model kurma
log_model = LogisticRegression().fit(X, y)

log_model.intercept_
log_model.coef_

y_pred = log_model.predict(X)

# tahmin edilen değerler:
y_pred[0:10]

# gerçek değerler
y[0:10]


####################################################
# Model Evaulation : Model başarı değerlendirme
####################################################

def plot_confusion_mateix(y,y_pred):
    acc = round(accuracy_score(y, y_pred),2)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt='.0f')
    plt.xlabel('y_pred')
    plt.ylabel('y')
    plt.title('Accuracy Score : {0}'.format(acc),size=10)
    plt.show()

plot_confusion_mateix(y, y_pred)

print(classification_report(y, y_pred))

# Accurasy : 0.78
# Precision : 0.74
# Recall : 0.58
# F1-score : 0.65


# ROC AUC
y_prob = log_model.predict_proba(X)[:,1]
roc_auc_score(y, y_prob)
# 0.83

#########################################################
# Model Validation : Holdout  (veriyi iki oarçaya ayırma)
#########################################################

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.20,
                                                    random_state=17)

log_model = LogisticRegression(max_iter=1000).fit(X_train, y_train)

# modele göstermediğimiz veriyi soruyoruz
y_pred = log_model.predict(X_test)
y_prob = log_model.predict_proba(X_test)[:,1]

# tahmin ettiğimiz dfeğerler ile gerçek değerleri karşılalştırıyoruz
print(classification_report(y_test, y_pred))

# Accuracy : 0.77
# Precision : 0.79
# Recall : 0.53
# F1-score : 0.63




# Modelden olasılık tahminlerini alın


# ROC Eğrisi için FPR (False Positive Rate) ve TPR (True Positive Rate) hesaplayın
fpr, tpr, _ = roc_curve(y_test, y_prob)

# AUC değerini hesaplayın
auc = roc_auc_score(y_test, y_prob)
# 0.88

# ROC Eğrisini Çizin
plt.figure()
plt.plot(fpr, tpr, label=f'Logistic Regression (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], 'r--')  # Rastgele model çizgisi
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.show()


##############################################################
# Model Validation : 10-Fold Cross Validation
##############################################################

# bağımlı değişken tanımla
y = df["Outcome"]

# bağımsız değişkenleri tanımla
X = df.drop(["Outcome"], axis=1)

log_model = LogisticRegression(max_iter=1000).fit(X, y)

cv_results = cross_validate(log_model,
                            X,
                            y,
                            cv=5,
                            scoring=['accuracy','precision','recall','f1','roc_auc'])

cv_results['test_accuracy'].mean()
# accuracy : 0.7721

cv_results['test_precision'].mean()
# precision : 0.7196

cv_results['test_recall'].mean()
# recall : 0.5747

cv_results['test_f1'].mean()
# f1 : 0.6374

cv_results['test_roc_auc'].std()
# roc_auc : 0.0238

####################################################
# Prediction for A New Observation
####################################################

X.columns

random_user = X.sample(1,random_state=45)
log_model.predict(random_user)
# sonuç 1 yani kişimiz diyabet.