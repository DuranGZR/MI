import warnings
import joblib
import pydotplus
import numpy as np
import pandas as pd
import seaborn as sns
from docutils.nodes import label
from matplotlib import pyplot as plt
from numba.core.typing.builtins import Print
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, validation_curve
from skompiler import skompile

pd.set_option('display.max_columns', None)
warnings.simplefilter(action='ignore', category=Warning)

#####  1) EDA (Keşifci veri analizi)  #####
df = pd.read_csv("C:/Users/duran/PycharmProjects/Mıuul/Machine Learning/datasets/diabetes.csv")

# İlk 5 satırı görüntüleme
print(df.head())

# Veri setinin boyutu
print(f"Veri seti boyutu: {df.shape}")

# Genel istatistikler
print(df.describe().T)

# Sınıf dağılımı
print(df["Outcome"].value_counts(normalize=True))  # Oranları görmek için normalize=True

# Eksik değer kontrolü
print(df.isnull().sum())

# Temel dağılım analizleri
plt.figure(figsize=(10, 6))
sns.histplot(df, kde=True)
plt.title("Genel Dağılım")
plt.show()

# Sınıf bazında özellik dağılımı
for col in df.columns[:-1]:  # Son sütun "Outcome" olduğu için hariç tutuyoruz
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df["Outcome"], y=df[col])
    plt.title(f"Outcome'a Göre {col} Dağılımı")
    plt.show()


##### 2) Data Preprocessing & Feature Engineering (Veri Ön İşleme) ######


# Bağımlı ve bağımsız değişkenleri ayırma
y = df["Outcome"]
X = df.drop(["Outcome"], axis=1).copy()

# Eksik değerlerin olup olmadığını kontrol etme
X.replace(0, np.nan, inplace=True)  # 0 olan değerleri NaN ile değiştir (Bazı değişkenlerde 0 anlamsız olabilir, örn: Glukoz)
X.fillna(X.median(), inplace=True)  # Eksik değerleri medyan ile doldur

# Aykırı değer analizi (IQR yöntemi)
Q1 = X.quantile(0.25)
Q3 = X.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Aykırı değerleri belirleme
outliers = ((X < lower_bound) | (X > upper_bound))
print("Aykırı değer sayısı:\n", outliers.sum())

# Aykırı değerleri sınır değerlerine çekme (Winsorization)
X = X.mask(X < lower_bound, lower_bound, axis=1)  # Alt sınırdan küçük olanları alt sınıra eşitle
X = X.mask(X > upper_bound, upper_bound, axis=1)  # Üst sınırdan büyük olanları üst sınıra eşitle

# Standardizasyon
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Standardize edilmiş DataFrame oluşturma
X = pd.DataFrame(X_scaled, columns=df.drop(["Outcome"], axis=1).columns)

# Son haliyle bağımsız değişkenleri görüntüleme
print(X.head())



###### 3) Modeling using CART ######

cart_model = DecisionTreeClassifier(random_state=1).fit(X, y)

# Confusion matris için y_pred
y_pred = cart_model.predict(X)

# AUC için y_prob
y_prob = cart_model.predict_proba(X)[:, 1]

# Confusion matrix
print( classification_report(y, y_pred))

# AUC
roc_auc_score(y, y_prob)


###### 4) Holdout yöntemi ile başarı değerlendirme ######

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=17)

cart_model = DecisionTreeClassifier(random_state=17).fit(X_train, y_train)

#Train hatası
y_pred = cart_model.predict(X_train)
y_prob = cart_model.predict_proba(X_train)[:, 1]

print(classification_report(y_train, y_pred))

roc_auc_score(y_train, y_prob)


# Test hatası
y_pred = cart_model.predict(X_test)
y_prob = cart_model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))

roc_auc_score(y_test, y_prob)

############ CV ile başarı değerlendirme ##############

cart_model = DecisionTreeClassifier(random_state=17).fit(X, y)

cv_results = cross_validate(cart_model,
                            X,
                            y,
                            cv=5, scoring=['accuracy', 'f1', 'roc_auc'])

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()



############  Hyperparameter Optimization with GridSearchCV #############

cart_model.get_params()

cart_params = {'max_depth': range(1,11),
               'min_samples_split' : range(2,20)}


cart_best_grid = GridSearchCV(cart_model, cart_params, cv=5, n_jobs=-1, verbose=1).fit(X, y)


cart_best_grid.best_params_

cart_best_grid.best_score_

random = X.sample(1,random_state=45)

cart_best_grid.predict(random)


###################  Final Model  #######################

cart_final = DecisionTreeClassifier(**cart_best_grid.best_params_ , random_state=17).fit(X, y)


cart_final.get_params()

cv_results = cross_validate(cart_final,
                            X,
                            y,
                            cv=5, scoring=['accuracy', 'f1', 'roc_auc'])

cv_results['test_accuracy'].mean() # 0.74

cv_results['test_f1'].mean() # 0.64

cv_results['test_roc_auc'].mean() # 0.77


##############  Özellik Önemi (Feature Importance)  ##################


def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])

    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


plot_importance(cart_final, X)


##############  Öğrenme Eğrileriyle Model Karmaşıklığını Analiz Etme  ############################

train_score, test_score = validation_curve(cart_final, X, y,
                                           param_name="max_depth",
                                           param_range=range(1, 11),
                                           cv= 10,
                                           scoring="roc_auc")


mean_train_score = np.mean(train_score, axis=1)
mean_test_score = np.mean(test_score, axis=1)



plt.plot(range(1,11),mean_train_score,
         label= "Training Score",color= "b")

plt.plot(range(1,11),mean_test_score,
         label= "Validation Score",color= "g")

plt.title("Validation Curve with CART")
plt.xlabel("Number of Max Depth")
plt.ylabel("AUC Score")
plt.tight_layout()
plt.legend(loc="best")
plt.show()



def val_curve_params(model, X, y, param_name, param_range, scoring="roc_auc", cv=10):
    train_score, test_score = validation_curve(
        model, X=X, y=y, param_name=param_name, param_range=param_range, scoring=scoring, cv=cv)

    mean_train_score = np.mean(train_score, axis=1)
    mean_test_score = np.mean(test_score, axis=1)

    plt.plot(param_range, mean_train_score,
             label="Training Score", color='b')

    plt.plot(param_range, mean_test_score,
             label="Validation Score", color='g')

    plt.title(f"Validation Curve for {type(model).__name__}")
    plt.xlabel(f"Number of {param_name}")
    plt.ylabel(f"{scoring}")
    plt.tight_layout()
    plt.legend(loc='best')
    plt.show()


val_curve_params(cart_final, X, y, "max_depth", range(1, 11), scoring="f1")


cart_val_params =[["max_depth",range(1,11)],["min_samples_split",range(2,20)]]

for i in range(len(cart_val_params)):
    val_curve_params(cart_final, X, y, cart_val_params[i][0], cart_val_params[i][1])


####################  Görselleştirme ( Visualizing the Decision Tree )  #############################

# conda install graphviz
import graphviz


def tree_graph(model, col_names, file_name):
    tree_str = export_graphviz(model, feature_names=col_names, filled=True, out_file=None)
    graph = pydotplus.graph_from_dot_data(tree_str)
    graph.write_png(file_name)

tree_graph(model= cart_final, col_names=X.columns, file_name="cart_final.png")







