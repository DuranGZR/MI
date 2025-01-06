##################################################################
# Sales Prediction with Lİnear regression
#################################################################

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

pd.set_option('display.float_format', lambda x: '%.2f' % x)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error , mean_absolute_error
from sklearn.model_selection import train_test_split , cross_val_score

########################################################
# Simple Linear Regression with OLS Using Scikit-Learn #
########################################################

df = pd.read_csv('C:/Users/duran/PycharmProjects/Mıuul/Machine Learning/datasets/advertising.csv')

df.shape

X = df[["TV"]]
Y = df[["sales"]]


##############
# MODEL
##############

reg_model = LinearRegression().fit(X, Y)

# Y_hat = b + w*x

#Sabit (b - bias)
reg_model.intercept_[0]

# TV nin kat sayısı w1
reg_model.coef_[0][0]


################################
#  TAHMİN
################################


# 150 birimlik TV harcaması olsa ne kadar satış olması beklenir?

reg_model.intercept_[0] * reg_model.coef_[0][0] * 150

# 500 birimlik TV harcaması olsa ne kadar satış olması beklenir?

reg_model.intercept_[0] * reg_model.coef_[0][0] * 500

df.describe().T

#modelin görselleştrilmesi

g = sns.regplot(x= X ,y=Y,scatter_kws={'color': 'b', 's': 9}, ci=False,color= 'r')

g.set_title(f"Model denklemi Sales: {round(reg_model.intercept_[0],2)} +  TV*{round(reg_model.coef_[0][0], 2 )}")
g.set_ylabel("Satış Sayısı")
g.set_xlabel("TV Harcemaları")
plt.xlim(-10,310)
plt.ylim(bottom=0)
plt.show()



################################
# Tahmin Başarısı
################################

#MSE
y_pred = reg_model.predict(X)
mean_squared_error(Y , y_pred)
#10.51

Y.mean()
Y.std()

#RMSE
np.sqrt(mean_squared_error(Y , y_pred))
#3.24

#MAE
mean_absolute_error(Y , y_pred)
#2.54

# R-Kare
reg_model.score(X,Y)



#################################################################################################################################
#          Multiple Linear Regression              #
#################################################################################################################################

df = pd.read_csv('C:/Users/duran/PycharmProjects/Mıuul/Machine Learning/datasets/advertising.csv')

X = df.drop(['sales'], axis = 1)  # Sales harisi değişkenleri tutar

y = df['sales']

######################
#  MODEL
#####################

X_trein ,X_test , y_trein , y_test = train_test_split(X, y, test_size=0.20, random_state=1)

X_trein.shape
X_test.shape

reg_model = LinearRegression().fit(X_trein, y_trein)

# Sabit (b - bias)
reg_model.intercept_
# 2.90

# Katsayı w
reg_model.coef_
#[0.0468431 , 0.17854434, 0.00258619]

####################
# TAHMİN
####################

# Aşağıdaki gözlem edğerlerine göre satışın değeri ne olur?

# TV - 30
# radio - 10
# newspaper - 40

# Sales = 2.90 + TV*0.04 + radio*0.17 + newspaper*0.002
2.90 + 30*0.04 + 10*0.17 + 40*0.002

yeni_veri = [[30],[10],[40]]

yeni_veri = pd.DataFrame(yeni_veri).T

reg_model.predict(yeni_veri)

###################################
# Tahmin başarısını değerlendirme
###################################

# Trein RMSE

y_pred = reg_model.predict(X_trein)
np.sqrt(mean_squared_error(y_trein , y_pred))
#1.73
# Trein RKARE

reg_model.score(X_trein, y_trein)

# Test Trein

y_pred = reg_model.predict(X_test)
np.sqrt(mean_squared_error(y_test , y_pred))
#1.41

# Test RKARE

reg_model.score(X_test, y_test)


# 10 katlı CV RMSE
np.mean(np.sqrt(-cross_val_score(reg_model,
                                 X,
                                 y,
                                 cv=10,
                                 scoring='neg_mean_squared_error')))
#1.69

# 5 katlı CV RMSE
np.mean(np.sqrt(-cross_val_score(reg_model,
                                 X,
                                 y,
                                 cv=5,
                                 scoring='neg_mean_squared_error')))
#1.71



#####################################################################################################
#####################################################################################################
#                 Simple Linear Regression with Gradient Descent from Scratch                       #
#####################################################################################################
#####################################################################################################

# Cost function MSE
def cost_function(Y,b,w,X):
    m = len(Y)
    sse = 0

    for i in range(0,m):
        y_hat = b + w * X[i]
        y = Y[i]
        sse += (y_hat - y) ** 2

    mse = sse / m
    return mse



# Update kuralı
def update_weights(Y,b,w,X,learning_rate):
    m = len(Y)

    b_deriv_sum = 0
    w_deriv_sum = 0

    for i in range (0,m):
        y_hat = b+ w * X[i]
        y = Y[i]
        b_deriv_sum += (y_hat - y)
        w_deriv_sum += (y_hat - y ) * X[i]

    new_b = b - (learning_rate *1 / m * b_deriv_sum)
    new_w = w - (learning_rate *1 / m * w_deriv_sum)
    return new_b, new_w



# Train foksiyonu
def train(Y,initial_b,initial_w,X,learning_rate,num_iters):

    print("Starting gradient descent at b = {0}, w = {1}, mse = {2}".format(initial_b,initial_w,
                                                                             cost_function(Y,initial_b,initial_w,X)))

    b = initial_b
    w = initial_w

    cost_history = []
    for i in range(num_iters):
        b,w = update_weights(Y,b,w,X,learning_rate)
        mse = cost_function(Y,b,w,X)
        cost_history.append(mse)

        if i % 100 == 0:
            print("iter={:d} b={:.2f} w={:.4f} mse={:.4f}".format(i,b,w,mse))

    print("After {0} iteration b={1}, w ={2}, mse={3}".format(num_iters,b,w,cost_function(Y,b,w,X)))
    return cost_history,b,w


df = pd.read_csv('C:/Users/duran/PycharmProjects/Mıuul/Machine Learning/datasets/advertising.csv')

X = df["radio"]
Y = df["sales"]


# hyperparameters
learning_rate = 0.001
initial_b = 0.001
initial_w = 0.001
num_iters = 100000

cost_history, b, w = train(Y,initial_b,initial_w,X,learning_rate,num_iters)



























