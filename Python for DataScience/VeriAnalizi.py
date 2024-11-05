#######################################
# NUMPY
#######################################

# Neden NumPy?

import numpy as np

a = [1,2,3,4]
b = [2,3,4,5]

ab =[]

for i in range(0,len(a)):
    ab.append(a[i] * b[i])

# NumPy Hali;
a = np.array([1,2,3,4])
b = np.array([2,3,4,5])
a*b


# NumPy Array'i Oluşturmak

import numpy as np

np.array([1,2,3,4])
type(np.array([1,2,3,4]))

np.zeros(10, dtype=int)
np.random.randint(0, 10,size=10)
np.random.normal(10,4,(3,4))


# NumPy Array Özellikleri

# ndim: boyut sayısı
# sahpe: boyut bilgisi
# size: toplam eleman sayısı
# dtype: array veri tipi

a = np.random.randint(10,size=5)
a.ndim
a.shape
a.size
a.dtype



# Yeniden Şekillendirme (Reshaping)

import numpy as np

np.random.randint(1,10,size=9)
np.random.randint(1,10,size=9).reshape(3,3)

ar = np.random.randint(1,10,size=9)
ar.reshape(3,3)



# Index Seçimi (Index Selection)

import numpy as np

a = np.random.randint(10,size=10)

a[0]
a[0:5]
a[0] =999

b = np.random.randint(10,size=(3,5))

b[0,0]
b[2,3] = 999


# Fancy Index

import numpy as np

v = np.arange(0,30,3)
v[1]
v[4]

catch = [1,2,3]
v[catch]


# Numpy'da Koşullu İşlemler (Conditions on Numpy)

import numpy as np
v = np.array([1,2,3,4,5])

# Klasik döngü ile;
ab = []

for i in v:
    if i < 3:
        ab.append(i)

# Numpy ile;
v[v <3]


# MAtematiksel işlemler (Mathematical Operations)

import numpy as np

v = np.array([1,2,3,4,5])

v / 5
v ** 2
v - 1

np.subtract(v,1)
np.add(v,1)
np.mean(v)
np.sum(v)
np.min(v)
np.max(v)
np.var(v)


# İki bilinmeyenli denkelem

# 5*x0 + x1 = 12
# x0 + 3*x1 = 10

a = np.array([[5,1],[1,3]])
b = np.array([12,10])

np.linalg.solve(a,b)


###############################################
# PANDAS
###############################################

# Pandas Series

import pandas as pd

s = pd.Series([10,2,66,73,9])
type(s)

s.index
s.dtype
s.size
s.ndim
s.values
s.head(3)
s.tail(3)

# Veri okuma (Reading Data)

import pandas as pd

df = pd.read_csv("datasets/Advertising.csv")
df.head()

# Veriye Hılı Bakış (Quick Look At Data)

import pandas as pd
import seaborn as sns

df = sns.load_dataset("titanic")
df.head()
df.tail()
df.shape
df.info()
df.describe().T
df.isnull().values.any()
df.isnull().sum()
df["sex"].head()
df["sex"].value_counts()


# Pandas'ta Seçim İşlemleri (Selection in Pandas)

import pandas as pd
import seaborn as sns

df = sns.load_dataset("titanic")
df.head()

df.index
df[0:13]
df.drop(0,axis=0).head()

delete_index = [1,3,5,7]
df.drop(delete_index,axis=0).head(10)

# Değişkeni Indexe Çevirmek

df["age"].head()
df.age.head()

df.index = df["age"]
df.drop("age",axis=1).head()

# Değişkenler Üzerinde İşlemler

import pandas as pd
import seaborn as sns

pd.set_option('display.max_columns', None)
df = sns.load_dataset("titanic")
df.head()

"age" in df

df["age"].head()
type(df["age"].head())
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
df[["age"]].head()
type(df[["age"]].head())

df[["age", "alive"]]

df["age2"] = df["age"]**2

df.drop("age",axis=1).head()



