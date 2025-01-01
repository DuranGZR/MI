import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('C:/Users/duran/PycharmProjects/Mıuul/GDG/data.csv',sep=";")
print(data.head())

data=data.apply(lambda x:x.str.replace(",","."))

print(data)

data.info()

data["enflasyon"] = data["enflasyon"].astype("float")
data["issizlik"] = data["issizlik"].astype("float")

data.info()



# Boş verileri silme
data.tail(2)

data.dropna(inplace=True)

data.tail(2)

data["yeni ver"] = data.enflasyon - data.issizlik
data.head()


data.insert(2,column="yeni_hesaplama",value=data.enflasyon+data.enflasyon)
data.head()



data.describe()

data.tail()
