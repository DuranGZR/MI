################# Veri Görselleştirme : MATPLOTLIB & SEABORN #################
import numpy as np
# Kategorik değişken : sütun grafik. countplot bar
# Sayısal değişken   : hist, boxplot


# Kategorik değişken görselleştirme #

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")
df.head()

df["sex"].value_counts().plot(kind='bar')
plt.show()


# Sayısal değişken görselleştirme #

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")
df.head()

plt.hist(df["age"])
plt.show()

plt.boxplot(df["fare"])
plt.show()


# Matplotlib'in özellikleri
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

#Plot

x = np.array([1,8])
y = np.array([0,150])

plt.plot(x,y)
plt.show()

# Marker

y = np.array([13,28,11,100])

plt.plot(y,marker='o')
plt.show()

plt.plot(y,marker='*')
plt.show()

# Line

y = np.array([13,28,11,100])

plt.plot(y, linestyle='dashed')
plt.show()


# Labels

x = np.array([80,85,95,100,105,110,120,125,130])
y = np.array([240,250,260,270,280,290,300,310,320])

plt.plot(x,y)
plt.show()

## başlık

plt.title("Bu ana başlık")

# X ekseni isimlendirme

plt.xlabel("x ekseni isimlendirme")
plt.ylabel("y ekseni isimlendirme")

plt.grid()


##### SEABORN #####

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

df = sns.load_dataset("tips")
df.head()

df["sex"].value_counts()
sns.countplot(x =df["sex"],data =df)
plt.show()

#matplot ile
df["sex"].value_counts().plot(kind='bar')
plt.show()

# Seaborn ile sayısal görselleşrirme

sns.boxplot(x=df["total_bill"])
plt.show()

df["total_bill"].hist()
plt.show()







