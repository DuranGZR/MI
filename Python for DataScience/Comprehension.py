# LİST COMPREHENSİON

salaries = [1000,2000,3000,4000,5000]

a = [salary * 2 for salary in salaries]

b = [salary * 2 for salary in salaries if salary < 3000]

c = [salary * 2 if salary <3000 else salary * 0 for salary in salaries]


students = ["John","Mark","Venessa","Mariam"]
student_no = ["John","Venessa"]

[student.lower() if student in student_no else student.upper() for student in students]
##########################

# DİCT COMPREHENSİON

dictionary = {'a':1,
              'b':2,
              'c':3,
              'd':4,
              'e':5}

dictionary.keys()
dictionary.values()
dictionary.items()

{k: v **2 for (k,v) in dictionary.items()}

{k.upper(): v  for (k,v) in dictionary.items()}

{k.upper(): v*2  for (k,v) in dictionary.items()}

###########################################
# UYGULAMA - MÜLAKAT SORUSU
###########################################

# Amaç : çift sayıların karesi alınarak bir sözlüğe eklemek istenmektedir.
# Key'ler orjinal değerler value'lar ise değiştirilmiş değerler olacak.

numbers = range(10)
new_dict = {}

{n: n ** 2 for n in numbers if n % 2 == 0}

#########################################
# List & Dict Comprehension Uygulamalar
#########################################


#########################################
# Bir Veri Setindeki Değişken İsimlerine Değiştirmek
#########################################

import seaborn as sns
df = sns.load_dataset("car_crashes")
df.columns = [col.upper() for col in df.columns]

########################################
# İsminde "INS" olan değişkenlerin başına FLAG diğerlerine NO_FLAG eklemek istiyoruz.
########################################

[col for col in df.columns if "INS" in col]

["FLAG_" + col for col in df.columns if "INS" in col]

["FLAG_" + col if "INS" in col else "NO_FLAGE" + col for col in df.columns]


########################################
# Amaç key'i string, value'su liste içinde fonksiyon adları olan sözlük oluşturmak.
# Sadece sayısal değişkenler için yapmak istiyoruz.
########################################


import seaborn as sns

df = sns.load_dataset("car_crashes")
df.columns

num_cols = [ col for col in df.columns if df[col].dtypes != "0"]

soz= {}
agg_list =["mean","min","max","sum"]

for col in num_cols:
    soz[col] = agg_list

#kısa yol
new_dict = {col: agg_list for col in num_cols}

df[num_cols].head()

df[num_cols].agg(new_dict)



