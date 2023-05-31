import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing as pr
import seaborn as sns

data = pd.read_csv('titanic.csv')

print(data.describe())
print(data.head())

print(data.count())

print(data.isnull().sum())

sns.heatmap(data.isnull(), cbar = False).set_title("Kayıp veriler heatmap")
plt.show()

train_df_num = data
train_df_mean = train_df_num.copy()

mean_age = train_df_num["Age"].mean()
train_df_mean['Age'] = train_df_mean['Age'].fillna(np.round(mean_age,1))

sns.distplot(train_df_num["Age"], color="olive", label="Orijinal dataset")
sns.distplot(train_df_mean["Age"], color="teal", label="Ortalama ile değiştirilmiş dataset")
plt.show()
sns.catplot(x="Sex", hue="Survived",  #cinsiyete göre hayatta kalmayla alakalı bir grafik
            kind="count",data=data)

plt.show()


group = data.groupby(['Pclass','Survived']) #üst klasmanlarda yolculuk yapanlarla alakalı bir grafik
pclass_survived = group.size().unstack()
sns.heatmap(pclass_survived, annot=True, fmt="d")
plt.show()


sns.violinplot(x = "Sex", y="Age", hue= "Survived",  #hayatta kalma oranının yaş aralığına göre bir grafiği
               data=data, split= True)

plt.show()


data['Fare'] = pd.qcut(data['Fare'], 4)  #yüksek ücret ödeyenlerin hayatta kalmış olma olasılığını gösteren bir grafik.
sns.barplot(x = 'Fare', y = 'Survived',
            data= data)
plt.show()

sns.catplot(x ='Embarked', hue ='Survived',
kind ='count', col ='Pclass', data = data)
#bu grafikten sonra yolcuların buyuk cogunlugunun ucaga s den bindiğini görürüz
plt.show()

ages = data["Age"]
fare = data["Fare"]
plt.style.use("classic")
plt.hist(ages, color="red")
plt.title("Fare Plot by Age")
plt.xlabel("ages")
plt.ylabel("fare")
plt.tight_layout()
plt.show()

#histogram grafiği bilet ücretlerinin yaşlara göre dağılımı 



filter = data.loc[(data["Survived"] == 0) & (data["Sex"] == "male")]
print(filter)


#age değişkeni için aykırı değer olup olmadğını grafik ile anlama.

sns.boxplot(y = train_df_num["Age"])
plt.show()
train_df_show = train_df_num.copy()
train_df_show.head()









