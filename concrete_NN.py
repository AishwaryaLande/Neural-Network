import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

con = pd.read_csv("C:/Users/ADMIN/Desktop/Data_Science_Assig/Neural Network/concrete (1).csv")
con.shape
con.columns
con.describe()
con.head()
con.drop_duplicates(keep='first',inplace=True)
con.isnull()
con.isnull().sum()

plt.hist(con['cement']);plt.xlabel('cement');plt.ylabel('strength');plt.title('histogram of cement')
plt.hist(con['slag']);plt.xlabel('slag');plt.ylabel('strength');plt.title('histogram of slag')
plt.hist(con['ash']);plt.xlabel('ash');plt.ylabel('strength');plt.title('histogram of ash')
plt.hist(con['water']);plt.xlabel('water');plt.ylabel('strength');plt.title('histogram of water')
plt.hist(con['superplastic']);plt.xlabel('superplastic');plt.ylabel('strength');plt.title('histogram of superplastic')
plt.hist(con['coarseagg']);plt.xlabel('coarseagg');plt.ylabel('strength');plt.title('histogram of coarseagg')
plt.hist(con['fineagg']);plt.xlabel('fineagg');plt.ylabel('strength');plt.title('histogram of fineagg')
plt.hist(con['age']);plt.xlabel('age');plt.ylabel('strength');plt.title('histogram of age')

from scipy.stats import skew, kurtosis
skew(con)
kurtosis(con)

sns.boxplot(con.cement)
sns.boxplot(con.slag)
sns.boxplot(con.ash)
sns.boxplot(con.water)
sns.boxplot(con.superplastic)
sns.boxplot(con.coarseagg)
sns.boxplot(con.fineagg)
sns.boxplot(con.strength)

sns.pairplot(con)

#Q-plot
plt.plot(con);plt.legend(list(con.columns))
cement= np.array(con['cement'])
slag = np.array(con['slag'])
ash = np.array(con['ash'])
water = np.array(con['water'])
superplastic = np.array(con['superplastic'])
coarseagg= np.array(con['coarseagg'])
fineagg = np.array(con['fineagg'])
age = np.array(con['age'])
strength = np.array(con['strength'])

from scipy import stats
stats.probplot(cement, dist='norm', plot=plt);plt.title('Probability Plot of cement')
stats.probplot(slag, dist='norm', plot=plt);plt.title('Probability Plot of slag')
stats.probplot(ash, dist='norm', plot=plt);plt.title('Probability Plot of ash')
stats.probplot(water, dist='norm', plot=plt);plt.title('Probability Plot of water')
stats.probplot(superplastic, dist='norm', plot=plt);plt.title('Probability Plot of superplastic')
stats.probplot(coarseagg, dist='norm', plot=plt);plt.title('Probability Plot of coarseagg')
stats.probplot(fineagg, dist='norm', plot=plt);plt.title('Probability Plot of fineagg')
stats.probplot(age, dist='norm', plot=plt);plt.title('Probability Plot of age')
stats.probplot(strength, dist='norm', plot=plt);plt.title('Probability Plot of strength')

corr = con.corr()
corr
sns.heatmap(corr, annot=True)

x = con.drop(["strength"],axis=1)
y = con["strength"]
x=x.astype('int')
y=y.astype('int')
plt.hist(y)
con.strength.value_counts()

from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x.shape
y.shape
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(8,8))

mlp.fit(x_train,y_train)
prediction_train = mlp.predict(x_train)
prediction_test = mlp.predict(x_test)

from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,prediction_test))
np.mean(y_test==prediction_test)  # 0.07936
np.mean(y_train==prediction_train)  #  0.10358565


