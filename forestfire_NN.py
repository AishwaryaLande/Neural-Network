import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

forest = pd.read_csv("C:/Users/ADMIN/Desktop/Data_Science_Assig/Neural Network/forestfires (1).csv")
forest.shape
forest.columns
forest.describe()
forest.head()
forest.drop_duplicates(keep='first',inplace=True)
forest.isnull()
forest.isnull().sum()

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
forest['size_category']=le.fit_transform(forest['size_category'])
forest
forest['size_category'].unique()

forest.drop(["month","day","dayfri","daymon","daysat","daysun","daythu","daytue","daywed","monthapr","monthaug","monthdec","monthfeb","monthjan","monthjul","monthjun","monthmar","monthmay","monthnov","monthoct","monthsep"],axis=1,inplace=True) # Dropping the uncessary column
forest.size_category.value_counts()

forest.columns
forest.head()
forest.describe()
np.var(forest)
np.std(forest)
# Skewness and Kurtosis

#string_col = ["month","day","size_category"]
#startup.drop(["State"],axis=1, inplace=True)
#for i in string_col:
 #   forest[i] = le.fit_transform(forest[i])
#colunames = forest.columns
#len(colunames[0:31])
#forest = forest[colunames[0:31]]
#forest.drop_duplicates(keep='first', inplace=True) # 25 duplicate rows there

plt.hist(forest['FFMC']);plt.xlabel('FFMC');plt.ylabel('y');plt.title('histogram of FFMC')
plt.hist(forest['DMC']);plt.xlabel('DMC');plt.ylabel('y');plt.title('histogram of DMC')
plt.hist(forest['DC']);plt.xlabel('DC');plt.ylabel('y');plt.title('histogram of DC')
plt.hist(forest['ISI']);plt.xlabel('ISI');plt.ylabel('y');plt.title('histogram of ISI')
plt.hist(forest['RH']);plt.xlabel('RH');plt.ylabel('y');plt.title('histogram of RH')
plt.hist(forest['area']);plt.xlabel('area');plt.ylabel('y');plt.title('histogram of area')
plt.hist(forest['temp']);plt.xlabel('temp');plt.ylabel('y');plt.title('histogram of temp')
plt.hist(forest['rain']);plt.xlabel('rain');plt.ylabel('y');plt.title('histogram of rain')
plt.hist(forest['wind']);plt.xlabel('wind');plt.ylabel('y');plt.title('histogram of wind')

from scipy.stats import skew, kurtosis
skew(forest)
kurtosis(forest)

sns.boxplot(forest.size_category)
sns.boxplot(forest.temp)
sns.boxplot(forest.rain)
sns.boxplot(forest.wind)
sns.boxplot(forest.FFMC)
sns.boxplot(forest.DMC)
sns.boxplot(forest.DC)
sns.boxplot(forest.ISI)
sns.boxplot(forest.RH)
sns.boxplot(forest.area)

sns.pairplot(forest)
#Q-plot
plt.plot(forest);plt.legend(list(forest.columns))
FFMC= np.array(forest['FFMC'])
DMC = np.array(forest['DMC'])
DC = np.array(forest['DC'])
ISI = np.array(forest['ISI'])
temp = np.array(forest['temp'])
RH= np.array(forest['RH'])
wind = np.array(forest['wind'])
rain = np.array(forest['rain'])
area = np.array(forest['area'])
size_category = np.array(forest['size_category'])

from scipy import stats
stats.probplot(FFMC, dist='norm', plot=plt);plt.title('Probability Plot of FFMC')
stats.probplot(DMC, dist='norm', plot=plt);plt.title('Probability Plot of DMC')
stats.probplot(DC, dist='norm', plot=plt);plt.title('Probability Plot of DC')
stats.probplot(ISI, dist='norm', plot=plt);plt.title('Probability Plot of ISI')
stats.probplot(temp, dist='norm', plot=plt);plt.title('Probability Plot of temp')
stats.probplot(RH, dist='norm', plot=plt);plt.title('Probability Plot of RH')
stats.probplot(wind, dist='norm', plot=plt);plt.title('Probability Plot of wind')
stats.probplot(rain, dist='norm', plot=plt);plt.title('Probability Plot of rain')
stats.probplot(area, dist='norm', plot=plt);plt.title('Probability Plot of area')
stats.probplot(size_category, dist='norm', plot=plt);plt.title('Probability Plot of size_category')

corr = forest.corr()
corr
sns.heatmap(corr, annot=True)


x = forest.drop(["size_category"],axis=1)
y = forest["size_category"]
x=x.astype('int')
y=y.astype('int')
plt.hist(y)
forest.size_category.value_counts()

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

mlp = MLPClassifier(hidden_layer_sizes=(9,9))

mlp.fit(x_train,y_train)
prediction_train = mlp.predict(x_train)
prediction_test = mlp.predict(x_test)

from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,prediction_test))
np.mean(y_test==prediction_test)  # 0.828125
np.mean(y_train==prediction_train)  #  821522309711286


