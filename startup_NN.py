import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

startup = pd.read_csv("C:/Users/ADMIN/Desktop/Data_Science_Assig/Neural Network/50_Startups (1).csv")
startup.columns
startup.shape
startup.head()
startup.describe()


from sklearn import preprocessing
le = preprocessing.LabelEncoder()
startup['State']=le.fit_transform(startup['State'])
startup
startup['State'].unique()

#startup.drop(["State"],axis=1, inplace=True)
startup.columns
startup.head()
startup.describe()
#null_columns=startup.columns[startup.isnull().any()]
#startup[null_columns].isnull().sum()
#startup.isnull()
startup.isnull().sum()

x = startup.drop(["Profit"],axis=1)
y = startup["Profit"]
y=y.astype('int')
x=x.astype('int')
plt.hist(y)
plt.hist(x)
startup.Profit.value_counts()

plt.hist(startup['Spend_rd']);plt.xlabel('Spend_rd');plt.ylabel('y');plt.title('histogram of Spend_rd')
plt.hist(startup['Administration']);plt.xlabel('Administration');plt.ylabel('y');plt.title('histogram of Administration')
plt.hist(startup['Marketing_Spend']);plt.xlabel('Marketing_Spend');plt.ylabel('y');plt.title('histogram of Marketing_Spend')
plt.hist(startup['State']);plt.xlabel('State');plt.ylabel('y');plt.title('histogram of State')

sns.boxplot(startup.Spend_rd)
sns.boxplot(startup.Administration)
sns.boxplot(startup.Marketing_Spend)
sns.boxplot(startup.Profit)

sns.pairplot(startup)

# Normal Q- plot
plt.plot(startup);plt.legend(list(startup.columns))
Spend_rd= np.array(startup['Spend_rd'])
Administration = np.array(startup['Administration'])
Marketing_Spend = np.array(startup['Marketing_Spend'])
State = np.array(startup['State'])
Profit = np.array(startup['Profit'])

from scipy import stats
stats.probplot(Spend_rd, dist='norm', plot=plt);plt.title('Probability Plot of Spend_rd')
stats.probplot(Administration, dist='norm', plot=plt);plt.title('Probability Plot of Administration')
stats.probplot(Marketing_Spend, dist='norm', plot=plt);plt.title('Probability Plot of Marketing_Spend')
stats.probplot(State, dist='norm', plot=plt);plt.title('Probability Plot of State')
stats.probplot(y, dist='norm', plot=plt);plt.title('Probability Plot of Profit')

from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(3,3))

mlp.fit(x_train,y_train)
prediction_train=mlp.predict(x_train)
prediction_test=mlp.predict(x_test)
prediction_train
prediction_train.mean()  #  103807
prediction_test
prediction_test.mean()   # 105608

from sklearn.metrics import classification_report,confusion_matrix

print(confusion_matrix(y_test,prediction_test))
np.mean(y_test==prediction_test)
np.mean(y_train==prediction_train)
prediction_test
prediction_test.mean()  # 105608
prediction_train
prediction_train.mean() # 103807
