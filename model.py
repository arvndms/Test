import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import xlrd

iris_data=pd.read_excel("iris.xls")
x = iris_data.drop('Classification',axis = 1)
le = LabelEncoder()
y = pd.Series(le.fit_transform(iris_data['Classification']))
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


regressor=LogisticRegression()
regressor=regressor.fit(x_train,y_train)

pickle.dump(regressor,open('model.pkl','wb'))
