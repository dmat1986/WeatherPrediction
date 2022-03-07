import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#Load the data into a pandas Dataframe from csv
dataset = 'weatherAUS.csv'
rain = pd.read_csv(dataset)

#Decrease the high cardinality of the Date column by dropping the Date
#column and adding 'year,' 'month,' and 'day' separately.
rain['Date'] = pd.to_datetime(rain['Date'])
rain['year'] = rain['Date'].dt.year
rain['month'] = rain['Date'].dt.month
rain['day'] = rain['Date'].dt.day

rain.drop('Date', axis = 1, inplace = True)

#Extracting the categorical and numerical features of the dataset
categorical_features = [column_name for column_name in rain.columns if rain[column_name].dtype == 'O']
numerical_features = [column_name for column_name in rain.columns if rain[column_name].dtype != 'O']

#For missing values in the categorical features, we replace the missing
#values with the most common value
categorical_features_with_null = [feature for feature in categorical_features if rain[feature].isnull().sum()]
for each_feature in categorical_features_with_null:
	mode_val = rain[each_feature].mode()[0]
	rain[each_feature].fillna(mode_val,inplace=True)

#For missing values in the numerical features, we replace the missing
#values with the median
numerical_features_with_null = [feature for feature in numerical_features if rain[feature].isnull().sum()]
for each_feature in numerical_features_with_null:
	median_val = rain[each_feature].median()
	rain[each_feature].fillna(median_val,inplace=True)

def encode_data(feature_name):
	#This function takes feature name as a parameter and returns 
	#mapping dictionary to replace/map categorical data with 
	#numerical data.
	
	mapping_dict = {}

	unique_values = list(rain[feature_name].unique())

	for i in range(len(unique_values)):
		mapping_dict[unique_values[i]] = i

	return mapping_dict

rain['RainToday'].replace({'No':0,'Yes':1},inplace=True)
rain['RainTomorrow'].replace({'No':0,'Yes':1},inplace=True)
rain['WindGustDir'].replace(encode_data('WindGustDir'),inplace=True)
rain['WindDir9am'].replace(encode_data('WindDir9am'),inplace=True)
rain['WindDir3pm'].replace(encode_data('WindDir3pm'),inplace=True)
rain['Location'].replace(encode_data('Location'),inplace=True)

# plt.figure(figsize=(20,20))
# sns.heatmap(rain.corr(),linewidths=0.5,annot=False,fmt='.2f',cmap='viridis')

# plt.show()

#Split data into independent and dependent features
X = rain.drop(['RainTomorrow'],axis=1)
y = rain['RainTomorrow']

#In order to determine which features are most relevant, we use
#the ExtraTreesRegressor class for Feature Importance
etr_model = ExtraTreesRegressor()
etr_model.fit(X,y)
etr_model.feature_importances_

feature_imp = pd.Series(etr_model.feature_importances_,index=X.columns)
#feature_imp.nlargest(10).plot(kind='barh')

#Splitting data into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)

#Normalizing the data within a range from 0 to 1
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Using Logistic Regression for the prediction model
classifier_logreg = LogisticRegression(solver='liblinear',random_state=0)
classifier_logreg.fit(X_train,y_train)

y_pred = classifier_logreg.predict(X_test)

#Print the results. 
#0 = no rain, 1 = rain
print(y_pred)

################################
#Print model performance data
################################

#The accuracy score calculates the accuracy of a model prediction 
#on unseen data
print("Accuracy score: {}".format(accuracy_score(y_test,y_pred)))

#Here, we check for overfitting or underfitting
print("Train data score: {}".format(classifier_logreg.score(X_train,y_train)))
print("Test data score: {}".format(classifier_logreg.score(X_test,y_test)))