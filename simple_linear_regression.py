from sklearn import linear_model
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#read csv file
df = pd.read_csv("FuelConsumption.csv")
#select some columns only
few_columns = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]

#plot the four columns in the shape of a histogram
viz = few_columns[['CYLINDERS','ENGINESIZE','CO2EMISSIONS','FUELCONSUMPTION_COMB']]
viz.hist()
plt.show()

#plot each of these features against the Emission, to see the linear relation
plt.scatter(few_columns.FUELCONSUMPTION_COMB, few_columns.CO2EMISSIONS,  color='blue')
plt.xlabel("FUELCONSUMPTION_COMB")
plt.ylabel("Emission")
plt.show()

plt.scatter(few_columns.ENGINESIZE, few_columns.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

plt.scatter(few_columns.CYLINDERS, few_columns.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

#create train and test datasets. 80% of the entire dataset will be used for training and 20% for testing
#We create a mask to select random rows
msk = np.random.rand(len(df)) < 0.8
train = few_columns[msk] #these are still dataframes
test = few_columns[~msk] #the rest

#simple regression model (linear fit)
regr = linear_model.LinearRegression() #model
train_x = np.asanyarray(train[['ENGINESIZE']]) #independent variable x
train_y = np.asanyarray(train[['CO2EMISSIONS']]) #dependent variable y
regr.fit(train_x, train_y) #fit the train data using the linear regression model

# The coefficients
print ('Coefficients/slope: ', regr.coef_[0][0])
print ('Intercept: ',regr.intercept_[0])

plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r') #plot fit
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

#evaluate the model with R-squared (R**2=1-relative square error). The best possible score is 1.0 and it can be negative
test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
predicted_y = regr.predict(test_x)

print("Mean absolute error: %.2f" % np.mean(np.absolute(predicted_y - test_y))) #print with 2 decimals
print("Residual sum of squares (MSE): %.2f" % np.mean((predicted_y - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y , predicted_y) )

