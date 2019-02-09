import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import statsmodels.api as sm
import math

data = pd.read_csv('master.csv')
dummies = pd.get_dummies(data.sex)
data = data.join(dummies)
dummies = pd.get_dummies(data.age)
data = data.join(dummies)
data.dropna()
data = data.drop(columns=['country','suicides/100k pop','country-year','HDI for year','generation',' gdp_for_year ($) ','sex','age'])
data = data[['year','suicides_no','population','gdp_per_capita ($)','female', 'male', '5-14 years', '15-24 years',  '25-34 years' , '35-54 years',    '55-74 years',  '75+ years']]
data_size = len(data.index) #27820
data_size_fifth = math.ceil(data_size/5) 
error = []




#print(data)
#print(data.loc[0]['year'])
for i in range(5):
    test=data.iloc[0+i*data_size_fifth:data_size_fifth+i*data_size_fifth]
    train=data.drop(data.index[0+i*data_size_fifth:data_size_fifth+i*data_size_fifth])   
    X = train[['year','population','gdp_per_capita ($)','female', 'male', '5-14 years', '15-24 years',  '25-34 years' , '35-54 years',    '55-74 years',  '75+ years']] 
    Y = train['suicides_no']
    regr = linear_model.LinearRegression()
    regr.fit(X, Y)
    mean_difference=0
    for i in test.index:
        predicted_value = regr.predict([   [data.loc[i]['year'],
                                            data.loc[i]['population'],
                                            data.loc[i]['gdp_per_capita ($)'],
                                            data.loc[i]['female'],
                                            data.loc[i]['male'],
                                            data.loc[i]['5-14 years'],
                                            data.loc[i]['15-24 years'],
                                            data.loc[i]['25-34 years'],
                                            data.loc[i]['35-54 years'],
                                            data.loc[i]['55-74 years'],
                                            data.loc[i]['75+ years']]])
        print(i)
        real_value = data.loc[i]['suicides_no']
        if(i == 0):
            mean_difference += abs((real_value/predicted_value)*100)
        else:
            mean_difference = (mean_difference + abs((real_value/predicted_value)*100))/2
    error.append(mean_difference)
print(error)


#print('Intercept: \n', regr.intercept_)
#print('Coefficients: \n', regr.coef_)
#print ('Predicted suicide number: \n', regr.predict([[2019,40000000,13811,0,1,0,1,0,0,0,0]]))
