import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

wine_quality = pd.read_csv('winequality-red.csv', sep = ';')
#print(wine_quality.head())
wine_quality.rename(columns= lambda x: x.replace(' ','_'),inplace = True)
#print(wine_quality.head())
x_train, x_test, y_train, y_test = train_test_split(wine_quality['alcohol'],wine_quality["quality"], random_state= 42)

x_train = pd.DataFrame(x_train); x_test = pd.DataFrame(x_test)
y_train = pd.DataFrame(y_train); y_test = pd.DataFrame(y_test)
#print(x_train.head())

#def mean(values):
#    return round(sum(values)/float(len(values)),2)

alcohol_mean = x_train['alcohol'].mean()
quality_mean = y_train['quality'].mean()

alcohol_variance = x_train['alcohol'].var()
quality_variance = y_train['quality'].var()
quality_variance_test = y_test['quality'].var()
### calculate covariance

covariance_all = np.cov(x_train['alcohol'],y_train['quality'])
#print(covariance_all )
covariance= covariance_all[0][1]
#covariance = sum()
#covariance = round(sum((x_train['alcohol'] - alcohol_mean) * (y_train['quality'] - quality_mean )),2)
print(covariance )
b1 = covariance/alcohol_variance
b0= quality_mean - b1* alcohol_mean
print(b0,b1)
y_test['y_pred'] = pd.DataFrame(x_test['alcohol'] *b1 + b0)
R_square = 1- sum((y_test['y_pred']-y_test['quality'])**2 )/sum((y_test['quality']- y_test['quality'].mean())**2)
print(R_square)
print('For R_square < 0.7, no obvious relation between two relation')