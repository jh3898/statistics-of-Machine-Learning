import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

wine_quality = pd.read_csv('winequality-red.csv', sep = ';')
print(wine_quality.head())
wine_quality.rename(columns= lambda x: x.replace(' ','_'),inplace = True)
#print(wine_quality.head())

eda_colmns = ['volatile_acidity', 'chlorides','sulphates','alcohol','quality']

## plot - pair plots
sns.set(style= 'whitegrid', context = 'notebook')
#sns.pairplot(wine_quality[eda_colmns],size =1, x_vars = eda_colmns, y_vars = eda_colmns)
#plt.show()

# Correlation coefficients
corr_mat = np.corrcoef(wine_quality[eda_colmns].values.T)
#sns.set(font_scale = 1)
#full_mat = sns.heatmap(corr_mat,cbar= True, annot= True, square= True, fmt = '.2f', annot_kws={'size': 15},yticklabels = eda_colmns,
#                       xticklabels= eda_colmns)
#plt.show()

colmns = ['fixed_acidity', 'volatile_acidity','citric_acid', 'residual_sugar','chlorides',
          'free_sulfur_dioxide',  'total_sulfur_dioxide', 'density', 'pH',  'sulphates', 'alcohol']

pdx = wine_quality[colmns]
pdy = wine_quality['quality']

x_train, x_test, y_train, y_test = train_test_split(pdx,pdy, random_state= 42)
x_train_new = sm.add_constant(x_train)
x_test_new = sm.add_constant(x_test)
full_mod = sm.OLS(y_train, x_train_new)
full_res = full_mod.fit()
#print('\n \n ', full_res.summary())

print('\nVariance Inflation Factor')
cnames = x_train.columns

for i in np.arange(0,len(cnames)):
    xvars = list(cnames)
    yvar = xvars.pop(i)
    mod = sm.OLS(x_train[yvar], sm.add_constant(x_train_new[xvars]))
    res = mod.fit()
    vif = 1/(1- res.rsquared)
    print(yvar, round(vif, 3))
















#x_train, x_test, y_train, y_test = train_test_split(wine_quality['alcohol'],wine_quality["quality"], random_state= 42)

#x_train = pd.DataFrame(x_train); x_test = pd.DataFrame(x_test)
#y_train = pd.DataFrame(y_train); y_test = pd.DataFrame(y_test)
#print(x_train.head())

