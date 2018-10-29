# KNN Classifier
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
breast_cancer = pd.read_csv('breast-cancer-wisconsin.data.txt')

breast_cancer.columns = ['ID_Number','Clump_thickness','Unit_cell_Size','Unit_cell_shape','Marg_Adhesion','Single_Epith_cell_size','Bare_Nuclei','Bland_Chromatin',
                         'Normal_Nucleoli','Mitose','Class']
#print(breast_cancer)
breast_cancer['Bare_Nuclei'] = breast_cancer['Bare_Nuclei'].replace('?',np.NaN)
breast_cancer['Bare_Nuclei'].fillna(breast_cancer['Bare_Nuclei'].value_counts().index[0],inplace = True)
#breast_cancer[:].replace('?',np.NaN)
#print(breast_cancer)
#breast_cancer.dropna()
breast_cancer.fillna(breast_cancer.mean(), inplace= True)
#breast_cancer[:].fillna(breast_cancer[:].value_counts().index[0])
breast_cancer['Cancer_Ind'] = 0
breast_cancer.loc[breast_cancer['Class']==4, 'Cancer_Ind'] = 1
x_vars = breast_cancer.drop(['ID_Number','Class','Cancer_Ind'], axis= 1)
y_vars = breast_cancer['Cancer_Ind']
from sklearn.preprocessing import StandardScaler
x_vars_stdscle = StandardScaler().fit_transform(x_vars.values)
from sklearn.model_selection import train_test_split

x_vars_stdscle_df = pd.DataFrame(x_vars_stdscle, index = x_vars.index, columns= x_vars.columns)
x_train, x_test, y_train, y_test = train_test_split(x_vars_stdscle, y_vars, train_size = 0.7, random_state= 42)

from sklearn.neighbors import KNeighborsClassifier
knn_fit = KNeighborsClassifier(n_neighbors=3, p=2, metric= 'minkowski')
knn_fit.fit(x_train, y_train)
print('\nK-Nearest Neighbors - Train COnfusion Matrxi\n\n',pd.crosstab(y_train,knn_fit.predict(x_train),rownames= ['Acutuall'],
     colnames =['Predicted']))
print('\nK-Nearest Neighbors - Train accuracy:', round(accuracy_score(y_train, knn_fit.predict(x_train)),3))
print('\nK-Nearest Neighbors - Train Classification Report\n', classification_report(y_train,knn_fit.predict(x_train)))

print('\n\nK-Nearest Neighbors - Test Co')

print('\nK-Nearest Nieghbor - Test Confusion Matrix\n\n', pd.crosstab(y_test,knn_fit.predict(x_test),rownames= ['Acutual'],
      colnames= ['Predicted']))
print('\nK-Nearest Neighbor - Train accuracy:', round(accuracy_score(y_test,knn_fit.predict(x_test)),3))
print('\nK-Nearest Neighbor\n', classification_report(y_test,knn_fit.predict(x_test)))

# Tuning of K- value for Train & Test data
dummyarray = np.empty((10,3))
k_valchart = pd.DataFrame(dummyarray)
k_valchart.columns = ['K_value','Train_Acc','Test_ACC']
k_vals = [1,2,3,4,5,6,7,8,9,10]

for i in range(len(k_vals)):
    knn_fit = KNeighborsClassifier(n_neighbors= k_vals[i],p=2, metric= 'minkowski')
    knn_fit.fit(x_train,y_train)

    print('\nK-value',k_vals[i])
    tr_accscore = round(accuracy_score(y_train, knn_fit.predict(x_train)),3)
    print('\nK-Nearest Neighbor- Train confusion Matrix\n\n',pd.crosstab(y_train, knn_fit.predict(x_train),rownames= ['actuall'],
                                                                         colnames= ['predicted']))
    print('\nK-Nearest Neighbor - Train accuracy:',tr_accscore )
    print('\nK-Nearest- Neighbor\n', classification_report(y_train, knn_fit.predict(x_train)))

    ts_accscore = round(accuracy_score(y_test, knn_fit.predict(x_test)), 3)
    print('\nK-Nearest Neighbor- Test confusion Matrix\n\n',
          pd.crosstab(y_test, knn_fit.predict(x_test), rownames=['actuall'],
                      colnames=['predicted']))
    print('\nK-Nearest Neighbor - Test accuracy:', tr_accscore)
    print('\nK-Nearest- Neighbor - Test Classfication report\n', classification_report(y_test, knn_fit.predict(x_test)))
    k_valchart.loc[i,'K_value'] = k_vals[i]
    k_valchart.loc[i,'Train_acc'] = tr_accscore
    k_valchart.loc[i, 'Test_acc'] = ts_accscore

import matplotlib.pyplot as plt

plt.plot(k_valchart['K_value'], k_valchart['Train_acc'])
plt.plot(k_valchart['K_value'], k_valchart['Test_acc'])

plt.axis([0.9,5, 0.92, 1.005])
plt.xticks([1,2,3,4,5,6,7,8,9,10])

for a,b in zip(k_valchart['K_value'],k_valchart['Train_acc']):
    plt.text(a,b, str(b), fontsize = 10)
for a,b in zip(k_valchart['K_value'],k_valchart['Test_acc']):
    plt.text(a,b, str(b), fontsize = 10)

plt.legend(loc = 'upper right')

plt.show()