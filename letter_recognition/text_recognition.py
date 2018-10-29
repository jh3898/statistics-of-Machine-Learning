import os
'''
First change the following directory link to where all input files do exist
'''
#os.chdir('iCloud Drive\Desktop\coding\python\statistics_ML\letter_recognition')
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
letterdata = pd.read_csv('letter-recognition.data.txt')

letterdata.columns = ['letter','xbox','ybox','width','height','onpix','xbar','ybar','x2bar','y2bar','xybar',
                      'x2ybar','xy2bar','xedge','xedgey','yedge','yedgex']
#letterdata.index = letterdata['letter']
x_vars= letterdata.drop(['letter'],axis =1)
y_var = letterdata['letter']
#print(x_vars.head())
#print(y_var.head())
y_var = y_var.replace({'A':1 ,'B':2,'C':3, 'D':4,'E':5, 'F':6, 'G':7,'H':8,'I':9, 'J':10,'K':11, 'L':12,
                       'M':13,'N':14,'O':15, 'P':16, 'Q':17,'R':18,'S':19,'T':20,'U':21, 'V':22, 'W':23,
                       'X':24,'Y':25,'Z':26})
#y_var = y_var.replace({'A':'1' ,'B':'2','C':'3', 'D':'4','E':'5', 'F':'6', 'G':'7','H':'8','I':'9', 'J':'10','K':'11', 'L':'12',
                       #'M':'13','N':'14','O':'15', 'P':'16', 'Q':'17','R':'18','S':'19','T':'20','U':'21', 'V':'22', 'W':'23',
                       #'X':'24','Y':'25','Z':'26'})
print(y_var.head())
#y_var.astype('int32')
x_train, x_test, y_train, y_test = train_test_split(x_vars, y_var,  random_state= 42)

svm_fit = SVC(kernel= 'linear',C =1.0, random_state= 43)
svm_fit.fit(x_train, y_train)

print('\nSVM linear Classifier -Train Confusion Matrix\n\n', pd.crosstab(y_train, svm_fit.predict(x_train), rownames=['Actuall'],
                                                                         colnames= ['Predicted']))
print('\nSVM Linear Classifier- Train accuracy:', round(accuracy_score(y_train,svm_fit.predict(x_train)),3))
print('\nSVM Linear Classifier- Train Classification Report\n', classification_report(y_train, svm_fit.predict(x_train)))

print('\nSVM linear Classifier -Test Confusion Matrix\n\n', pd.crosstab(y_test, svm_fit.predict(x_test), rownames=['Actuall'],
                                                                         colnames= ['Predicted']))
print('\nSVM Linear Classifier- Test accuracy:', round(accuracy_score(y_test,svm_fit.predict(x_test)),3))
print('\nSVM Linear Classifier- Test Classification Report\n', classification_report(y_test, svm_fit.predict(x_test)))

# Grid Search -RBF Kernel
pipeline = Pipeline([('clf', SVC(kernel='rbf', C= 1, gamma= 0.1))])
parameters = {'clf_C':(0.1, 0.3, 1,3, 10, 30), 'clf_gamma':(0.001, 0.01, 0.1, 0.3, 1)}

grid_search_in = GridSearchCV(pipeline, parameters, n_jobs= -1, cv = 5, verbose= 1, scoring ='accuracy')
grid_search_in.fit(x_train, y_train)
print(grid_search_in.best_score_)
best_parameters= grid_search_in.best_estimator_.get_params()
#print(grid_search_in.best_estimator_.get_params())
for param_name in sorted(parameters.keys()):
    print('\t%s: %r' % (param_name, best_parameters[param_name]))