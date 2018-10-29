# Neural Networks - Classifying hand-written digits

import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
digits = load_digits()
#print(digits)
X= digits.data
y= digits.target

# checking dimensions
print( X.shape)
print(y.shape)

# plotting first digit
plt.matshow(digits.images[2])
# plt.show()

x_vars_stdscle = StandardScaler().fit_transform(X)
x_train, x_test, y_train, y_test = train_test_split(x_vars_stdscle, y, random_state= 42)

pipeline = Pipeline([('mlp', MLPClassifier(hidden_layer_sizes= (100,50,), activation= 'relu', solver = 'adam', alpha = 0.0001,
                                           max_iter = 300))])
parameters = {'mlp_alpha':(0.001, 0.01, 0.1, 0.3, 0.5, 1.0), 'mlp_max_iter':(100,200,300)}

grid_search_in = GridSearchCV(pipeline, parameters, n_jobs= -1, cv = 5, verbose= 1, scoring ='accuracy')
grid_search_in.fit(x_train, y_train)
print(grid_search_in.best_score_)
best_parameters= grid_search_in.best_estimator_.get_params()
#print(grid_search_in.best_estimator_.get_params())
for param_name in sorted(parameters.keys()):
    print('\t%s: %r' % (param_name, best_parameters[param_name]))