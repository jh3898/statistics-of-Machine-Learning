import os
#os.chdir()

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

credit_data = pd.read_csv('german.data.txt', sep= ' ', header= None)
credit_data.columns = ['Status of existing checking account','Duration in month','Credit history','Purpose','Credit amount','Saving account',
                        'Present employment since','Installment rate in percentage of disposable income','Personal status and sex','Other debtors',
                        'Present residence since','Property','Age in years','Other installment plans','Housing','Number of existing credits at this bank',
                       'Job','Number of people being liable to provide maintenance for','Telephone','foreign worker','class']
#print(credit_data.head())
credit_data.rename(columns= lambda x : x.replace(' ','_'),inplace = True)
#print(credit_data.head())
credit_data['class'] = credit_data['class'] -1

def IV_calc(data, var):
    if data[var].dtype == 'object':
        dataf = data.groupby([var])['class'].agg(['count','sum'])
        dataf.columns = ['Total','bad']
        dataf['good'] = dataf['Total']-dataf['bad']
        dataf['bad_per'] = dataf['bad']/dataf['bad'].sum()
        dataf['good_per'] = dataf['good'] / dataf['good'].sum()
        dataf['I_V'] = (dataf['good_per'] - dataf['bad_per'])*np.log(dataf['good_per']/dataf['bad_per'])
        return dataf
    else:
        data['bin_var'] = pd.qcut(data[var].rank(method = 'first'), 10)
        dataf = data.groupby(['bin_var'])['class'].agg(['count', 'sum'])
        dataf.columns = ['Total', 'bad']
        dataf['good'] = dataf['Total'] - dataf['bad']
        dataf['bad_per'] = dataf['bad'] / dataf['bad'].sum()
        dataf['good_per'] = dataf['good'] / dataf['good'].sum()
        dataf['I_V'] = (dataf['good_per'] - dataf['bad_per']) * np.log(dataf['good_per'] / dataf['bad_per'])
        return dataf

print('\n\nCredit History - Information Value\n')
print(IV_calc(credit_data,'Credit_history'))
print ("\n\nCredit History - Duration in month\n")
print (IV_calc(credit_data,'Duration_in_month'))

discrete_columns = ['Status_of_existing_checking_account','Credit_history','Purpose','Saving_account','Present_employment_since',
                    'Personal_status_and_sex','Other_debtors','Property','Other_installment_plans','Housing','Job','Telephone','foreign_worker']
continous_columns = ['Duration_in_month','Credit_amount','Installment_rate_in_percentage_of_disposable_income','Present_residence_since',
                     'Age_in_years','Number_of_existing_credits_at_this_bank','Number_of_people_being_liable_to_provide_maintenance_for']
total_columns = discrete_columns + continous_columns
##### List of IV values
Iv_list = []
for col in total_columns:
    assigned_data = IV_calc(data= credit_data, var = col)
    iv_val = round(assigned_data['I_V'].sum(),3)
    dt_type = credit_data[col].dtypes
    Iv_list.append((iv_val,col,dt_type))

Iv_list = sorted(Iv_list, reverse= True)

for i in range(len(Iv_list)):
    print(Iv_list[i][0],',',Iv_list[i][1],',', 'type =', Iv_list[i][2])

dummy_stseca = pd.get_dummies(credit_data['Status_of_existing_checking_account'],prefix= 'status_exs_accnt')
dummy_ch = pd.get_dummies(credit_data['Credit_history'],prefix= 'cred_hist')
dummy_purpose = pd.get_dummies(credit_data['Purpose'],prefix = 'purpose')
dummy_savacc = pd.get_dummies(credit_data['Saving_account'],prefix= 'sav_acc')
dummy_presc = pd.get_dummies(credit_data['Present_employment_since'],prefix= 'pre_emp_snc')
credit_data_new = pd.concat([dummy_stseca,dummy_ch, dummy_purpose, dummy_savacc, dummy_presc ],axis =1 )
x_train, x_test, y_train, y_test = train_test_split(credit_data_new,credit_data['class'],train_size=0.7,random_state= 42)
y_train = pd.DataFrame(y_train)
y_test = pd.DataFrame(y_test)
remove_cols_extra_dummy = ['status_exs_accnt_A11','cred_hist_A30',
   'purpose_A40', 'sav_acc_A61','pre_emp_snc_A71']
remove_cols_insig = []
remove_cols = list(set(remove_cols_extra_dummy+ remove_cols_insig))

import statsmodels.api as sm
logistic_model = sm.Logit(y_train,sm.add_constant(x_train.drop(remove_cols, axis= 1))).fit()
print(logistic_model.summary())
print('\nVariance Inflation Factor')
cnames = x_train.drop(remove_cols, axis =1).columns
#print(cnames)
for i in np.arange(0,len(cnames)):
    xvars= list(cnames)
    yvar= xvars.pop(i)
    mod = sm.OLS(x_train.drop(remove_cols,axis =1)[yvar],sm.add_constant(x_train.drop(remove_cols,axis=1)[xvars]))
    res = mod.fit()
    vif = 1/(1- res.rsquared)
    print(yvar, round(vif,3))



#dummy_savacc =
#print(dummy_stseca)
#print(credit_data['Status_of_existing_checking_account'])
y_pred = pd.DataFrame(logistic_model.predict(sm.add_constant(x_train.drop(remove_cols,axis=1))))
y_pred.columns = ['probs']
#print(y_train)
both = pd.concat([y_train,y_pred],axis =1)
zeros = both[['class','probs']][both['class']==0]
ones = both[['class','probs']][both['class']==1]
#print(zeros)

def df_crossjoin(df1,df2,**kwargs):
    df1['_tempkey'] =1
    df2['_tempkey'] =1
    res = pd.merge(df1, df2, on = '_tempkey',**kwargs).drop('_tempkey',axis =1)
    res.index = pd.MultiIndex.from_product((df1.index, df2.index))
    df1.drop('_tempkey',axis=1, inplace = True)
    df2.drop('_tempkey', axis=1, inplace=True)
    return res

join_data = df_crossjoin(ones,zeros)
print(join_data)
join_data['concordant_pair'] =0
join_data.loc[join_data['probs_x']> join_data['probs_y'], 'concordant_pair'] =1

import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import auc
fpr, tpr, thresholds = metrics.roc_curve(both['class'],both['probs'], pos_label = 1)
roc_auc = auc(fpr, tpr)
plt.figure()
lw =2
plt.plot(fpr, tpr, color = 'darkorange', lw = lw, label = 'ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0,1], [0,1], color = 'navy', linestyle = '--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive rate')
plt.ylabel('True Positive rate')
plt.title('Roc Curve - German Credit data')
plt.legend(loc= 'lower right')
plt.savefig('roc_curve.pdf')
print(both)

both['y_pred'] =0
both.loc[both['probs']> 0.5, 'y_pred'] =1
print('\nTrain COndusion Matrix\n\n', pd.crosstab(both['class'],both['y_pred'], rownames= ['Acutuall'],
colnames = ['Predicted']))
print('\nTrain Accuracy:', round(accuracy_score(both['class'],both['y_pred']),4))

y_pred_test = pd.DataFrame(logistic_model.predict(sm.add_constant(x_test.drop(remove_cols,axis =1))))
y_pred_test.columns = ['probs']
both_test = pd.concat([y_test, y_pred_test], axis= 1)
both_test['y_pred'] =0
both_test.loc[both_test['probs']> 0.5, 'y_pred'] =1
len(both_test['class'])
print('\nTest Confusion Matrix\n\n', pd.crosstab(both_test['class'], both_test['y_pred'],
 rownames= ['Acutuall'], colnames = ['Predicted']))
print('\nTests Accuracy:', round(accuracy_score(both_test['class'],both_test['y_pred'],6)))