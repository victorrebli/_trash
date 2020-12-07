


X_train, X_test, y_train, y_test = train_test_split(data_new, label, test_size=0.3, random_state=42)

clf = GradientBoostingClassifier(random_state=42, n_estimators=100)
clf.fit(X_train[COLS].fillna(-999), y_train)

proba_predicted = clf.predict_proba(X_test[COLS].fillna(-999))[:,1]
expected = y_test
threshold = utils.best_threshold(expected, proba_predicted)
row_proba_predicted = proba_predicted.reshape(1, -1)
predicted = binarize(row_proba_predicted, threshold).reshape(-1, 1)


str_rep = metrics.classification_report(expected, predicted, digits=4, output_dict=True)
print(str_rep)

utils.plot_confusion_matrix(metrics.confusion_matrix(expected, predicted), target_names=['0','1'], normalize=True)


utils.compute_roc(expected, proba_predicted, plot=True)

utils.feature_importance(clf, X_train[COLS].keys())


### Treina o modelo e retira as regras para cada árvore do boosting

dici_final = dict()
dici_final_regras = dict()

clf = GradientBoostingClassifier(random_state=42, n_estimators=100)
clf.fit(X_train[COLS].fillna(-999), y_train)

prval = clf.predict_proba(X_test[COLS].fillna(-999))[:,1]
fpr, tpr, _ = metrics.roc_curve(y_test, prval)
print(f'AUC - GradientBoosting: {metrics.auc(fpr,tpr)}')

try:
    dici_final = {}
    for idx_c, clf_ in enumerate(clf.estimators_):
        print(f'Tree:{idx_c}')
        if 'GradientBoostingClassifier' not in str(clf.__class__):
            dic = utils.find_rules(clf_, X_test[COLS].fillna(-999).reset_index(drop=True), COLS)
        else:
            dic = utils.find_rules(clf_[0], X_test[COLS].fillna(-999).reset_index(drop=True), COLS)

        idu = []
        for idx, sample_id in enumerate(X_test.reset_index(drop=True).index):

            if idx % 1000 == 0:
                print(f'sample ID: {idx}')

            if 'GradientBoostingClassifier' not in str(clf.__class__):
                idu.append(utils.find_leave(clf_, X_test[COLS].fillna(-999).values, sample_id))

            else:
                idu.append(utils.find_leave(clf_[0], X_test[COLS].fillna(-999).values, sample_id))            


        dici_final[idx_c] = (dic, pd.value_counts(idu))

except:
    print('Continue.....')


### Para cada árvore, obtêm a regra com maior cobertura, e verifica a acurácia (quantidade de uns certos)


dici_summary = dict()
for tree in list(dici_final.keys()):
    try:
        dici_summary[tree] = [dici_final[tree][0][dici_final[tree][1].keys()[0]],X_test[eval(dici_final[tree][0][dici_final[tree][1].keys()[0]])]['label'].value_counts(normalize=True)]
    except:
        dici_summary[tree] = ['nenhuma regra']


### Cria o DataFrame

results = []
for s_tree in dici_summary.keys():
    try:
        row = OrderedDict(set=dici_summary[s_tree][0], accu=dici_summary[s_tree][1][0])
        results.append(row)
    except:
        row = OrderedDict(set='nenhuma regra', accu=0.0)
        results.append(row)    


final_results = pd.DataFrame(results)

# Ordena

final_results = final_results.sort_values('accu', ascending=False)

for row in np.arange(0,10):
    print(f'rule: {final_results.iloc[row, 0]} -- accuracy: {final_results.iloc[row, 1]}')
    print('######################')


dat_interm = data_new.copy()
indices = {}
for row in final_results.iterrows():
    dat_interm_1 = dat_interm[eval(row[1].set.replace('X_test', 'data_new'))]
    indices[row[0]] = dat_interm_1.copy()
    dat_interm = dat_interm.loc[~dat_interm.index.isin(dat_interm_1.index)]
    if dat_interm.shape[0] == 0:
        print('break')
        break 


results_cobertura = []
for rule in indices.keys():
    try:
        row = OrderedDict(set=rule, cobertura=indices[rule].label.value_counts()[0] / data_new.label.value_counts()[0],
                         abs_value=indices[rule].label.value_counts()[0])
        results_cobertura.append(row)
    except:
        row = OrderedDict(set='nenhuma regra', cobertura=0.0)
        results_cobertura.append(row)   
        
final_results_cobertura = pd.DataFrame(results_cobertura)  


final_results_cobertura['cobertura_cumsum'] = final_results_cobertura.cobertura.cumsum()


regras = final_results_cobertura[(final_results_cobertura.cobertura_cumsum <= 0.95) & (final_results_cobertura.set != 'nenhuma regra')]


for rule in regras.set.values:
    print(f'{rule} - {final_results.loc[rule].set}')