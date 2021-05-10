

## Modelos
from sklearn.linear_model import LogisticRegression 
from sklearn.svm import SVC
from sklearn import svm 
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
from sklearn.dummy import DummyClassifier #BASELINE

## Otras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## Validacion y entrenamiento
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report, roc_auc_score, accuracy_score
from sklearn import metrics
import scikitplot as skplt

import random
import warnings

warnings.filterwarnings('ignore')


"""
1era parte : Probar clasificadores (mas rapido) para clasificacion binaria
	- clasificadores_prueba
	- probar_clas
	- TEST_CLASIFICADOR
	- optimizar parametros
	- entrenamiento**
	- optimizar_punto_corte

2da parte : Probar Regresiones

"""

########### 1ERA PARTE: PRUEBA CLASIFICADOR

def clasificadores_prueba(random_state=123):
    ## DUMMY
    """
    """
    c0 = ('DUMMY', DummyClassifier(random_state=random_state), {'strategy':['most_frequent']}, {'strategy':['stratified', 'most_frequent', 'prior', 'uniform', 'constant']})

    ## LOGIT
    """
    grid={"C":np.logspace(-3,3,7), "penalty":["l1","l2"]}
    """
    c1 = ('LOGIT', LogisticRegression(random_state=random_state),{'C':np.logspace(-3,3,3), 'penalty':['l2']}, {"C":np.logspace(-3,3,7), "penalty":["l2"]})

    ## SVM
    """
    grid = {'C': [0.1, 1, 10, 100, 1000], 
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf', 'linear']} 
    """
    c2 = ('SVM', SVC(random_state=random_state), {'C':[0.1,1,1000]}, {'C': [0.1, 1, 10, 100, 1000], 
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf', 'linear', 'poly', 'sigmoid']})

    c2_1 = ('SVM', svm.LinearSVC(random_state=random_state), {}, {})

    ## ARBOL DE DECISION
    """
    grid = {'max_depth':[5,7,10,13], 'min_samples_leaf':[100, 1000, 5000, 10000], 'criterion':['entropy', 'gini']}
    """
    c3 = ('DECISION TREE', DecisionTreeClassifier(random_state=random_state), {'max_depth':[5,7,10], 'min_samples_leaf':[50,100,300], 'criterion':['entropy', 'gini']},{'max_depth':[5,7,10,13], 'min_samples_leaf':[100, 1000, 5000, 10000], 'criterion':['entropy', 'gini']})

    ## RANDOM FOREST
    """
    grid = {'max_depth':[5,7,10,13], 'n_estimators':[50,100,150,200,500], 'criterion':['gini','entropy', 'max_features':auto]}
    """
    c4 = ('RANDOM FOREST', RandomForestClassifier(random_state=random_state),{'max_depth':[5,7,10], 'n_estimators':[50,100,150], 'criterion':['gini','entropy'], 'max_features':['auto']}, {'max_depth':[5,7,10,13], 'n_estimators':[50,100,150,200,500], 'criterion':['gini','entropy'], 'max_features':['auto']})

    ## XGBOOST
    """
    grid = {'objective':['binary:logistic'], #objective='multi:softprob'
            'eval_metric': ['auc'],
            'learning_rate': [0.01,0.03,0.05], #so called `eta` value
            'max_depth': [5,7,10,13],
            'subsample': [0.5, 0.8, 0.9],
            'colsample_bytree': [0.5, 0.7, 0.9],
            'n_estimators': [5,10,15,30],
            'alpha':[10],
            'eta':[0.05]}
    """
    c5 = ('XGBOOST', xgb.XGBClassifier(random_state=random_state),{'objective':['binary:logistic'], #objective='multi:softprob'
            'eval_metric': ['auc'],
            'learning_rate': [0.01,0.05], #so called `eta` value
            'max_depth': [5,7,10],
            'subsample': [0.8, 0.9],
            'colsample_bytree': [0.5,0.9],
            'n_estimators': [10,30],
            'alpha':[10],
            'eta':[0.05]}
    , {'objective':['binary:logistic'], #objective='multi:softprob'
            'eval_metric': ['auc'],
            'learning_rate': [0.01,0.03,0.05], #so called `eta` value
            'max_depth': [5,7,10,13],
            'subsample': [0.5, 0.8, 0.9],
            'colsample_bytree': [0.5, 0.7, 0.9],
            'n_estimators': [5,10,15,30],
            'alpha':[10],
            'eta':[0.05]})
    return [c0, c1,c2, c2_1, c3,c4,c5]

def probar_clasificador(clf, X_test,y_test, pc=0.3):
    y_test = y_test.iloc[:,0]
    #X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=321)
    #clf.fit(X_train, y_train)
    try:
        pred = clf.predict_proba(X_test)[:,1]
    except:
        pred = clf.predict(X_test)
    print('--------------------------------------------------------')
    print('Valor AUC: ', roc_auc_score(y_test,pred))
    pred2 = np.where(pred>pc,1,0)
    print('Matriz de confusion')
    #print(confusion_matrix(y_test, pred2)) 
    print(pd.crosstab(y_test,pred2))
    print('accuracy del modelo: ', accuracy_score(y_test, pred2))
    print(classification_report(y_test,pred2))
    print('--------------------------------------------------------')
    p=np.zeros((len(pred),2))
    p[:,1] = pred
    plt.clf()
    skplt.metrics.plot_cumulative_gain(y_test, p)
    plt.show()
    return y_test, pred

"""
def TEST_CLASIFICADOR(X, y, pc=0.3):
    clasificadores = clasificadores_prueba()
    for name, clas in clasificadores:
        print('---------------------------------------------------------------------------------')
        print('------------------ ',name,' ------------------')
        y_test, p = probar_clas(clas,X,y,pc=pc)
        plt.clf()
        skplt.metrics.plot_cumulative_gain(y_test, p)
        plt.show()
        print('---------------------------------------------------------------------------------')
        print('\n')
"""

def optimizar_parametros(clf, parameters, X, y, cv=3):
    clf = GridSearchCV(clf, parameters, n_jobs=-1, cv=cv, verbose=1)
    clf.fit(X,y)
    print(clf.best_params_)
    return clf

def entrenamiento(clf, X, y, n_pliegues=10, pc=0.25):
    # creando pliegues
    kpliegues = StratifiedKFold(n_pliegues)
    print(kpliegues.get_n_splits(X, y))
    # iterando entre los plieges
    roc = []
    recall = []
    precision = []
    k=0
    for train, test in kpliegues.split(X, y):
        clf.fit(X.iloc[train], y.iloc[train]) 
        score = roc_auc_score(y.iloc[test],clf.predict(X.iloc[test]))
        rec = recall_score(y.iloc[test],np.where(clf.predict(X.iloc[test])>pc,1,0))
        prec = precision_score(y.iloc[test],np.where(clf.predict(X.iloc[test])>pc,1,0))
        roc.append(score)
        recall.append(rec)
        precision.append(prec)
        print('Pliegue: {0:}, Dist Clase: {1:}, roc: {2:.3f}, recall: {3:.3f}, precision: {4:.3f}'.format(k+1,
                        np.bincount(y.iloc[train,0]), score, rec, prec))
        k=k+1
    # imprimir promedio y desvio estandar
    print('roc promedio: {0: .3f} +/- {1: .3f}'.format(np.mean(roc),
                                          np.std(roc)))
    print('recall promedio: {0: .3f} +/- {1: .3f}'.format(np.mean(recall),
                                          np.std(recall)))
    print('precision promedio: {0: .3f} +/- {1: .3f}'.format(np.mean(precision),
                                          np.std(precision)))
    return clf


def optimizar_punto_corte(modelo,X_test, y_test, plot=1):
    try:
        pred = modelo.predict_proba(X_test)[:,1]
    except:
        pred = modelo.predict(X_test)
    pc = [0.10, 0.15, 0.20, 0.25,0.3,0.35,0.4,0.45,0.5,0.55, 0.6, 0.65, 0.7, 0.75,0.8, 0.85,0.9,0.95]
    prec = []
    rec = []
    f1 = []
    for i in pc:
        pred2 = np.where(pred>i,1,0)
        prec.append(precision_score(y_test,pred2))
        rec.append(recall_score(y_test,pred2))
        f1.append(f1_score(y_test,pred2))
    if plot==1:
        plt.plot(pc, prec, label = 'precision')
        plt.plot(pc, rec, label = 'recall')
        plt.plot(pc, f1, label = 'f1_score')
        #plt.axhline(max(f1), color='blue')
        plt.axvline(pc[pd.Series(f1).idxmax()], color='purple', label='max f1_score')
        plt.legend()
        plt.xticks(pc)
        plt.show()
    return pc[pd.Series(f1).idxmax()]

## 2DA PARTE: REGRESIONES