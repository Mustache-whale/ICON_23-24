
import matplotlib.pyplot as plt
import pandas as pd

from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import ConfusionMatrixDisplay,classification_report
from sklearn.model_selection import cross_val_score, train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from statistics import median


#make X,y
df = pd.read_csv("../Dataset/diabetes_dataset.csv")

df = df.drop('location', axis = 1)
df = df.drop(df[df.gender == 'Other'].index)
df['gender'] = df['gender'].apply(lambda t: False if t == 'Male' else True)
df = df.drop(df[df.smoking_history == 'No Info'].index)
df['smoking_history'] = df['smoking_history'].apply(lambda s:0 if s == 'never' or s == 'ever' else s)
df['smoking_history'] = df['smoking_history'].apply(lambda s:1 if s == 'former' or s == 'not current' else s)
df['smoking_history'] = df['smoking_history'].apply(lambda s:2 if s == 'current' else s)
df= df.rename(columns ={'hbA1c_level': 'hbA1c', 'blood_glucose_level': 'glucose'})
df = df.drop_duplicates()

features = ['glucose','hbA1c','age','bmi','smoking_history','heart_disease','hypertension']

def confusion_matrix(df,features):

    X = df.loc[:,features]
    y = df.diabetes

    rus = RandomUnderSampler(sampling_strategy='majority')
    X,y = rus.fit_resample(X,y)

    X_train,X_test,y_train,y_test= train_test_split(X, y, test_size=0.40, random_state= 42)

    model = KNeighborsClassifier(n_neighbors=11)
    model = model.fit(X_train,y_train)
    y_pred = model.predict(X_test)

    ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test,y_pred)).plot()
    plt.title('K-nearest neighbors')
    plt.show()

def model_classification_report(df,features):

    X = df.loc[:,features]
    y = df.diabetes

    rus = RandomUnderSampler(sampling_strategy='majority')
    X,y = rus.fit_resample(X,y)

    X_train,X_test,y_train,y_test= train_test_split(X, y, test_size=0.40, random_state= 42)

    model = KNeighborsClassifier(n_neighbors=11)
    model = model.fit(X_train,y_train)
    y_pred = model.predict(X_test)

    print('<nome modello>:\n',classification_report(y_test,y_pred))

def param_variation(df,features):

    #usato per testare il variare dei parametri che assumono un range di valori

    X = df.loc[:,features]
    y = df.diabetes
    scores_list = []
    best_param = []

    for j in range (0,7): #creiamo più grafici per avere un' idea generale dell'andamento

        rus = RandomUnderSampler(sampling_strategy='majority',)
        X,y = rus.fit_resample(X,y)

        scores = []
        params = []

        for i in range(2,16,2):

            print(i)

            model = KNeighborsClassifier(n_neighbors=i)
            s = sum(cross_val_score(model,X,y,cv = 5,scoring = 'f1').tolist())/5
            scores.append(s)
            params.append(i)

        scores_list.append(scores)
        best_param.append(params[scores.index(max(scores))])

    for t in scores_list:
        plt.plot(params, t, label =str(scores_list.index(t)) + ', ' + str(best_param[scores_list.index(t)]))

    m = str(median(best_param))  #mediana
    #m = str(max(set(best_param), key = best_param.count))  #moda
    m = str(sum(best_param)/len(best_param))  #media

    plt.xlabel('n_neighbors')
    plt.ylabel('F1-score')
    plt.title('n_neighbors variation')
    plt.legend(loc = 'best',title = '#iteration, best value\n median best value: '+m)
    plt.show()

def compare(df,features):

    #usato per confrontare diversi classificatori
    #o lo stesso classificatore con parametri che assumono valori specifici (es. oob_score in random forest)
    #strutturato come template per facilitare test in sequenza senza troppe modifiche al codice

    elements = list()

    scores1 = list()
    scores2 = list()
    scores3 = list()
    scores4 = list()


    for i in range(100,63700,250): #range di elementi testati (start,stop,step)
        print(i)

        a = df.sample(n = i)
        X = a.loc[:,features]
        y = a.diabetes

        rus = RandomUnderSampler(sampling_strategy='majority',)
        X,y = rus.fit_resample(X,y)

        elements.append(y.count())

        model = DecisionTreeClassifier(max_depth=7)
        s = sum(cross_val_score(model,X,y,cv = 5,scoring = 'f1').tolist())/5
        scores1.append(s)

        model = RandomForestClassifier(n_estimators=170,max_features=2)
        s = sum(cross_val_score(model,X,y,cv = 5,scoring = 'f1').tolist())/5
        scores2.append(s)

        model = RandomForestClassifier()
        s = sum(cross_val_score(model,X,y,cv = 5,scoring = 'f1').tolist())/5
        scores3.append(s)

        model = GaussianNB()
        #s = sum(cross_val_score(model,X,y,cv = 5,scoring = 'f1').tolist())/5
        #scores4.append(s)


    plt.plot(elements,scores1, '.b', label ='(decision tree)')
    plt.plot(elements,scores2, '.r', label ='(random forest)')
    plt.plot(elements,scores3, '.g', label ='(random forest (control))')
    #plt.plot(elements,scores4, '.y', label ='(Naive Bayes)')


    plt.ylabel('F1-score')
    plt.xlabel('number of elements')
    plt.legend(loc = 'best')
    plt.title('<title>')
    plt.show()