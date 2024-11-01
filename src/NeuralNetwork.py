import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,f1_score

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


X = df.loc[:,features]
y = df.diabetes


def report(X,y):

    rus = RandomUnderSampler(sampling_strategy='majority',)
    X,y = rus.fit_resample(X,y)

    X_train,X_test,y_train,y_test= train_test_split(X, y, test_size=0.40, random_state= 42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print('---')
    model = MLPClassifier(max_iter=350)

    model = model.fit(X_train,y_train)

    y_pred = model.predict(X_test)

    print('neural network:\n ',classification_report(y_test,y_pred))

    model = RandomForestClassifier(n_estimators=170,max_features=1,max_depth=7)

    model = model.fit(X_train,y_train)

    y_pred = model.predict(X_test)

    print('random forest:\n ',classification_report(y_test,y_pred))

def best_param(a,b):

    scores_list = []
    best_param = []

    for j in range (0,5):

        scores = list()
        params = list()

        rus = RandomUnderSampler(sampling_strategy='majority',)
        X,y = rus.fit_resample(a,b)

        X_train,X_test,y_train,y_test= train_test_split(X, y, test_size=0.30)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)


        for i in range(1,21,2):
            print(i)
            p = i*0.00001

            model = MLPClassifier(alpha=p, max_iter= 300)
            model = model.fit(X_train,y_train)
            y_pred = model.predict(X_test)
            scores.append(f1_score(y_test,y_pred))
            params.append(p)

        scores_list.append(scores)
        best_param.append(params[scores.index(max(scores))])

    for t in scores_list:
        plt.plot(params, t, label =str(scores_list.index(t)) + ', ' + str(best_param[scores_list.index(t)]))

    #m = str(max(set(best_param), key = best_param.count))
    m = str(sum(best_param)/len(best_param))
    plt.xlabel('alpha variation')
    plt.ylabel('F1-score')
    plt.title('alpha')
    plt.legend(loc = 'best',title = '#iteration, best value\n avg best value: '+m)
    plt.show()


def compare(X,y):
    elements = list()

    scores1 = list()
    scores2 = list()
    scores3 = list()
    scores4 = list()


    for i in range(0,16):
        print(i)

        rus = RandomUnderSampler(sampling_strategy='majority',)
        X,y = rus.fit_resample(X,y)

        X_train,X_test,y_train,y_test= train_test_split(X, y, test_size=0.30)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        elements.append(y.count())


        model = MLPClassifier(max_iter=350)#1 baseline
        model = model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
        scores1.append(f1_score(y_test,y_pred))


        model = RandomForestClassifier(n_estimators=170,max_features=1,max_depth=7)#2
        model = model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
        scores2.append(f1_score(y_test,y_pred))

        #model = MLPClassifier(max_iter=350,learning_rate='adaptive',solver='sgd')#3
        #model = model.fit(X_train,y_train)
        #y_pred = model.predict(X_test)
        #scores3.append(f1_score(y_test,y_pred))

        #model = MLPClassifier(max_iter=350,solver='adam')#3
        #model = model.fit(X_train,y_train)
        #y_pred = model.predict(X_test)
        #scores4.append(f1_score(y_test,y_pred))




    plt.plot(scores1, '.', label ='(neural network)')
    plt.plot(scores2, '.', label ='(random forest)')
    #plt.plot(scores3, '.', label ='(sgd adaptive)')
    #plt.plot(scores4, '.', label ='(adam)')


    plt.ylabel('F1-score')
    plt.legend(loc = 'best')
    plt.title('learning rate for stochastic gradient descent')
    plt.show()



report(X,y)













