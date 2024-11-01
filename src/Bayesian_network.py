import pandas as pd
from pgmpy.estimators import PC, ExpectationMaximization
from pgmpy.models.BayesianNetwork import BayesianNetwork
from pgmpy.inference import VariableElimination



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

df = df.drop(df[df['diabetes'] == 0].sample(frac=0.70).index)


features = ['age','hypertension','heart_disease','smoking_history','bmi','hbA1c','glucose','diabetes']

data = df.loc[:,features]



def make_model1():

    model = BayesianNetwork([('age','hypertension'),('age','heart_disease'),('age','smoking_history'),('age','bmi'),
                            ('age','hbA1c'),('age','glucose'),('hypertension','heart_disease'),('bmi','hypertension'),
                            ('hypertension','diabetes'),('smoking_history','hypertension'),
                            ('smoking_history','heart_disease'),('bmi','heart_disease'),('bmi','hbA1c'),
                            ('bmi','glucose'),('glucose','hbA1c'),('hbA1c','diabetes'),('glucose','diabetes')])

    return model

def make_model2(df):
    est = PC(data = df)
    est = est.estimate()
    model = BayesianNetwork(est)
    return model

print('0-')
model = make_model2(df.loc[:,features])
print('1-')
model.fit(data=data, estimator=ExpectationMaximization)
print('2-')

inferenza = VariableElimination(model)
print(inferenza.query(['diabetes'], evidence = {'hypertension':0,'heart_disease':0,'hbA1c':6.1}))
