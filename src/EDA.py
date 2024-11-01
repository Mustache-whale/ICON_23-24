import pandas as pd

import seaborn as sns                       #visualisation
import matplotlib.pyplot as plt



def heatmap(x):
    corr = x.corr()
    plt.figure(figsize = (12,10))
    sns.heatmap(corr, annot=True, cmap = 'coolwarm', linewidths = 0.5)
    plt.title('correlations')
    plt.show()

def preprocess(df):
    df = df.drop('location', axis = 1)
    df = df.drop(df[df.gender == 'Other'].index) # 18 rows
    df['gender'] = df['gender'].apply(lambda t: False if t == 'Male' else True)
    df = df.drop(df[df.smoking_history == 'No Info'].index)
    df['smoking_history'] = df['smoking_history'].apply(lambda s:0 if s == 'never' or s == 'ever' else s)
    df['smoking_history'] = df['smoking_history'].apply(lambda s:1 if s == 'former' or s == 'not current' else s)
    df['smoking_history'] = df['smoking_history'].apply(lambda s:2 if s == 'current' else s)
    df= df.rename(columns ={'hbA1c_level': 'hbA1c', 'blood_glucose_level': 'glucose'})
    df = df.drop_duplicates()

    return df

def plot_value_counts(df):
    subset = ['gender','diabetes']
    counts = df.value_counts(subset = subset, sort = False)
    counts.plot(kind = 'bar')
    plt.show()


df = pd.read_csv("../Dataset/diabetes_dataset.csv")
df = preprocess(df)

df.info(verbose=True)

heatmap(df)

plot_value_counts(df)