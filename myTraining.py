import pandas as pd 
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle

def data_split(data,ratio):
    np.random.seed(42)
    shuffled=np.random.permutation(len(data))
    test_set_size=int(len(data) * ratio)
    test_indices=shuffled[:test_set_size]
    train_indices=shuffled[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


if __name__ == "__main__":
    df = pd.read_csv('data.csv')
    df = df.drop('Unnamed: 6', 1)
    train, test = data_split(df,0.2)
    X_train = train [['fever', 'bodyPain','age', 'runnyNose', 'difffBreath']].to_numpy()
    X_test = test [['fever', 'bodyPain','age', 'runnyNose', 'difffBreath']].to_numpy()
    Y_train=train[['infectionProb']].to_numpy().reshape(2400,)
    Y_test=test[['infectionProb']].to_numpy().reshape(599,)
    clf=LogisticRegression()
    clf.fit(X_train, Y_train)
    file = open('model.pk1', 'wb')
    pickle.dump(clf, file)
    #Code for Inference
    inputFeatures=[100,1,22,-1,1]
    infProb=clf.predict_proba([inputFeatures])[0][1]