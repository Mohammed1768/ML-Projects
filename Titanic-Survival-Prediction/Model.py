import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("titanic_train.csv")

df.drop(['passenger_id', 'cabin', 'body', 'home.dest', 'ticket'], axis=1, inplace=True)
df['boat'] = df['boat'].notna().astype(int)
df['boat'] = df['boat'].fillna(0)
df.dropna(subset=['fare', 'embarked', 'age'], axis=0, inplace=True)
df['title'] = df['name'].str.extract(r',\s*([^ ]+)\s')
df.drop(['name'], axis=1, inplace=True)
enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
x = enc.fit_transform(df[['sex', 'title', 'embarked']])
df = df.drop(['sex', 'title', 'embarked'], axis=1)
df = pd.concat([pd.DataFrame(x, columns=enc.get_feature_names_out(['sex', 'title', 'embarked'])), df], axis=1)
df.dropna(subset=['survived'], inplace=True)
df['survived'] = df['survived'].astype(int)
x = df.drop("survived", axis=1)  
y = df["survived"]              
model = DecisionTreeClassifier(max_depth=3, max_leaf_nodes=4, random_state=42)
model.fit(x, y)



df_test = pd.read_csv("titanic_test.csv")
passenger_ids = df_test['passenger_id']
df_test['fare'] = df_test['fare'].fillna(df_test['fare'].median())
df_test['age'] = df_test['age'].fillna(df_test['age'].median())
df_test['embarked'] = df_test['embarked'].fillna(df_test['embarked'].mode()[0])
df_test['boat'] = df_test['boat'].notna().astype(int)
df_test['title'] = df_test['name'].str.extract(r',\s*([^ ]+)\s')
x_test_enc = enc.transform(df_test[['sex', 'title', 'embarked']])
x_test_enc_df = pd.DataFrame(x_test_enc, columns=enc.get_feature_names_out(['sex', 'title', 'embarked']))
df_test.drop(['sex', 'title', 'embarked', 'name', 'passenger_id', 'cabin', 'body', 'home.dest', 'ticket'], axis=1, inplace=True)
X_test = pd.concat([x_test_enc_df, df_test], axis=1)
X_test = X_test[x.columns]  # ensures same order and feature names
y_pred = model.predict(X_test)

submission = pd.DataFrame({
    "passenger_id": passenger_ids,
    "Survived": y_pred
})

submission.to_csv("submission.csv", index=False)
