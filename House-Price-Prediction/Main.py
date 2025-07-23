import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Read the CSV and drop rows with missing data
df = pd.read_csv('housing.csv').dropna(axis=0)

# Separate features and target
proximity = df[['ocean_proximity']]
X = df.drop(['ocean_proximity', 'median_house_value'], axis=1)
Y = df['median_house_value']

# Scale numerical features
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Encode categorical feature
encoder = OneHotEncoder(sparse_output=False)
proximity_encoded = pd.DataFrame(
    encoder.fit_transform(proximity),
    columns=encoder.get_feature_names_out(['ocean_proximity'])
)

# Combine all into a single dataset
data = pd.concat([X.reset_index(drop=True),
            proximity_encoded.reset_index(drop=True),
            Y.reset_index(drop=True)], axis=1)

# Train-test split
training_data, testing_data = train_test_split(data, test_size=0.2, random_state=42)

X_train = training_data.drop(['median_house_value'], axis=1)
Y_train = training_data['median_house_value']

X_test = testing_data.drop(['median_house_value'], axis=1)
Y_test = testing_data['median_house_value']

# attempted to use linear regression model
# model was not accurate due to underfitting of the data
# tree = LinearRegression()


# Train a Decision Tree
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, Y_train)

# Predict for 10 random test samples
l = np.random.randint(0,np.shape(X_test)[0])
x = model.predict(X_test.iloc[range(l,l+10)])
y = Y_test.iloc[l:l+10]

for (a,b) in zip(x,y):
    print("Predicted: ",a," | Actual: ",b)