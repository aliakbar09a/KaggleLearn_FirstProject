import pandas as pd

main_file_path = '../data/train.csv'
data = pd.read_csv(main_file_path)
print(data.describe())

# SalePrice column
SalePrice = data.SalePrice
print(SalePrice.head())

# two columns data
two_columns = ['GarageArea', 'YrSold']
two_columns_data = data[two_columns]
print(two_columns_data.describe())
print(data.columns)
# selecting the features for training
features = [ 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
             'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF','FullBath', 'BedroomAbvGr',
             'TotRmsAbvGrd', '1stFlrSF', '2ndFlrSF']
X = data[features]
y = data[['SalePrice']]

from sklearn.tree import DecisionTreeRegressor

# define model
model = DecisionTreeRegressor()

# fit model
model.fit(X, y)

# predicting some values
print("Making Predictions for the following 5 houses")
print(X.head())
print("The predictions are")
print(model.predict(X.head()))
print(data.SalePrice.head())

# splitting training and test data
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
X_train, val_X, y_train, val_y = train_test_split(X, y, random_state=0)
model = DecisionTreeRegressor()
model.fit(X_train, y_train)
predicted_y = model.predict(val_X)
print(mean_absolute_error(val_y, predicted_y))

# defining a utility function to return mae
def mae(max_leaf, X_train, val_X, y_train, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes= max_leaf, random_state=0)
    model.fit(X_train,y_train)
    predicted_y = model.predict(val_X)
    m_a_e = mean_absolute_error(val_y, predicted_y)
    return(m_a_e)

print("The errors are for different values of max leaf nodes")
# checking error for different values of max_leaf
for i in [5, 50, 100, 250, 500, 750, 1000]:
    print("i = ", i)
    error = mae(i, X_train, val_X,y_train, val_y)
    print(error)

#using random forest
from sklearn.ensemble import RandomForestRegressor
forest = RandomForestRegressor()
forest.fit(X_train, y_train)
predicted_y = forest.predict(val_X)
print(mean_absolute_error(val_y, predicted_y))
