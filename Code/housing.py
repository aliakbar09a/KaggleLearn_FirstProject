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