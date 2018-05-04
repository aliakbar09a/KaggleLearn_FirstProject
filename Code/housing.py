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
features = ['MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',
       'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
       'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
       'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
       'RoofStyle', 'RoofMatl']
X = data[features]
y = data[['SalePrice']]