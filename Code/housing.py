import pandas as pd

main_file_path = '../data/train.csv'
data = pd.read_csv(main_file_path)
print(data.describe())
#SalePrice column
SalePrice = data.SalePrice
print(SalePrice.head())
#two columns data
two_columns = ['GarageArea', 'YrSold']
two_columns_data = data[two_columns]
print(two_columns_data.describe())