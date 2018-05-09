import pandas as pd
data = pd.read_csv('../data/train.csv')
ndata = data.select_dtypes(exclude= ['object'])

col_with_missing = [col for col in ndata.columns
                        if ndata[col].isnull().any()]

for col in col_with_missing:
    ndata[col + 'was_missing'] = ndata[col].isnull()

from sklearn.preprocessing import Imputer
imputer = Imputer()
new_data = imputer.fit_transform(ndata)