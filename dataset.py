import pandas as pd
data=pd.read_csv('processed.cleveland.data')

# drop the missing values in the dataset
data=data.dropna()
data['num'] = (data['num']>0)+0
data['num'].unique()

data.to_csv('./HeartAttack.csv',index=False)