import numpy as np
import pandas as pd
import yfinance as yf
import os

# Define base directory for loading data
BASE_DIR = "./QEncoder_SP500_prediction/"
datafiles_dir = os.path.join(BASE_DIR,'processed_data/') #it has X.npy,Y.npy,F.npy,tX.npy and tY.npy
dataset_dir = os.path.join(BASE_DIR,'datasets/')

# Load datasets
test_df= pd.read_csv(os.path.join(dataset_dir, "sp.csv"))

#sp.csv is the test set.

test_df = test_df[test_df.columns[:-2]]
test_df = test_df.reset_index()


class MinMaxScaler:
    epsilon = 1e-5

    @classmethod
    def fit_transform(cls, data):
        x = np.array(data)
        mn = x.min() - cls.epsilon
        mx = x.max()
        rng = (mx - mn) + cls.epsilon

        def norm(y):
            return (y - mn) / rng

        return np.array([norm(y) for y in data])


# Check if numpy arrays exist
if os.path.exists(os.path.join(datafiles_dir, "X.npy")):
    with open(os.path.join(datafiles_dir, "X.npy"), 'rb') as f:
        X = np.load(f)
    with open(os.path.join(datafiles_dir, "Y.npy"), 'rb') as f:
        Y = np.load(f)
    with open(os.path.join(datafiles_dir, "F.npy"), 'rb') as f:
        flattened = np.load(f)
    with open(os.path.join(datafiles_dir, "tX.npy"), 'rb') as f:
        tX = np.load(f)
    with open(os.path.join(datafiles_dir, "tY.npy"), 'rb') as f:
        tY = np.load(f)
else:
    # Load stocks dataset
    train_df = pd.read_csv(os.path.join(dataset_dir, "stocks.csv"))
    #stocks.csv is training set.
    train_df = train_df.dropna()
    
    stocks = set(train_df['Name']).union(set(test_df['Name']))
    
    print(f'Number of stocks: {len(stocks)}')

    X = []
    Y = []
    tX = []
    tY = []
    window = 10

    def fillxy(stock_df, test=False):
        
        op, hi, lo, cl, vo = [], [], [], [], []
        
        for idx in range(len(stock_df)):
            _, o, h, l, c, v, _ = list(stock_df.iloc[idx])
            op.append(o)
            hi.append(h)
            lo.append(l)
            cl.append(c)
            vo.append(v)

        op = MinMaxScaler.fit_transform(op)
        hi = MinMaxScaler.fit_transform(hi)
        lo = MinMaxScaler.fit_transform(lo)
        cl = MinMaxScaler.fit_transform(cl)
        vo = MinMaxScaler.fit_transform(vo)

        for i in range(0, len(op), window):
            if i + window >= len(op):
                break
            if test:
                tX.append(np.column_stack((op[i:i+window], hi[i:i+window], lo[i:i+window], cl[i:i+window], vo[i:i+window])))
                tY.append(op[i+window])
            else:
                X.append(np.column_stack((op[i:i+window], hi[i:i+window], lo[i:i+window], cl[i:i+window], vo[i:i+window])))
                Y.append(op[i+window])

    for stock in train_df['Name'].unique():
        stock_data = train_df[train_df['Name'] == stock] 
        fillxy(stock_data, test=False)  
    
    fillxy(test_df, test=True) #independently passing the test set 

    X = np.array(X)
    Y = np.array(Y)
    tX = np.array(tX)
    tY = np.array(tY)

    X = X.reshape((-1, 5, 10))
    tX = tX.reshape((-1, 5, 10))

    flattened = np.array(X).reshape((-1, 10))  #flattened is all the features for 10 timesteps.

    # Save numpy arrays in the correct directory
    with open(os.path.join(datafiles_dir, "X.npy"), 'wb') as f:
        np.save(f, X)
    with open(os.path.join(datafiles_dir, "Y.npy"), 'wb') as f:
        np.save(f, Y)
    with open(os.path.join(datafiles_dir,"F.npy"), 'wb') as f:
        np.save(f, flattened)
    with open(os.path.join(datafiles_dir, "tX.npy"), 'wb') as f:
        np.save(f, tX)
    with open(os.path.join(datafiles_dir, "tY.npy"), 'wb') as f:
        np.save(f, tY)


