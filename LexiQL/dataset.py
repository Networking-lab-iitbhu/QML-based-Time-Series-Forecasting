import numpy as np
import pandas as pd
import os

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

if os.path.exists('./X.npy'):
    with open('./X.npy', 'rb') as f:
        X = np.load(f)
    with open('./Y.npy', 'rb') as f:
        Y = np.load(f)
    with open('./F.npy', 'rb') as f:
        flattened = np.load(f)
else:
    df = pd.read_csv('./stocks.csv')
    df = df.dropna()
    stocks = set(df['Name'])
    print(f'number of stocks: {len(stocks)}')

    X = []
    Y = []
    window = 10

    for stock in stocks:
        stock_df = df[df['Name'] == stock]
        op, hi, lo, cl, vo = [], [], [], [], []
        # print(stock_df.iloc[0])
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
            if i + window >= len(op): break
            X.append(np.column_stack((op[i: i+window], hi[i: i+window], lo[i: i+window], cl[i: i+window], vo[i: i+window])))
            Y.append(op[i+window])
            
    X = np.array(X)
    Y = np.array(Y)
    flattened = np.array(X).reshape((-1, 5))
    with open('X.npy', 'wb') as f:
        np.save(f, X)
    with open('Y.npy', 'wb') as f:
        np.save(f, Y)
    with open('F.npy', 'wb') as f:
        np.save(f, flattened)



