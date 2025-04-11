# import numpy as np
# import pandas as pd
# import yfinance as yf
# import os

# df2 = pd.read_csv(r"C:\Users\geeth\Downloads\quantum-ml-main\quantum-ml-main\QEncoder_SP500_prediction\sp.csv")

# df2 = df2[df2.columns[:-2]]
# df2 = df2.reset_index()
# df2['Name'] = 'S&P500'
# df2 = df2.rename(columns={x: x.lower() if x[0]!='N' else x for x in df2.columns})


# class MinMaxScaler:
#     epsilon = 1e-5
#     @classmethod
#     def fit_transform(cls, data):
#         x = np.array(data)
#         mn = x.min() - cls.epsilon
#         mx = x.max()
#         rng = (mx - mn) + cls.epsilon
#         def norm(y):
#             return (y - mn) / rng
#         return np.array([norm(y) for y in data])

# if os.path.exists('./X.npy'):
#     with open('./X.npy', 'rb') as f:
#         X = np.load(f)
#     with open('./Y.npy', 'rb') as f:
#         Y = np.load(f)
#     with open('./F.npy', 'rb') as f:
#         flattened = np.load(f)
#     with open('./tX.npy', 'rb') as f:
#         tX = np.load(f)
#     with open('./tY.npy', 'rb') as f:
#         tY = np.load(f)
# else:
#     df = pd.read_csv(r"C:\Users\geeth\Downloads\quantum-ml-main\quantum-ml-main\QEncoder_SP500_prediction\stocks.csv")

#     df = df.dropna()
#     df = pd.concat([df, df2])
#     df = df.drop(columns=['index'])
#     stocks = set(df['Name'])
    
#     print(f'number of stocks: {len(stocks)}')

#     X = []
#     Y = []
#     tX = []
#     tY = []
#     window = 10

#     def fillxy(stock, test=False):
#         stock_df = df[df['Name'] == stock]
#         op, hi, lo, cl, vo = [], [], [], [], []
#         # print(stock_df.iloc[0])
#         for idx in range(len(stock_df)):
#             _, o, h, l, c, v, _ = list(stock_df.iloc[idx])
#             op.append(o)
#             hi.append(h)
#             lo.append(l)
#             cl.append(c)
#             vo.append(v)
#         op = MinMaxScaler.fit_transform(op)
#         hi = MinMaxScaler.fit_transform(hi)
#         lo = MinMaxScaler.fit_transform(lo)
#         cl = MinMaxScaler.fit_transform(cl)
#         vo = MinMaxScaler.fit_transform(vo)
        
#         for i in range(0, len(op), window):
#             if i + window >= len(op): break
#             if test:
#                 tX.append(np.column_stack((op[i: i+window], hi[i: i+window], lo[i: i+window], cl[i: i+window], vo[i: i+window])))
#                 tY.append(op[i+window])
#             else:
#                 X.append(np.column_stack((op[i: i+window], hi[i: i+window], lo[i: i+window], cl[i: i+window], vo[i: i+window])))
#                 Y.append(op[i+window])

#     for stock in stocks:
#         if stock != "S&P500":
#             fillxy(stock)
#         else:
#             fillxy(stock, True)
            
#     X = np.array(X)
#     Y = np.array(Y)
#     tX = np.array(tX)
#     tY = np.array(tY)

#     X = X.reshape((-1, 5, 10))
#     tX = tX.reshape((-1, 5, 10))
    
#     flattened = np.array(X).reshape((-1, 10))

#     with open('X.npy', 'wb') as f:
#         np.save(f, X)
#     with open('Y.npy', 'wb') as f:
#         np.save(f, Y)
#     with open('F.npy', 'wb') as f:
#         np.save(f, flattened)
#     with open('tX.npy', 'wb') as f:
#         np.save(f, tX)
#     with open('tY.npy', 'wb') as f:
#         np.save(f, tY)        


import numpy as np
import pandas as pd
import yfinance as yf
import os

# Define base directory for loading data
BASE_DIR = "./QEncoder_SP500_prediction"

# Load datasets
df2 = pd.read_csv(os.path.join(BASE_DIR, "sp.csv"))

df2 = df2[df2.columns[:-2]]
df2 = df2.reset_index()
df2['Name'] = 'S&P500'
df2 = df2.rename(columns={x: x.lower() if x[0] != 'N' else x for x in df2.columns})


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
if os.path.exists(os.path.join(BASE_DIR, "X.npy")):
    with open(os.path.join(BASE_DIR, "X.npy"), 'rb') as f:
        X = np.load(f)
    with open(os.path.join(BASE_DIR, "Y.npy"), 'rb') as f:
        Y = np.load(f)
    with open(os.path.join(BASE_DIR, "F.npy"), 'rb') as f:
        flattened = np.load(f)
    with open(os.path.join(BASE_DIR, "tX.npy"), 'rb') as f:
        tX = np.load(f)
    with open(os.path.join(BASE_DIR, "tY.npy"), 'rb') as f:
        tY = np.load(f)
else:
    # Load stocks dataset
    df = pd.read_csv(os.path.join(BASE_DIR, "stocks.csv"))

    df = df.dropna()
    df = pd.concat([df, df2])
    df = df.drop(columns=['index'])
    stocks = set(df['Name'])

    print(f'Number of stocks: {len(stocks)}')

    X = []
    Y = []
    tX = []
    tY = []
    window = 10

    def fillxy(stock, test=False):
        stock_df = df[df['Name'] == stock]
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

    for stock in stocks:
        if stock != "S&P500":
            fillxy(stock)
        else:
            fillxy(stock, True)

    X = np.array(X)
    Y = np.array(Y)
    tX = np.array(tX)
    tY = np.array(tY)

    X = X.reshape((-1, 5, 10))
    tX = tX.reshape((-1, 5, 10))

    flattened = np.array(X).reshape((-1, 10))

    # Save numpy arrays in the correct directory
    with open(os.path.join(BASE_DIR, "X.npy"), 'wb') as f:
        np.save(f, X)
    with open(os.path.join(BASE_DIR, "Y.npy"), 'wb') as f:
        np.save(f, Y)
    with open(os.path.join(BASE_DIR, "F.npy"), 'wb') as f:
        np.save(f, flattened)
    with open(os.path.join(BASE_DIR, "tX.npy"), 'wb') as f:
        np.save(f, tX)
    with open(os.path.join(BASE_DIR, "tY.npy"), 'wb') as f:
        np.save(f, tY)
