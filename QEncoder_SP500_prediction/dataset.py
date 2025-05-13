import numpy as np
import pandas as pd
import os


# Define base directory for loading data
BASE_DIR = "./QEncoder_SP500_prediction/"
datafiles_dir = os.path.join(BASE_DIR,'processed_data/') #it has X.npy,Y.npy,F.npy,tX.npy and tY.npy
dataset_dir = os.path.join(BASE_DIR,'datasets/')

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




def load_and_split_csv(filename: str, test_ratio: float = 0.2):
    """
    Loads a CSV file and splits it into features and labels as NumPy arrays 
    for training and test sets. 

    Args:
        filename (str): Name of the CSV file in the datasets directory.
        test_ratio (float): Fraction of the data to be used for testing (e.g., 0.2 means 20%).

    Returns:
        (X_train, X_test, y_train, y_test): 
        - X_train, X_test: Feature arrays for training and testing
        - y_train, y_test: Corresponding labels (target values)
    """
    # Full path to the CSV file
    path = os.path.join(dataset_dir, filename)

    # Read the CSV file into a DataFrame (table-like structure)
    df = pd.read_csv(path)

    # Separate features and labels:
    # df.iloc[:, :-1] selects all rows and all columns except the last one.
    # These are the input features (e.g., open, high, low prices).
    features = df.iloc[:, :-1].to_numpy()  # Convert to NumPy array for model use

    # df.iloc[:, -1] selects the last column of the DataFrame.
    # This is assumed to be the target or label (e.g., the closing price).
    labels = df.iloc[:, -1].to_numpy()

    # Compute the index at which to split the data for training/testing
    split_idx = int(len(df) * (1 - test_ratio))  # e.g., if 80% train, 20% test

    # Training data: from start to split index
    X_train = features[:split_idx]
    y_train = labels[:split_idx]

    # Testing data: from split index to end
    X_test = features[split_idx:]
    y_test = labels[split_idx:]

    # Return the split data
    return X_train, X_test, y_train, y_test



def split_features_labels(features: np.ndarray, labels: np.ndarray, val_ratio: float = 0.2):
    """
    Splits features and labels into training and validation sets, maintaining alignment.

    Args:
        features (np.ndarray): Full feature dataset.
        labels (np.ndarray): Full label dataset.
        val_ratio (float): Fraction to use for validation.

    Returns:
        (X_train, X_val, y_train, y_val): Tuple of splits.
    """
    assert len(features) == len(labels), "Mismatch between features and labels length"
    split_idx = int(len(features) * (1 - val_ratio))
    X_train = features[:split_idx]
    X_val = features[split_idx:]
    y_train = labels[:split_idx]
    y_val = labels[split_idx:]
    return X_train, X_val, y_train, y_val





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
    # Load datasets
    df= pd.read_csv(os.path.join(dataset_dir, "combined_dataset.csv"))
    stocks = set(df['Name'])
    
    print(f'Number of stocks: {len(stocks)}')

    X = []
    Y = []
    tX = []
    tY = []
    window = 10

    def fillxy(stock, test=False):
        stock_df = df[df['Name'] == stock].copy()

        # Drop rows with NaNs in any of the required columns
        stock_df = stock_df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])

        op, hi, lo, cl, vo = [], [], [], [], []

        for idx in range(len(stock_df)):
            _, o, h, l, c, v, _ = list(stock_df.iloc[idx])
            op.append(o)
            hi.append(h)
            lo.append(l)
            cl.append(c)
            vo.append(v)

        # Apply MinMaxScaler after dropping NaNs
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
        if stock!= 'S&P500':
            fillxy(stock)
        else:
            fillxy(stock,True)

    X = np.array(X)
    Y = np.array(Y)
    tX = np.array(tX)
    tY = np.array(tY)

    X = X.reshape((-1, 5, 10))
    tX = tX.reshape((-1, 5, 10))

    flattened = np.array(X).reshape((-1, 10))  #flattened is all the features for 10 timesteps.
    print(flattened)
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


