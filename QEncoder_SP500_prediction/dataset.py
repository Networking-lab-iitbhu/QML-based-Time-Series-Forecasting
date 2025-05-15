import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler


# Define base directory for loading data
BASE_DIR = "./QEncoder_SP500_prediction/"
datafiles_dir = os.path.join(BASE_DIR, 'processed_data/')  # Contains X.npy, Y.npy, F.npy, tX.npy, and tY.npy
dataset_dir = os.path.join(BASE_DIR, 'datasets/')


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


def load_and_split_csv(filename: str, args):
    """
    Loads a CSV file and splits it into features as NumPy arrays 
    for training and test sets. 

    Args:
        filename (str): Name of the CSV file in the datasets directory.
        args.test_ratio (float): command line argument for fraction of the data to be used for testing (e.g., 0.2 means 20%).

    Returns:
        (X_train, X_test): 
        - X_train, X_test: Feature arrays for training and testing
    """
    # Full path to the CSV file
    path = os.path.join(dataset_dir, filename)

    # Read the CSV file into a DataFrame (table-like structure)
    df = pd.read_csv(path)
    df=df.drop(columns=['Date'])

    # Separate features and labels
    features = df.to_numpy()  # Convert to NumPy array for model use

    # Compute the index at which to split the data for training/testing
    split_idx = int(len(df) * (1 - args.test_ratio))  # e.g., if 80% train, 20% test

    # Training data: from start to split index
    X_train = features[:split_idx]
   
    # Testing data: from split index to end
    X_test = features[split_idx:]
    print(X_train[:5])
   
    # Return the split data
    return X_train, X_test


def split_features_labels(features: np.ndarray, labels: np.ndarray, args):
    """
    Splits features and labels into training and validation sets, maintaining alignment.

    Args:
        features (np.ndarray): Full feature dataset.
        labels (np.ndarray): Full label dataset.
        args.val_ratio (float):command line argument denoting fraction to use for validation.

    Returns:
        (X_train, X_val, y_train, y_val): Tuple of splits.
    """
    assert len(features) == len(labels), "Mismatch between features and labels length"
    split_idx = int(len(features) * (1 - args.val_ratio))
    X_train = features[:split_idx]
    X_val = features[split_idx:]
    y_train = labels[:split_idx]
    y_val = labels[split_idx:]
    return X_train, X_val, y_train, y_val



def fillxy(data):
    X = []
    Y = []
    window = 10
    
    # Apply MinMaxScaler for each feature
    scaler = MinMaxScaler()
    op = scaler.fit_transform(data[:, 0].reshape(-1, 1))  # Open
    hi = scaler.fit_transform(data[:, 1].reshape(-1, 1))  # High
    lo = scaler.fit_transform(data[:, 2].reshape(-1, 1))  # Low
    cl = scaler.fit_transform(data[:, 3].reshape(-1, 1))  # Close
    vo = scaler.fit_transform(data[:, 4].reshape(-1, 1))  # Volume
    
    # Now create windows from the scaled data
    for i in range(len(op) - window):
        X.append(np.column_stack((op[i:i+window], hi[i:i+window], lo[i:i+window], cl[i:i+window], vo[i:i+window])))
        Y.append(op[i+window][0])  # Using 'Open' as the label (first column)
        
        #op is a 2D array so op[i+window] is like [[1]] so to get [1] we do index 0.
    
    return X, Y


def load_dataset(args):
    """
    Loads the dataset based on the provided argument. This function will load
    and process the dataset, scaling features and splitting into training, 
    validation, and test sets.

    Args:
        args (argparse.Namespace): Arguments containing the dataset name and test_ratio.

    Returns:
        (X_train, X_test, y_train, y_test, flattened): 
        - Processed and split feature arrays and labels.
    """
    # Format test_ratio as percentage without decimals, e.g., 0.2 -> 20
    test_ratio_str = str(int(args.test_ratio * 100))

    if args.dataset == 'wti':
        prefix = f"wti_{test_ratio_str}"
        if os.path.exists(os.path.join(datafiles_dir, f"X_{prefix}.npy")):
            with open(os.path.join(datafiles_dir, f"X_{prefix}.npy"), 'rb') as f:
                X = np.load(f)
            with open(os.path.join(datafiles_dir, f"Y_{prefix}.npy"), 'rb') as f:
                Y = np.load(f)
            with open(os.path.join(datafiles_dir, f"F_{prefix}.npy"), 'rb') as f:
                flattened = np.load(f)
            with open(os.path.join(datafiles_dir, f"tX_{prefix}.npy"), 'rb') as f:
                tX = np.load(f)
            with open(os.path.join(datafiles_dir, f"tY_{prefix}.npy"), 'rb') as f:
                tY = np.load(f)
        else:
            X_train_raw, X_test_raw = load_and_split_csv("WTI_Offshore_Cleaned_Data.csv", args)
            X, Y = fillxy(X_train_raw)  # For training
            tX, tY = fillxy(X_test_raw)  # For testing
            
            X = np.array(X)
            Y = np.array(Y)
            tX = np.array(tX)
            tY = np.array(tY)
         
            X = X.reshape((-1, 5, 10))
            tX = tX.reshape((-1, 5, 10))
            
            flattened = np.array(X).reshape((-1, 10))  # Flatten all the features for 10 timesteps.
            
            # Save numpy arrays with test_ratio in the filename
            with open(os.path.join(datafiles_dir, f"X_{prefix}.npy"), 'wb') as f:
                np.save(f, X)
            with open(os.path.join(datafiles_dir, f"Y_{prefix}.npy"), 'wb') as f:
                np.save(f, Y)
            with open(os.path.join(datafiles_dir, f"F_{prefix}.npy"), 'wb') as f:
                np.save(f, flattened)
            with open(os.path.join(datafiles_dir, f"tX_{prefix}.npy"), 'wb') as f:
                np.save(f, tX)
            with open(os.path.join(datafiles_dir, f"tY_{prefix}.npy"), 'wb') as f:
                np.save(f, tY)


    elif args.dataset == 'nifty':
        prefix = f"nifty_{test_ratio_str}"
        if os.path.exists(os.path.join(datafiles_dir, f"X_{prefix}.npy")):
            with open(os.path.join(datafiles_dir, f"X_{prefix}.npy"), 'rb') as f:
                X = np.load(f)
            with open(os.path.join(datafiles_dir, f"Y_{prefix}.npy"), 'rb') as f:
                Y = np.load(f)
            with open(os.path.join(datafiles_dir, f"F_{prefix}.npy"), 'rb') as f:
                flattened = np.load(f)
            with open(os.path.join(datafiles_dir, f"tX_{prefix}.npy"), 'rb') as f:
                tX = np.load(f)
            with open(os.path.join(datafiles_dir, f"tY_{prefix}.npy"), 'rb') as f:
                tY = np.load(f)
        else:
            X_train_raw, X_test_raw = load_and_split_csv("NIFTY50_Cleaned_Data.csv", args)
            X, Y = fillxy(X_train_raw)  # For training
            tX, tY = fillxy(X_test_raw)  # For testing
            
            X = np.array(X)
            Y = np.array(Y)
            tX = np.array(tX)
            tY = np.array(tY)
         
            X = X.reshape((-1, 5, 10))
            tX = tX.reshape((-1, 5, 10))
            
            flattened = np.array(X).reshape((-1, 10))  # Flatten all the features for 10 timesteps.
            
            # Save numpy arrays with test_ratio in the filename
            with open(os.path.join(datafiles_dir, f"X_{prefix}.npy"), 'wb') as f:
                np.save(f, X)
            with open(os.path.join(datafiles_dir, f"Y_{prefix}.npy"), 'wb') as f:
                np.save(f, Y)
            with open(os.path.join(datafiles_dir, f"F_{prefix}.npy"), 'wb') as f:
                np.save(f, flattened)
            with open(os.path.join(datafiles_dir, f"tX_{prefix}.npy"), 'wb') as f:
                np.save(f, tX)
            with open(os.path.join(datafiles_dir, f"tY_{prefix}.npy"), 'wb') as f:
                np.save(f, tY)


    elif args.dataset == 'sp500':
        prefix = f"sp500_{test_ratio_str}"
        if os.path.exists(os.path.join(datafiles_dir, f"X_{prefix}.npy")):
            with open(os.path.join(datafiles_dir, f"X_{prefix}.npy"), 'rb') as f:
                X = np.load(f)
            with open(os.path.join(datafiles_dir, f"Y_{prefix}.npy"), 'rb') as f:
                Y = np.load(f)
            with open(os.path.join(datafiles_dir, f"F_{prefix}.npy"), 'rb') as f:
                flattened = np.load(f)
            with open(os.path.join(datafiles_dir, f"tX_{prefix}.npy"), 'rb') as f:
                tX = np.load(f)
            with open(os.path.join(datafiles_dir, f"tY_{prefix}.npy"), 'rb') as f:
                tY = np.load(f)
        else:
            # Your existing SP500 loading and processing code unchanged
            df = pd.read_csv(os.path.join(dataset_dir, "combined_dataset.csv"))
            stocks = set(df['Name'])
            X = []
            Y = []
            tX = []
            tY = []
            window = 10

            def fillxy_sp500(stock, test=False):
                stock_df = df[df['Name'] == stock].copy()
                stock_df = stock_df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])

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
                if stock != 'S&P500':
                    fillxy_sp500(stock)
                else:
                    fillxy_sp500(stock, True)

            X = np.array(X)
            Y = np.array(Y)
            tX = np.array(tX)
            tY = np.array(tY)

            X = X.reshape((-1, 5, 10))
            tX = tX.reshape((-1, 5, 10))

            flattened = np.array(X).reshape((-1, 10))
            
            with open(os.path.join(datafiles_dir, f"X_{prefix}.npy"), 'wb') as f:
                np.save(f, X)
            with open(os.path.join(datafiles_dir, f"Y_{prefix}.npy"), 'wb') as f:
                np.save(f, Y)
            with open(os.path.join(datafiles_dir, f"F_{prefix}.npy"), 'wb') as f:
                np.save(f, flattened)
            with open(os.path.join(datafiles_dir, f"tX_{prefix}.npy"), 'wb') as f:
                np.save(f, tX)
            with open(os.path.join(datafiles_dir, f"tY_{prefix}.npy"), 'wb') as f:
                np.save(f, tY)

    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    return X, Y, tX, tY, flattened
