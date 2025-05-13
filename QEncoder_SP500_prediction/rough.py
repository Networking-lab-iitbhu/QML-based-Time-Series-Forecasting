import pandas as pd
import os

# Paths
BASE_DIR = "./QEncoder_SP500_prediction/datasets/"
stocks_path = os.path.join(BASE_DIR, "stocks.csv")
sp_path = os.path.join(BASE_DIR, "sp.csv")
combined_path = os.path.join(BASE_DIR, "combined_dataset.csv")

# Load both CSVs
stocks_df = pd.read_csv(stocks_path)
sp_df = pd.read_csv(sp_path)

# Remove any unwanted columns from sp_df (columns with missing data)
sp_df = sp_df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]  # Keep only the relevant columns

# Add 'Name' column to sp_df (set it as 'S&P500' for all rows)
sp_df['Name'] = 'S&P500'

# Concatenate the two datasets (train and test)
combined_df = pd.concat([stocks_df, sp_df], ignore_index=True)

# Optional: Drop any duplicates and sort by 'Name' and 'Date'
combined_df = combined_df.drop_duplicates()
combined_df = combined_df.sort_values(by=['Name', 'Date']).reset_index(drop=True)

# Save the merged dataset to a new CSV file
combined_df.to_csv(combined_path, index=False)

print(f"Combined dataset saved to: {combined_path}")
