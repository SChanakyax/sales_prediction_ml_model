import os
import pandas as pd

print(os.listdir("dataset"))

# Get the current directory or specify one
folder_path = os.getcwd()
file_path = os.path.join(folder_path, "dataset", "retail_sales_dataset.csv")

# Verify before reading to be sure
if os.path.exists(file_path):
    data = pd.read_csv(file_path)
    print("File loaded successfully!")
    print(data.columns)
    print(data.info())
else:
    print(f"File NOT found at: {file_path}")
