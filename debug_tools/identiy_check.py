import pandas as pd

# Read the CSV files
csv1 = pd.read_csv('D:/DATA/CASBUSI/exports/export_12_08_2023/ImageData.csv')
csv2 = pd.read_csv('D:/DATA/CASBUSI/exports/export_11_11_2023/ImageData.csv')

# Function to compare two dataframes
def are_identical(df1, df2):
    # Check if the columns are the same
    if list(df1.columns) != list(df2.columns):
        return False

    # Check if the number of rows is the same
    if len(df1) != len(df2):
        return False

    # Compare the data
    return df1.equals(df2)

# Check if the CSVs are identical
identical = are_identical(csv1, csv2)

print(f"The CSV files are {'identical' if identical else 'not identical'}.")
