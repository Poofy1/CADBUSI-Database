import pandas as pd

def process_hash_comparison():
    # File paths
    csv1_path = "D:/DATA/CASBUSI/hash_translation4.csv"
    csv2_path = "C:/Users/Tristan/Desktop/hash_translation.csv"
    output_path = "C:/Users/Tristan/Desktop/hash_translation5.csv"
    
    try:
        # Read the CSV files
        print("Reading CSV files...")
        df1 = pd.read_csv(csv1_path)
        df2 = pd.read_csv(csv2_path)
        
        print(f"CSV1 loaded: {len(df1)} rows")
        print(f"CSV2 loaded: {len(df2)} rows")
        
        # Create a set of ContentHash values from csv2 for efficient lookup
        csv2_hashes = set(df2['ContentHash'])
        print(f"Unique hashes in CSV2: {len(csv2_hashes)}")
        
        # Add the new column 'Exists_In_New' based on whether ContentHash exists in csv2
        df1['Exists_In_New'] = df1['ContentHash'].isin(csv2_hashes)
        
        # Calculate the percentage of matching rows
        matching_count = df1['Exists_In_New'].sum()
        total_count = len(df1)
        percentage = (matching_count / total_count) * 100
        
        print(f"\nResults:")
        print(f"Total rows in CSV1: {total_count}")
        print(f"Matching rows: {matching_count}")
        print(f"Percentage of CSV1 rows with matching ContentHash in CSV2: {percentage:.2f}%")
        
        # Save the result to a new CSV file
        df1.to_csv(output_path, index=False)
        print(f"\nOutput saved to: {output_path}")
        
        # Display some sample results
        print(f"\nSample of results:")
        print(df1[['ImageName', 'ContentHash', 'Exists_In_New']].head(10))
        
        return percentage, matching_count, total_count
        
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
    except KeyError as e:
        print(f"Error: Column not found - {e}")
        print("Please check that the CSV files have the expected column names.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    process_hash_comparison()