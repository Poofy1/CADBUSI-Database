import csv

def filter_csv_based_on_malignancy(instance_labels, true_labels, output_csv_path):
    # Step 1: Read the second CSV and create a mapping of ID to Has_Malignant status
    id_to_malignancy_status = {}
    with open(true_labels, mode='r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            id_to_malignancy_status[row['ID']] = row['Has_Malignant'].lower() == 'true'

    # Step 2: Read the first CSV, filter rows based on the mapping
    filtered_rows = []
    with open(instance_labels, mode='r', newline='') as file:
        reader = csv.DictReader(file)
        headers = reader.fieldnames
        for row in reader:
            accession_number = row['Accession_Number']
            # Convert 'Malignant Lesion Present' to a boolean for comparison
            malignant_present = row['Malignant Lesion Present'].lower() == 'true'
            # Check if the row should be included based on the second CSV's mapping
            if id_to_malignancy_status.get(accession_number, False) == malignant_present:
                filtered_rows.append(row)

    # Step 3: Write the filtered rows into a new output CSV file
    with open(output_csv_path, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        writer.writerows(filtered_rows)

# Example usage
instance_labels = "D:/DATA/CASBUSI/labelbox_data/InstanceLabels.csv"
true_labels = "D:/DATA/CASBUSI/exports/export_11_11_2023/TrainData.csv"
output_csv_path = "D:/DATA/CASBUSI/labelbox_data/InstanceLabelsFiltered.csv"

filter_csv_based_on_malignancy(instance_labels, true_labels, output_csv_path)
