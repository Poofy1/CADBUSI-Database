import csv
import os
from collections import Counter

env = os.path.dirname(os.path.abspath(__file__))

def check_image_files(csv_file):
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        image_names = []
        for row in reader:
            image_name = row.get('ImageName')
            if not os.path.isfile(f'{env}/export/images/{image_name}'):
                print(f"Image file '{image_name}' does not exist or cannot be accessed.")
            image_names.append(image_name)

        duplicate_image_names = [name for name, count in Counter(image_names).items() if count > 1]
        if duplicate_image_names:
            print("Duplicate image names found:")
            for name in duplicate_image_names:
                print(f"Image name '{name}' is duplicated.")

# Usage example
csv_file_path = f'{env}/export/ImageData.csv'
check_image_files(csv_file_path)
