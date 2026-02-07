import sqlite3
import pandas as pd

# Paths
instance_labels_path = r"C:\Users\Tristan\Desktop\labelbox_data_InstanceLabels.csv"
crop_data_path = r"C:\Users\Tristan\Desktop\labelbox_data_crop_data_labeled_1_19_26.csv"
db_path = r"C:\Users\Tristan\Desktop\labelbox.db"

# Create database
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Schema
cursor.execute("""
CREATE TABLE IF NOT EXISTS image_labels (
    dicom_hash TEXT PRIMARY KEY,
    reject INTEGER,
    only_normal INTEGER,
    cyst INTEGER,
    benign INTEGER,
    malignant INTEGER,
    quality TEXT
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS mask_labels (
    dicom_hash TEXT PRIMARY KEY,
    mask_image TEXT,
    quality TEXT,
    FOREIGN KEY (dicom_hash) REFERENCES image_labels(dicom_hash)
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS box_labels (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dicom_hash TEXT,
    x1 INTEGER,
    y1 INTEGER,
    x2 INTEGER,
    y2 INTEGER,
    quality TEXT,
    FOREIGN KEY (dicom_hash) REFERENCES image_labels(dicom_hash)
)
""")

cursor.execute("CREATE INDEX IF NOT EXISTS idx_box_dicom_hash ON box_labels(dicom_hash)")

# Load and insert data
df_instance = pd.read_csv(instance_labels_path)
df_instance = df_instance.drop_duplicates(subset=['dicom_hash'], keep='first')

# Insert mask labels first (before filtering)
df_masks = df_instance[['dicom_hash', 'mask_image']].dropna(subset=['mask_image']).copy()
df_masks['quality'] = 'gold'
df_masks.to_sql('mask_labels', conn, if_exists='replace', index=False)

# Filter out null reject, then insert image labels
df_instance_filtered = df_instance.dropna(subset=['reject'])
df_image_labels = df_instance_filtered[['dicom_hash', 'reject', 'only_normal', 'cyst', 'benign', 'malignant']].copy()
df_image_labels['quality'] = 'gold'
df_image_labels.to_sql('image_labels', conn, if_exists='replace', index=False)

# Insert box labels
df_crop = pd.read_csv(crop_data_path)
df_box_clean = df_crop[['dicom_hash', 'x1', 'y1', 'x2', 'y2']].copy()
df_box_clean['quality'] = 'gold'
df_box_clean.to_sql('box_labels', conn, if_exists='replace', index=False)

conn.commit()
conn.close()

print(f"Database created: {db_path}")