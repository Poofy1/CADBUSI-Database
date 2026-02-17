"""Copy the main CADBUSI database, then populate label tables from a labels .db file."""
import shutil
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.DB_processing.database import DatabaseManager

INPUT_DB = r"C:\Users\Tristan\Desktop\cadbusi.db"
LABELS_DB = r"C:\Users\Tristan\Desktop\new_labels_v2.db"
OUTPUT_DB = r"C:\Users\Tristan\Desktop\cadbusi2.db"


def apply_labels():
    # Copy the full database so we don't modify the original
    print(f"Copying {INPUT_DB} -> {OUTPUT_DB}...")
    shutil.copy2(INPUT_DB, OUTPUT_DB)

    # Use DatabaseManager to create label tables via create_schema()
    db = DatabaseManager()
    db.db_file = OUTPUT_DB
    db.connect()
    db.create_schema()
    db.conn.execute("PRAGMA foreign_keys = OFF")
    cursor = db.conn.cursor()
    cursor.execute("ATTACH DATABASE ? AS labels", (LABELS_DB,))

    cursor.execute("""
        INSERT OR REPLACE INTO ImageLabels (dicom_hash, reject, only_normal, cyst, benign, malignant, quality, version)
        SELECT dicom_hash, reject, only_normal, cyst, benign, malignant, quality, version
        FROM labels.image_labels
    """)
    print(f"ImageLabels: {cursor.rowcount} rows")

    cursor.execute("""
        INSERT INTO LesionLabels (dicom_hash, x1, y1, x2, y2, mask_image, quality, version)
        SELECT dicom_hash, x1, y1, x2, y2, mask_image, quality, version
        FROM labels.lesion_labels
    """)
    print(f"LesionLabels: {cursor.rowcount} rows")

    cursor.execute("""
        INSERT INTO RegionLabels (dicom_hash, crop_x, crop_y, crop_h, crop_w, us_polygon, debris_polygon, version)
        SELECT dicom_hash, crop_x, crop_y, crop_h, crop_w, us_polygon, debris_polygon, version
        FROM labels.region_labels
    """)
    print(f"RegionLabels: {cursor.rowcount} rows")

    db.conn.commit()
    cursor.execute("DETACH DATABASE labels")
    db.close()
    print(f"\nOutput: {OUTPUT_DB}")


if __name__ == "__main__":
    apply_labels()
