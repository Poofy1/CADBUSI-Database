"""Rename all images to {dicom_hash}.png in-place.
Resumable - already-renamed files are skipped. 32 threads."""
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from src.DB_processing.database import DatabaseManager
from tools.storage_adapter import rename_file, file_exists, StorageClient
from config import CONFIG

# Initialize storage
StorageClient.get_instance(
    windir=CONFIG.get('WINDIR', ''),
    bucket_name=CONFIG.get('BUCKET', '')
)

IMAGE_DIR = f"{CONFIG['DATABASE_DIR']}images/"

with DatabaseManager() as db:
    df = pd.read_sql_query("SELECT image_name, dicom_hash FROM Images", db.conn)
    print(f"Total images in DB: {len(df)}")

    img_map = {}
    for _, row in df.iterrows():
        old_name = row['image_name']
        new_name = f"{row['dicom_hash']}.png"
        if old_name != new_name:
            img_map[old_name] = new_name

    print(f"Images to rename: {len(img_map)}")
    print(f"Already correct: {len(df) - len(img_map)}")

    if not img_map:
        print("Nothing to rename.")
        exit()

    # Rename files (32 threads)
    print("\n--- Renaming images ---")
    errors = []
    skipped = 0

    def rename_one(old_name):
        new_name = img_map[old_name]
        new_path = f"{IMAGE_DIR}{new_name}"
        if file_exists(new_path):
            return "skipped"
        try:
            rename_file(f"{IMAGE_DIR}{old_name}", new_path)
            return None
        except Exception as e:
            return (old_name, str(e))

    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = {executor.submit(rename_one, old): old for old in img_map}
        with tqdm(total=len(futures), desc="Renaming images") as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result == "skipped":
                    skipped += 1
                elif result is not None:
                    errors.append(result)
                pbar.update()

    if skipped:
        print(f"Skipped (already renamed): {skipped}")
    if errors:
        print(f"{len(errors)} errors:")
        for name, err in errors[:10]:
            print(f"  {name}: {err}")

    # Update database
    print("\n--- Updating database ---")
    cursor = db.conn.cursor()
    cursor.execute("PRAGMA foreign_keys = OFF")

    cursor.execute("CREATE TEMP TABLE img_map (old_name TEXT PRIMARY KEY, new_name TEXT)")
    cursor.executemany("INSERT INTO img_map VALUES (?, ?)", list(img_map.items()))
    db.conn.commit()

    for table, col in [
        ("Images", "image_name"),
        ("Images", "closest_fn"),
        ("Images", "inpainted_from"),
        ("BadImages", "image_name"),
        ("Lesions", "image_name"),
        ("CaliperPairs", "caliper_image_name"),
        ("CaliperPairs", "clean_image_name"),
        ("CaliperPairs", "inpainted_image_name"),
    ]:
        cursor.execute(f"""
            UPDATE {table} SET {col} = (
                SELECT new_name FROM img_map WHERE img_map.old_name = {table}.{col}
            ) WHERE {col} IN (SELECT old_name FROM img_map)
        """)
        print(f"  {table}.{col}: {cursor.rowcount} rows")

    cursor.execute("DROP TABLE img_map")
    cursor.execute("PRAGMA foreign_keys = ON")
    db.conn.commit()

    print("\nDone!")
