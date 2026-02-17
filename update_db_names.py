"""Update image_name and video images_path in the database to match
already-renamed files ({dicom_hash}.png / {dicom_hash})."""
import pandas as pd
from src.DB_processing.database import DatabaseManager

with DatabaseManager() as db:
    cursor = db.conn.cursor()

    # --- Images ---
    df = pd.read_sql_query("SELECT image_name, dicom_hash FROM Images", db.conn)
    img_map = {
        row['image_name']: f"{row['dicom_hash']}.png"
        for _, row in df.iterrows()
        if row['image_name'] != f"{row['dicom_hash']}.png"
    }
    print(f"Images to update: {len(img_map)}  (already correct: {len(df) - len(img_map)})")

    if img_map:
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

    # --- Videos ---
    vdf = pd.read_sql_query("SELECT images_path, dicom_hash FROM Videos", db.conn)
    vid_map = {
        row['images_path']: row['dicom_hash']
        for _, row in vdf.iterrows()
        if row['images_path'] != row['dicom_hash']
    }
    print(f"Videos to update: {len(vid_map)}  (already correct: {len(vdf) - len(vid_map)})")

    if vid_map:
        cursor.execute("CREATE TEMP TABLE vid_map (old_name TEXT PRIMARY KEY, new_name TEXT)")
        cursor.executemany("INSERT INTO vid_map VALUES (?, ?)", list(vid_map.items()))
        db.conn.commit()

        cursor.execute("""
            UPDATE Videos SET images_path = (
                SELECT new_name FROM vid_map WHERE vid_map.old_name = Videos.images_path
            ) WHERE images_path IN (SELECT old_name FROM vid_map)
        """)
        print(f"  Videos.images_path: {cursor.rowcount} rows")

        cursor.execute("DROP TABLE vid_map")
        db.conn.commit()

    print("\nDone!")
