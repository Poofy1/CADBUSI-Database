"""Rename all video folders to {dicom_hash}/ in-place.
Resumable - already-renamed frames are skipped. 32 threads."""
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from src.DB_processing.database import DatabaseManager
from tools.storage_adapter import rename_file, file_exists, list_files, StorageClient
from config import CONFIG

# Initialize storage
storage = StorageClient.get_instance(
    windir=CONFIG.get('WINDIR', ''),
    bucket_name=CONFIG.get('BUCKET', '')
)

VIDEO_DIR = f"{CONFIG['DATABASE_DIR']}videos/"

with DatabaseManager() as db:
    vdf = pd.read_sql_query("SELECT images_path, dicom_hash FROM Videos", db.conn)
    print(f"Total videos in DB: {len(vdf)}")

    vid_map = {}
    for _, row in vdf.iterrows():
        old_folder = row['images_path']
        new_folder = row['dicom_hash']
        if old_folder != new_folder:
            vid_map[old_folder] = new_folder

    print(f"Video folders to rename: {len(vid_map)}")
    print(f"Already correct: {len(vdf) - len(vid_map)}")

    if not vid_map:
        print("Nothing to rename.")
        exit()

    # Rename video folders (32 threads)
    print("\n--- Renaming video folders ---")
    errors = []

    def rename_folder(old_folder):
        new_folder = vid_map[old_folder]
        try:
            frames = list_files(f"{VIDEO_DIR}{old_folder}")
            for frame_path in frames:
                if storage.is_gcp:
                    frame_file = frame_path.split('/')[-1]
                else:
                    import os
                    frame_file = os.path.basename(frame_path)

                old_frame = frame_path if storage.is_gcp else f"{VIDEO_DIR}{old_folder}/{frame_file}"
                new_frame = f"{VIDEO_DIR}{new_folder}/{frame_file}"
                if file_exists(new_frame):
                    continue
                rename_file(old_frame, new_frame)
            return None
        except Exception as e:
            return (old_folder, str(e))

    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = {executor.submit(rename_folder, old): old for old in vid_map}
        with tqdm(total=len(futures), desc="Renaming video folders") as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    errors.append(result)
                pbar.update()

    if errors:
        print(f"{len(errors)} errors:")
        for name, err in errors[:10]:
            print(f"  {name}: {err}")

    # Update database
    print("\n--- Updating database ---")
    cursor = db.conn.cursor()

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
