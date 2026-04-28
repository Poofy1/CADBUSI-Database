"""
One-shot temp script: walks raw DICOMs and writes a (dicom_hash, acquisition_time) CSV.

Reads the AcquisitionTime tag (0008,0032) directly from the un-anonymized DICOMs in
CONFIG["UNZIPPED_DICOMS"]. For multi-frame files, AcquisitionTime represents the start
of acquisition (effectively the first frame) — which is what we want.

Output: acquisition_times.csv in the current working directory.
The user merges this into the existing DB themselves by joining on dicom_hash.
"""
import os
import io
import warnings
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

import pydicom
import pandas as pd
from tqdm import tqdm

from tools.storage_adapter import StorageClient, read_binary, get_files_by_extension
from config import CONFIG

logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings('ignore', category=UserWarning, message='.*Invalid value for VR UI.*')

OUTPUT_CSV = "acquisition_times.csv"
BATCH_SIZE = 200


def extract_one(dcm_path):
    try:
        data = read_binary(dcm_path)
        if data is None:
            return None
        ds = pydicom.dcmread(
            io.BytesIO(data),
            force=True,
            specific_tags=['AcquisitionTime'],
            stop_before_pixels=True,
        )
        acq = getattr(ds, 'AcquisitionTime', '') or ''
        dicom_hash = os.path.splitext(os.path.basename(dcm_path))[0]
        return (dicom_hash, str(acq))
    except Exception as e:
        print(f"Failed {dcm_path}: {e}")
        return None


def process_batch(batch):
    return [r for r in (extract_one(p) for p in batch) if r is not None]


def main():
    StorageClient.get_instance(CONFIG["WINDIR"], CONFIG["BUCKET"])
    raw_dir = CONFIG["UNZIPPED_DICOMS"]
    files = get_files_by_extension(raw_dir, '.dcm')
    print(f"Found {len(files)} DICOMs in {raw_dir}")

    if not files:
        print("No DICOM files to process. Exiting.")
        return

    batches = [files[i:i + BATCH_SIZE] for i in range(0, len(files), BATCH_SIZE)]
    rows = []
    workers = min(32, multiprocessing.cpu_count())

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(process_batch, b) for b in batches]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Reading DICOMs"):
            rows.extend(f.result())

    df = pd.DataFrame(rows, columns=['dicom_hash', 'acquisition_time'])
    df = df.drop_duplicates(subset='dicom_hash')
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {len(df)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
