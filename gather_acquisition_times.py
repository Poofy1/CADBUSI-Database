"""
One-shot temp script: walks raw DICOMs and writes a (dicom_hash, acquisition_time, source_tag) CSV.

Tries DICOM time tags in fallback order:
    AcquisitionTime (0008,0032)
  → ContentTime    (0008,0033)
  → SeriesTime     (0008,0031)
  → StudyTime      (0008,0030)

Output time format: HH:MM:SS.fff (DICOM TM 'HHMMSS[.FFFFFF]' is reformatted).
Reads from CONFIG["storage"]["download_path"] (same source as Parse_Dicom_Files).

Output: acquisition_times.csv in the current working directory.
DICOMs that yield no usable time are still written (with empty acquisition_time
and source_tag) and the reason is printed to stdout.
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

TIME_TAG_FALLBACK = ('AcquisitionTime', 'ContentTime', 'SeriesTime', 'StudyTime')


def format_dicom_tm(tm):
    """Format DICOM TM 'HHMMSS[.FFFFFF]' as 'HH:MM:SS[.fff]'. Returns '' if unparseable."""
    if not tm:
        return ''
    s = str(tm).strip()
    if '.' in s:
        hms, frac = s.split('.', 1)
    else:
        hms, frac = s, ''
    if len(hms) < 6 or not hms[:6].isdigit():
        return ''
    h, m, sec = hms[0:2], hms[2:4], hms[4:6]
    if frac:
        ms = (frac + '000')[:3]
        return f"{h}:{m}:{sec}.{ms}"
    return f"{h}:{m}:{sec}"


def extract_one(dcm_path):
    """Returns (dicom_hash, acquisition_time, source_tag, reason).
    `reason` is empty on success, otherwise describes why no time was extracted."""
    dicom_hash = os.path.splitext(os.path.basename(dcm_path))[0]
    try:
        data = read_binary(dcm_path)
        if data is None:
            return (dicom_hash, '', '', 'read_failed')
        ds = pydicom.dcmread(
            io.BytesIO(data),
            force=True,
            specific_tags=list(TIME_TAG_FALLBACK),
            stop_before_pixels=True,
        )
    except Exception as e:
        return (dicom_hash, '', '', f'parse_error: {e}')

    raw_seen = []
    for tag in TIME_TAG_FALLBACK:
        raw = getattr(ds, tag, '') or ''
        if raw:
            formatted = format_dicom_tm(str(raw))
            if formatted:
                return (dicom_hash, formatted, tag, '')
            raw_seen.append((tag, str(raw)))

    if raw_seen:
        return (dicom_hash, '', '', f'all_time_tags_unparseable: {raw_seen}')
    return (dicom_hash, '', '', 'no_time_tags_present')


def process_batch(batch):
    return [extract_one(p) for p in batch]


def main():
    StorageClient.get_instance(CONFIG["WINDIR"], CONFIG["BUCKET"])
    raw_dir = f'{CONFIG["storage"]["download_path"]}/'
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
            for r in f.result():
                if r is None:
                    continue
                rows.append(r)
                # r = (hash, time, tag, reason); print on failure
                if not r[1]:
                    print(f"  [no time] {r[0]}: {r[3]}")

    df = pd.DataFrame(rows, columns=['dicom_hash', 'acquisition_time', 'source_tag', 'reason'])
    df = df.drop_duplicates(subset='dicom_hash')

    # Summary
    total = len(df)
    filled = (df['acquisition_time'] != '').sum()
    pct = (filled / total * 100) if total else 0.0
    print(f"\nSummary: {filled:,}/{total:,} got an acquisition_time ({pct:.2f}%)")
    if filled:
        print("By source tag:")
        for tag, count in df.loc[df['acquisition_time'] != '', 'source_tag'].value_counts().items():
            print(f"  {tag}: {count:,}")
    missing = df[df['acquisition_time'] == '']
    if not missing.empty:
        print("Missing-reason breakdown:")
        # collapse the 'all_time_tags_unparseable: [...]' detail to just its prefix
        reasons = missing['reason'].str.split(':').str[0]
        for reason, count in reasons.value_counts().items():
            print(f"  {reason}: {count:,}")

    df[['dicom_hash', 'acquisition_time', 'source_tag']].to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {len(df)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
