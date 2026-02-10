import hashlib
from src.DB_processing.tools import append_audit
from src.DB_processing.database import DatabaseManager


def get_split_from_hash(patient_id, val_split, test_split):
    """Deterministically assign split based on patient_id hash."""
    hash_bytes = hashlib.md5(str(patient_id).encode()).digest()
    hash_int = int.from_bytes(hash_bytes[:8], byteorder='big')
    hash_float = hash_int / (2**64)

    if hash_float < test_split:
        return 2  # Test
    elif hash_float < test_split + val_split:
        return 1  # Validation
    else:
        return 0  # Training


def PerformSplit():
    val_split = 0.15
    test_split = 0.15

    with DatabaseManager() as db:
        df = db.get_study_cases_dataframe()

        # Compute split for each row
        df['valid'] = df['patient_id'].apply(
            lambda pid: get_split_from_hash(pid, val_split, test_split)
        )

        # Count samples and patients in each split
        train_samples = (df['valid'] == 0).sum()
        val_samples = (df['valid'] == 1).sum()
        test_samples = (df['valid'] == 2).sum()

        train_patients = df[df['valid'] == 0]['patient_id'].nunique()
        val_patients = df[df['valid'] == 1]['patient_id'].nunique()
        test_patients = df[df['valid'] == 2]['patient_id'].nunique()

        print(f"Split completed: {train_samples} training, {val_samples} validation, {test_samples} test samples")
        print(f"Patient split: {train_patients} training, {val_patients} validation, {test_patients} test patients")

        append_audit("export.train_patients", train_patients)
        append_audit("export.val_patients", val_patients)
        append_audit("export.test_patients", test_patients)

        # Write back to database
        update_data = [
            {'accession_number': row['accession_number'], 'valid': row['valid']}
            for _, row in df.iterrows()
        ]
        db.insert_study_cases_batch(update_data, update_only=True)