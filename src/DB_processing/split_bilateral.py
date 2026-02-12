from src.DB_processing.tools import append_audit
from src.DB_processing.database import DatabaseManager


def _determine_has_malignant(diagnosis):
    """Check if a diagnosis string contains MALIGNANT."""
    if diagnosis is None:
        return 0
    return 1 if 'MALIGNANT' in str(diagnosis).upper() else 0


def _determine_has_benign(diagnosis):
    """Check if a diagnosis string contains BENIGN."""
    if diagnosis is None:
        return 0
    return 1 if 'BENIGN' in str(diagnosis).upper() else 0


def _get_bilateral_image_lateralities(cursor):
    """Get image lateralities grouped by accession_number for all bilateral cases."""
    cursor.execute("""
        SELECT i.accession_number, UPPER(i.laterality)
        FROM Images i
        INNER JOIN StudyCases s ON s.accession_number = i.accession_number
        WHERE s.study_laterality = 'BILATERAL' AND s.modality = 'US'
    """)

    groups = {}
    for acc, lat in cursor.fetchall():
        if acc not in groups:
            groups[acc] = []
        groups[acc].append(lat)
    return groups


def split_bilateral_cases_in_db():
    """
    Split bilateral StudyCases into separate LEFT and RIGHT rows in the database.

    For each bilateral case:
    - If images exist for BOTH sides: split into two rows (ACC_L, ACC_R)
    - If images exist for only ONE side: convert to that side
    - If NO images exist: remove the case

    Child records (Images, Videos, Lesions) are reassigned to the correct split accession.
    """
    with DatabaseManager() as db:
        cursor = db.conn.cursor()

        # Ensure new columns exist (for databases created before this migration)
        cursor.execute("PRAGMA table_info(StudyCases)")
        existing_columns = {row[1] for row in cursor.fetchall()}
        if 'original_accession_number' not in existing_columns:
            cursor.execute("ALTER TABLE StudyCases ADD COLUMN original_accession_number TEXT")
            print("Added column 'original_accession_number' to StudyCases")
        if 'was_bilateral' not in existing_columns:
            cursor.execute("ALTER TABLE StudyCases ADD COLUMN was_bilateral INTEGER DEFAULT 0")
            print("Added column 'was_bilateral' to StudyCases")
        db.conn.commit()

        # Get all bilateral study cases
        cursor.execute("SELECT * FROM StudyCases WHERE study_laterality = 'BILATERAL' AND modality = 'US'")
        columns = [desc[0] for desc in cursor.description]
        bilateral_rows = [dict(zip(columns, row)) for row in cursor.fetchall()]

        if not bilateral_rows:
            print("No bilateral cases found to split.")
            return

        print(f"Found {len(bilateral_rows)} bilateral cases to process...")

        # Get image lateralities for all bilateral cases in one query
        image_groups = _get_bilateral_image_lateralities(cursor)

        split_count = 0
        converted_count = 0
        removed_count = 0

        for row in bilateral_rows:
            accession = row['accession_number']
            lateralities = image_groups.get(accession, [])

            if not lateralities:
                # No images — remove this case (CASCADE will clean up any child records)
                cursor.execute("DELETE FROM StudyCases WHERE accession_number = ?", (accession,))
                removed_count += 1
                continue

            has_left = any(lat == 'LEFT' for lat in lateralities)
            has_right = any(lat == 'RIGHT' for lat in lateralities)

            if has_left and has_right:
                # Split into LEFT and RIGHT
                _insert_split_row(cursor, columns, row, accession, 'LEFT')
                _insert_split_row(cursor, columns, row, accession, 'RIGHT')
                _reassign_child_records(cursor, accession, f"{accession}_L", 'LEFT')
                _reassign_child_records(cursor, accession, f"{accession}_R", 'RIGHT')
                # Delete original bilateral row
                cursor.execute("DELETE FROM StudyCases WHERE accession_number = ?", (accession,))
                split_count += 1
            elif has_left:
                _insert_split_row(cursor, columns, row, accession, 'LEFT')
                _reassign_child_records(cursor, accession, f"{accession}_L", 'LEFT')
                # Reassign any remaining child records (unknown laterality) to the LEFT side
                _reassign_remaining_child_records(cursor, accession, f"{accession}_L")
                cursor.execute("DELETE FROM StudyCases WHERE accession_number = ?", (accession,))
                converted_count += 1
            elif has_right:
                _insert_split_row(cursor, columns, row, accession, 'RIGHT')
                _reassign_child_records(cursor, accession, f"{accession}_R", 'RIGHT')
                _reassign_remaining_child_records(cursor, accession, f"{accession}_R")
                cursor.execute("DELETE FROM StudyCases WHERE accession_number = ?", (accession,))
                converted_count += 1
            else:
                # Images exist but none have LEFT/RIGHT laterality — remove
                cursor.execute("DELETE FROM StudyCases WHERE accession_number = ?", (accession,))
                removed_count += 1

        db.conn.commit()

        print(f"Bilateral split complete: {split_count} split into L+R, "
              f"{converted_count} converted to single-sided, "
              f"{removed_count} removed (no images or no laterality)")

        append_audit("bilateral_split.split_into_lr", split_count)
        append_audit("bilateral_split.converted_single_side", converted_count)
        append_audit("bilateral_split.removed", removed_count)


def _insert_split_row(cursor, columns, original_row, original_accession, laterality):
    """Insert a new StudyCases row for one side of a bilateral split."""
    suffix = '_L' if laterality == 'LEFT' else '_R'
    new_accession = f"{original_accession}{suffix}"

    # Determine diagnosis for this side
    if laterality == 'LEFT':
        diagnosis = original_row.get('left_diagnosis')
        has_malignant = _determine_has_malignant(diagnosis)
        has_benign = _determine_has_benign(diagnosis)
    else:
        diagnosis = original_row.get('right_diagnosis')
        has_malignant = _determine_has_malignant(diagnosis)
        has_benign = _determine_has_benign(diagnosis)

    # Build new row
    new_row = dict(original_row)
    new_row['accession_number'] = new_accession
    new_row['original_accession_number'] = original_accession
    new_row['was_bilateral'] = 1
    new_row['study_laterality'] = laterality
    new_row['has_malignant'] = has_malignant
    new_row['has_benign'] = has_benign

    # Clear non-relevant diagnosis
    if laterality == 'LEFT':
        new_row['right_diagnosis'] = None
        new_row['right_diagnosis_source'] = None
    else:
        new_row['left_diagnosis'] = None
        new_row['left_diagnosis_source'] = None

    # Filter to only columns that exist in the table
    # (exclude created_at to let default apply)
    insert_columns = [c for c in columns if c != 'created_at' and c in new_row]
    placeholders = ', '.join(['?'] * len(insert_columns))
    col_names = ', '.join(insert_columns)
    values = [new_row[c] for c in insert_columns]

    cursor.execute(f"INSERT INTO StudyCases ({col_names}) VALUES ({placeholders})", values)


def _reassign_child_records(cursor, old_accession, new_accession, laterality):
    """Reassign Images, Videos, and Lesions matching a specific laterality to the new accession."""
    # Reassign Images by laterality (case-insensitive match)
    cursor.execute("""
        UPDATE Images SET accession_number = ?
        WHERE accession_number = ? AND UPPER(laterality) = ?
    """, (new_accession, old_accession, laterality))

    # Reassign Videos by laterality (case-insensitive match)
    cursor.execute("""
        UPDATE Videos SET accession_number = ?
        WHERE accession_number = ? AND UPPER(laterality) = ?
    """, (new_accession, old_accession, laterality))

    # Reassign Lesions — route via their linked image's laterality
    cursor.execute("""
        UPDATE Lesions SET accession_number = ?
        WHERE accession_number = ? AND image_name IN (
            SELECT image_name FROM Images WHERE accession_number = ?
        )
    """, (new_accession, old_accession, new_accession))


def _reassign_remaining_child_records(cursor, old_accession, new_accession):
    """Reassign any remaining child records (unknown laterality) to the new accession."""
    cursor.execute("""
        UPDATE Images SET accession_number = ?
        WHERE accession_number = ?
    """, (new_accession, old_accession))

    cursor.execute("""
        UPDATE Videos SET accession_number = ?
        WHERE accession_number = ?
    """, (new_accession, old_accession))

    cursor.execute("""
        UPDATE Lesions SET accession_number = ?
        WHERE accession_number = ?
    """, (new_accession, old_accession))
