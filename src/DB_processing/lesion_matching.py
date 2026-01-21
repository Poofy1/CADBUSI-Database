"""
Lesion Matching Module

Matches detected lesions from Images with lesion descriptions from StudyCases.
Populates the Lesions table with clock position, distance, and description metadata.
"""

import re
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
from src.DB_processing.database import DatabaseManager


def parse_lesion_descriptions(description_text: str) -> List[Dict]:
    """
    Parse lesion descriptions from the StudyCases.lesion_descriptions field.

    Expected format: [clock_pos, distance, sizing, description], [clock_pos, distance, sizing, description], ...
    Example: [1:00, 5cm, 2.9cm, irregular hypoechoic mass with obscured borders], [2:00, 7cm, 1.2cm, irregular hypoechoic mass]

    Returns:
        List of dicts with keys: clock, distance_cm, sizing_cm, description
    """
    if not description_text or pd.isna(description_text):
        return []

    lesions = []

    # Find all bracketed sections: [...]
    pattern = r'\[([^\]]+)\]'
    matches = re.findall(pattern, description_text)

    for match in matches:
        # Split by comma, but be careful with commas in the description
        parts = match.split(',')

        if len(parts) < 4:
            continue  # Invalid format

        # Extract the 4 parts
        clock_pos = parts[0].strip()
        distance = parts[1].strip()
        sizing = parts[2].strip()
        description = ','.join(parts[3:]).strip()  # Rejoin any remaining parts as description

        # Convert distance to cm (handle formats like "5cm", "5 cm", "50mm", "50 mm")
        distance_cm = parse_distance_to_cm(distance)

        # Convert sizing to cm (handle formats like "2.9cm", "29mm", "10mm")
        sizing_cm = parse_distance_to_cm(sizing)

        lesions.append({
            'clock': clock_pos,
            'distance_cm': distance_cm,
            'sizing_cm': sizing_cm,
            'description': description
        })

    return lesions


def parse_distance_to_cm(distance_str: str) -> Optional[float]:
    """
    Parse distance string to centimeters.

    Handles formats like:
    - "5cm", "5 cm", "5.2cm"
    - "50mm", "50 mm", "5.2mm"
    - "5" (assumes cm)

    Returns:
        Distance in cm, or None if parsing fails
    """
    if not distance_str:
        return None

    # Remove whitespace
    distance_str = distance_str.strip().lower()

    # Try to extract number and unit
    match = re.match(r'([\d.]+)\s*(cm|mm)?', distance_str)

    if match:
        value = float(match.group(1))
        unit = match.group(2)

        if unit == 'mm':
            return value / 10  # Convert mm to cm
        else:
            return value  # Assume cm if no unit or unit is cm

    return None


def parse_clock_position(clock_str: str) -> Optional[int]:
    """
    Parse clock position string to integer hour (1-12).

    Handles formats like:
    - "1:00", "1", "01:00", "1 o'clock"

    Returns:
        Hour (1-12), or None if parsing fails
    """
    if not clock_str or pd.isna(clock_str):
        return None

    clock_str = str(clock_str).strip().lower()

    # Try to extract hour from "HH:MM" format
    match = re.match(r'(\d+):?\d*', clock_str)

    if match:
        hour = int(match.group(1))
        # Normalize to 1-12 range
        if 1 <= hour <= 12:
            return hour
        elif hour == 0:
            return 12

    return None


def calculate_clock_distance(clock1: Optional[int], clock2: Optional[int]) -> float:
    """
    Calculate the minimum distance between two clock positions (in hours).

    Args:
        clock1: Hour (1-12)
        clock2: Hour (1-12)

    Returns:
        Minimum distance in hours (0-6)
    """
    if clock1 is None or clock2 is None:
        return float('inf')

    # Calculate direct distance
    direct_dist = abs(clock1 - clock2)

    # Calculate wraparound distance
    wraparound_dist = 12 - direct_dist

    # Return minimum
    return min(direct_dist, wraparound_dist)


def match_lesion_to_description(
    image_clock: str,
    image_distance: float,
    lesion_descriptions: List[Dict],
    clock_tolerance: float = 1.5,
    distance_tolerance_cm: float = 2.0
) -> Optional[Dict]:
    """
    Find the best matching lesion description for an image.

    Args:
        image_clock: Clock position from Images table (e.g., "1:00", "2")
        image_distance: Distance from nipple in cm (Images.nipple_dist)
        lesion_descriptions: List of parsed lesion descriptions
        clock_tolerance: Maximum clock difference in hours (default 1.5 hours)
        distance_tolerance_cm: Maximum distance difference in cm (default 2.0 cm)

    Returns:
        Best matching lesion description dict, or None if no match found
    """
    if not lesion_descriptions:
        return None

    # Parse image clock position
    image_clock_parsed = parse_clock_position(image_clock)

    # Find candidates within tolerance
    candidates = []

    for desc in lesion_descriptions:
        desc_clock_parsed = parse_clock_position(desc['clock'])
        desc_distance = desc['distance_cm']

        # Check clock position match
        clock_dist = calculate_clock_distance(image_clock_parsed, desc_clock_parsed)

        # Check distance match
        if image_distance is not None and desc_distance is not None:
            distance_diff = abs(image_distance - desc_distance)
        else:
            distance_diff = float('inf')

        # Check if within tolerance
        if clock_dist <= clock_tolerance and distance_diff <= distance_tolerance_cm:
            candidates.append({
                'description': desc,
                'clock_diff': clock_dist,
                'distance_diff': distance_diff,
                'score': clock_dist + (distance_diff / 2)  # Weighted score
            })

    # Return best match (lowest score)
    if candidates:
        candidates.sort(key=lambda x: x['score'])
        return candidates[0]['description']

    return None


def match_lesion_measurement(
    lesion_measurement_cm: Optional[float],
    sizing_cm: Optional[float],
    tolerance_cm: float = 0.5
) -> bool:
    """
    Check if a lesion measurement matches the sizing from description.

    Args:
        lesion_measurement_cm: Measured lesion size from Lesions table
        sizing_cm: Expected size from lesion description
        tolerance_cm: Tolerance in cm (default 0.5 cm)

    Returns:
        True if measurements match within tolerance
    """
    if lesion_measurement_cm is None or sizing_cm is None:
        return False

    diff = abs(lesion_measurement_cm - sizing_cm)
    return diff <= tolerance_cm


def Match_Lesions():
    """
    Main function to match lesion descriptions with images and create/update lesion records.

    Process:
    1. Load Images, StudyCases, and Lesions data
    2. Parse lesion descriptions from StudyCases
    3. For each image with clock_pos and nipple_dist, find matching lesion descriptions
    4. For each match:
       - If existing lesion with matching measurement exists, update it with description
       - Otherwise, create new lesion record with description data
    5. Update/Insert Lesions table accordingly
    """
    print("="*60)
    print("LESION MATCHING")
    print("="*60)

    with DatabaseManager() as db:
        # Load data
        print("Loading data from database...")
        images_df = db.get_images_dataframe()
        study_cases_df = db.get_study_cases_dataframe()
        lesions_df = pd.read_sql_query("SELECT * FROM Lesions", db.conn)

        print(f"Found {len(images_df)} images")
        print(f"Found {len(study_cases_df)} study cases")
        print(f"Found {len(lesions_df)} existing lesions")

        # Parse lesion descriptions
        print("\nParsing lesion descriptions...")
        tqdm.pandas(desc="Parsing descriptions")
        study_cases_df['parsed_lesions'] = study_cases_df['lesion_descriptions'].progress_apply(parse_lesion_descriptions)

        # Count total parsed lesions
        total_parsed = sum(len(lesions) for lesions in study_cases_df['parsed_lesions'])
        print(f"Parsed {total_parsed} lesion descriptions from study cases")

        # Match lesions by iterating through images
        print("\nMatching images to lesion descriptions...")
        updated_count = 0
        created_count = 0
        updated_lesions = []
        new_lesions = []
        matched_images = set()  # Track images that got matched

        # Filter images that have clock_pos and nipple_dist
        valid_images = images_df[
            images_df['clock_pos'].notna() &
            images_df['nipple_dist'].notna()
        ]

        print(f"Found {len(valid_images)} images with clock position and distance")

        # Pre-build lookup dictionaries for O(1) access
        study_cases_by_accession = {}
        for _, row in study_cases_df.iterrows():
            study_cases_by_accession[row['accession_number']] = row

        lesions_by_image = {}
        for _, row in lesions_df.iterrows():
            img_name = row['image_name']
            if img_name not in lesions_by_image:
                lesions_by_image[img_name] = []
            lesions_by_image[img_name].append(row)

        for _, image_row in tqdm(valid_images.iterrows(), total=len(valid_images), desc="Processing images"):
            image_name = image_row['image_name']
            accession_number = image_row['accession_number']
            patient_id = image_row['patient_id']
            image_clock = image_row['clock_pos']
            image_distance = image_row['nipple_dist']

            # Find the corresponding study case (O(1) lookup)
            if accession_number not in study_cases_by_accession:
                continue

            study_row = study_cases_by_accession[accession_number]
            parsed_lesions = study_row['parsed_lesions']

            if not parsed_lesions:
                continue

            # Find ALL matching lesion descriptions (could be multiple lesions per image)
            for matched_desc in parsed_lesions:
                # Check if this description matches the image location
                desc_clock_parsed = parse_clock_position(matched_desc['clock'])
                desc_distance = matched_desc['distance_cm']

                image_clock_parsed = parse_clock_position(image_clock)

                # Check if within tolerance
                clock_dist = calculate_clock_distance(image_clock_parsed, desc_clock_parsed)
                if image_distance is not None and desc_distance is not None:
                    distance_diff = abs(image_distance - desc_distance)
                else:
                    distance_diff = float('inf')

                # Default tolerances
                clock_tolerance = 1.5
                distance_tolerance_cm = 2.0

                if clock_dist <= clock_tolerance and distance_diff <= distance_tolerance_cm:
                    # This description matches the image
                    matched_images.add(image_name)  # Track this image as matched

                    # Check if there's an existing lesion for this image that matches the measurement (O(1) lookup)
                    existing_lesions = lesions_by_image.get(image_name, [])

                    matched_existing = False
                    for existing_lesion in existing_lesions:
                        existing_measurement = existing_lesion['lesion_measurement_cm']

                        # Check if measurement matches
                        if match_lesion_measurement(
                            lesion_measurement_cm=existing_measurement,
                            sizing_cm=matched_desc['sizing_cm'],
                            tolerance_cm=0.5
                        ):
                            # Update this existing lesion
                            update_data = {
                                'lesion_id': existing_lesion['lesion_id'],
                                'clock': matched_desc['clock'],
                                'distance_cm': matched_desc['distance_cm'],
                                'description': matched_desc['description'],
                                'parsed_lesion_measurement_cm': matched_desc['sizing_cm']
                            }
                            updated_lesions.append(update_data)
                            updated_count += 1
                            matched_existing = True
                            break

                    if not matched_existing:
                        # Create new lesion record with description data
                        new_lesion = {
                            'accession_number': accession_number,
                            'patient_id': patient_id,
                            'image_name': image_name,
                            'lesion_measurement_cm': None,  # No detected measurement
                            'parsed_lesion_measurement_cm': matched_desc['sizing_cm'],
                            'clock': matched_desc['clock'],
                            'distance_cm': matched_desc['distance_cm'],
                            'description': matched_desc['description']
                        }
                        new_lesions.append(new_lesion)
                        created_count += 1

        # Update existing lesions
        if updated_lesions:
            print(f"\nUpdating {len(updated_lesions)} existing lesions...")
            cursor = db.conn.cursor()

            for update in tqdm(updated_lesions, desc="Updating lesions"):
                cursor.execute("""
                    UPDATE Lesions
                    SET clock = ?,
                        distance_cm = ?,
                        description = ?,
                        parsed_lesion_measurement_cm = ?
                    WHERE lesion_id = ?
                """, (
                    update['clock'],
                    update['distance_cm'],
                    update['description'],
                    update['parsed_lesion_measurement_cm'],
                    update['lesion_id']
                ))

            db.conn.commit()
            print(f"Successfully updated {len(updated_lesions)} lesions")

        # Insert new lesions
        if new_lesions:
            print(f"\nCreating {len(new_lesions)} new lesion records...")
            db.insert_lesions_batch(new_lesions)
            print(f"Successfully created {len(new_lesions)} new lesions")

        # Print summary statistics
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"Images with clock/distance data: {len(valid_images)}")
        print(f"Images matched with descriptions: {len(matched_images)}")
        if len(valid_images) > 0:
            match_percentage = (len(matched_images) / len(valid_images)) * 100
            print(f"Image match rate: {match_percentage:.1f}%")
        print(f"Lesion descriptions parsed: {total_parsed}")
        print(f"Existing lesions updated: {updated_count}")
        print(f"New lesions created: {created_count}")
        print(f"Total lesions processed: {updated_count + created_count}")
        print("="*60)


if __name__ == "__main__":
    Match_Lesions()
