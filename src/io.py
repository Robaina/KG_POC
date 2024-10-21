import os
import csv
from typing import Dict, List, Optional
from neo4j import Record
from pathlib import Path
import pandas as pd


def save_results_to_tsv(records: List[Record], output_file: str) -> None:
    """
    Save Neo4j query results to a single TSV file.
    This function takes a list of Neo4j Record objects and saves them to a TSV file.

    Args:
        records (List[Record]): A list of Neo4j Record objects.
        output_file (str): The full path to the output TSV file.

    Returns:
        None

    Raises:
        IOError: If there's an issue writing the file.
    """
    if not records:  # Return early if the list is empty
        print("No records to save.")
        return

    # Ensure the directory exists
    output_dir = os.path.dirname(output_file)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Get the column names from the first record
    columns = list(records[0].data().keys())

    try:
        with open(output_file, "w", newline="") as tsv_file:
            writer = csv.DictWriter(tsv_file, fieldnames=columns, delimiter="\t")

            # Write the header
            writer.writeheader()
            # Write the data
            for record in records:
                writer.writerow(record.data())
        print(f"Saved {len(records)} records to {output_file}")
    except IOError as e:
        print(f"Error writing to {output_file}: {e}")


def neo4j_records_to_dataframe(
    records: List[Dict], output_file: str = None
) -> pd.DataFrame:
    """
    Convert a list of Neo4j records to a pandas DataFrame with a header and optionally save it as a TSV file.

    Args:
        records (List[Dict]): A list of dictionaries, where each dictionary represents a Neo4j record.
        output_file (Optional[str]): The path to save the DataFrame as a TSV file. If None, the DataFrame is not saved.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the data from the Neo4j records with appropriate headers.

    Raises:
        ValueError: If the input records list is empty.
    """
    if not records:
        raise ValueError("The input records list is empty.")

    # Extract the keys from the first record to use as column names
    columns = list(records[0].keys())

    # Convert the list of dictionaries to a pandas DataFrame with specified column names
    df = pd.DataFrame(records, columns=columns)

    # If an output file is specified, save the DataFrame as a TSV file without index
    if output_file:
        df.to_csv(output_file, sep="\t", index=False)
        print(f"DataFrame saved to {output_file}")

    return df


def filter_ocean_data(
    file_path: str,
    depth_range: Optional[tuple[float, float]] = None,
    temperature_range: Optional[tuple[float, float]] = None,
    latitude_range: Optional[tuple[float, float]] = None,
    longitude_range: Optional[tuple[float, float]] = None,
) -> pd.DataFrame:
    """
    Filter rows of a TSV file based on ranges for depth, temperature, latitude, and longitude.

    Args:
        file_path (str): Path to the TSV file to be processed.
        depth_range (Optional[tuple[float, float]]): Min and max values for depth (meters).
        temperature_range (Optional[tuple[float, float]]): Min and max values for temperature (Celsius).
        latitude_range (Optional[tuple[float, float]]): Min and max values for latitude (degrees).
        longitude_range (Optional[tuple[float, float]]): Min and max values for longitude (degrees).

    Returns:
        pd.DataFrame: Filtered pandas DataFrame.

    Raises:
        ValueError: If any of the input ranges are invalid (min > max).
    """

    def validate_range(range_tuple: Optional[tuple[float, float]], name: str) -> None:
        if range_tuple is not None and range_tuple[0] > range_tuple[1]:
            raise ValueError(
                f"Invalid {name} range: min value should be less than or equal to max value"
            )

    # Validate input ranges
    validate_range(depth_range, "depth")
    validate_range(temperature_range, "temperature")
    validate_range(latitude_range, "latitude")
    validate_range(longitude_range, "longitude")

    # Read TSV file
    df = pd.read_csv(file_path, sep="\t", low_memory=False)

    # Apply filters
    if depth_range:
        df = df[
            (df["sample_depth"] >= depth_range[0])
            & (df["sample_depth"] <= depth_range[1])
        ]
    if temperature_range:
        df = df[
            (df["sample_temperature"] >= temperature_range[0])
            & (df["sample_temperature"] <= temperature_range[1])
        ]
    if latitude_range:
        df = df[
            (df["sample_latitude"] >= latitude_range[0])
            & (df["sample_latitude"] <= latitude_range[1])
        ]
    if longitude_range:
        df = df[
            (df["sample_longitude"] >= longitude_range[0])
            & (df["sample_longitude"] <= longitude_range[1])
        ]

    return df
