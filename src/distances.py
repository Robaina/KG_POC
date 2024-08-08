import os
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import itertools
from multiprocessing import Pool
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from typing import List, Tuple


def compute_fingerprint(compound):
    smiles = compound.get("smiles")
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return compound["id"], AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)


def compute_distance(pair):
    (id1, fp1), (id2, fp2) = pair
    similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
    return (id1, id2, 1 - similarity)


def compute_fingerprint_distances(
    compounds: list[dict], n_jobs=None
) -> list[tuple[str, str, float]]:
    # Compute fingerprints in parallel
    if n_jobs is None:
        n_jobs = os.cpu_count()
    with Pool(n_jobs) as pool:
        fingerprints = pool.map(compute_fingerprint, compounds)

    # Filter out None results
    fingerprints = [fp for fp in fingerprints if fp is not None]

    # Compute pairwise distances in parallel
    pairs = itertools.combinations(fingerprints, 2)
    with Pool(n_jobs) as pool:
        results = pool.map(compute_distance, pairs)

    return results


def store_distances_parquet(
    distances: List[Tuple[str, str, float]], output_file: str, chunk_size: int = 1000000
):
    """
    Store pre-computed pairwise distances in Parquet format.

    :param distances: List of tuples (compound1_id, compound2_id, distance)
    :param output_file: Path to the output Parquet file
    :param chunk_size: Number of rows to write in each chunk (for memory efficiency)
    """
    # Convert distances to a pandas DataFrame
    df = pd.DataFrame(distances, columns=["compound1", "compound2", "distance"])

    # Create a PyArrow Table
    table = pa.Table.from_pandas(df)

    # Write to Parquet file in chunks
    with pq.ParquetWriter(output_file, table.schema) as writer:
        for i in range(0, len(df), chunk_size):
            chunk = table.slice(i, chunk_size)
            writer.write_table(chunk)

    print(f"Stored {len(distances)} pairwise distances in {output_file}")


def read_distance_parquet(
    file_path: str, compound1_id: str, compound2_id: str
) -> float:
    """
    Read a specific distance from the Parquet file.

    :param file_path: Path to the Parquet file
    :param compound1_id: ID of the first compound
    :param compound2_id: ID of the second compound
    :return: Distance between the two compounds
    """
    filters = [("compound1", "=", compound1_id), ("compound2", "=", compound2_id)]

    try:
        df = pd.read_parquet(file_path, filters=filters)
        if len(df) > 0:
            return df.iloc[0]["distance"]

        # If not found, try swapping compound1 and compound2
        filters = [("compound1", "=", compound2_id), ("compound2", "=", compound1_id)]
        df = pd.read_parquet(file_path, filters=filters)
        if len(df) > 0:
            return df.iloc[0]["distance"]

        raise ValueError(
            f"No distance found for compounds {compound1_id} and {compound2_id}"
        )
    except Exception as e:
        print(f"Error reading distance: {e}")
        return None
