from __future__ import annotations
import os
import sys
from collections import defaultdict
import pyfastx
import json
import itertools
from Bio import SeqIO
from rdkit import Chem, RDLogger
from rdkit.Chem import DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols
import pandas as pd
from typing import List, Dict

RDLogger.DisableLog("rdApp.*")


def extract_protein_sequences(
    file_path: str, sequence_ids: List[str]
) -> Dict[str, str]:
    """
    Extract protein sequences from a given file based on a list of sequence IDs.

    Args:
        file_path (str): The path to the file containing protein sequences.
        sequence_ids (List[str]): A list of sequence IDs to extract.

    Returns:
        Dict[str, str]: A dictionary where keys are sequence IDs and values are the corresponding protein sequences.
    """
    sequences = {}
    with open(file_path, "r") as file:
        current_id = None
        current_sequence = []
        for line in file:
            if line.startswith(">"):
                if current_id and current_id in sequence_ids:
                    sequences[current_id] = "".join(current_sequence)
                current_id = line.split()[0][1:]
                current_sequence = []
            elif current_id in sequence_ids:
                current_sequence.append(line.strip())
        if current_id and current_id in sequence_ids:
            sequences[current_id] = "".join(current_sequence)
    return sequences


def parse_reaction_equation(reaction_dict):
    equation = reaction_dict.get("equation", "")
    substrates, products = [], []

    # Splitting the equation into substrates and products
    if "<=>" in equation:
        left_side, right_side = equation.split("<=>")
    elif "=>" in equation:
        left_side, right_side = equation.split("=>")
    elif "<=" in equation:
        right_side, left_side = equation.split("<=")
    else:
        # Invalid or unsupported reaction format
        return substrates, products

    def parse_compounds(compound_list):
        parsed_compounds = []
        for part in compound_list.split("+"):
            compound_info = part.strip().split(" ")
            compound_id = compound_info[1].split("[")[
                0
            ]  # Remove '[0]' from the compound ID
            stoichiometry_str = compound_info[0][1:]  # Extract the stoichiometry part
            stoichiometry = float("".join(filter(str.isdigit, stoichiometry_str)))
            parsed_compounds.append((compound_id, stoichiometry))
        return parsed_compounds

    # Extracting compound IDs and coefficients from each side
    substrates = parse_compounds(left_side)
    products = parse_compounds(right_side)

    return substrates, products


def add_substrate_product_keys_to_reactions(reactions_list):
    for reaction_dict in reactions_list:
        # Use the existing function to parse the reaction equation
        substrates, products = parse_reaction_equation(reaction_dict)

        # Add the parsed data back to the reaction dictionary
        reaction_dict["substrates"] = substrates
        reaction_dict["products"] = products

    return reactions_list


def extract_reaction_data(
    reactions_path: str, compounds_path: str, n: int = None
) -> tuple[list[dict], list[dict]]:
    # Load reactions data
    with open(reactions_path, "r") as file:
        reactions = json.load(file)

    # Extract the first n reactions or all if n is None
    selected_reactions = reactions[:n] if n is not None else reactions

    # Extract compound IDs from the selected reactions
    compound_ids = set()
    for reaction in selected_reactions:
        compounds_in_reaction = reaction.get("compound_ids", "").split(";")
        compound_ids.update(compounds_in_reaction)

    # Load compounds data
    with open(compounds_path, "r") as file:
        all_compounds = json.load(file)

    # Select compounds that are in the selected reactions
    selected_compounds = [
        compound for compound in all_compounds if compound["id"] in compound_ids
    ]

    return selected_reactions, selected_compounds


def compute_fingerprint_distances(
    compounds: list[dict],
) -> list[tuple[str, str, float]]:
    results = []
    for compound1, compound2 in itertools.combinations(compounds, 2):
        smiles1 = compound1.get("smiles")
        smiles2 = compound2.get("smiles")

        # Skip if either compound lacks a SMILES representation
        if not smiles1 or not smiles2:
            continue

        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)

        fp1 = FingerprintMols.FingerprintMol(mol1)
        fp2 = FingerprintMols.FingerprintMol(mol2)

        similarity = DataStructs.FingerprintSimilarity(fp1, fp2)
        results.append((compound1["id"], compound2["id"], 1 - similarity))
    return results


def read_protein_ids(filename):
    with open(filename, "r") as f:
        return [line.strip() for line in f]


def group_by_mag(protein_ids):
    mag_groups = defaultdict(list)
    for protein_id in protein_ids:
        mag_id = protein_id.split("_")[0]
        mag_groups[mag_id].append(protein_id)
    return mag_groups


def retrieve_sequences(faa_dir, mag_groups, output_file):
    with open(output_file, "w") as out_f:
        for mag_id, protein_ids in mag_groups.items():
            faa_file = os.path.join(faa_dir, f"{mag_id}.faa")
            if not os.path.exists(faa_file):
                print(f"Warning: {faa_file} not found. Skipping.")
                continue

            fasta = pyfastx.Fasta(faa_file)
            protein_set = set(protein_ids)
            for protein_id in protein_ids:
                if protein_id in fasta:
                    sequence = fasta[protein_id]
                    out_f.write(
                        f">{sequence.name} {sequence.description}\n{sequence.seq}\n"
                    )
                    protein_set.remove(protein_id)
                else:
                    print(f"Warning: Protein {protein_id} not found in {faa_file}")

            if protein_set:
                print(f"Warning: The following proteins were not found in {faa_file}:")
                print(", ".join(protein_set))


def extract_protein_ids_from_gbk(gbk_file: str) -> list[str]:
    """
    Extract full protein IDs from CDS features in a GenBank file.

    This function parses a GenBank file and extracts full protein IDs from the
    'locus_tag' qualifier of each CDS feature. The protein ID is the entire
    locus_tag value, typically in the form "OceanDNA-b29856_00043_18".

    Args:
        gbk_file (str): Path to the GenBank file.

    Returns:
        List[str]: A list of full protein IDs extracted from the file, in the order
                   they appear.

    Raises:
        FileNotFoundError: If the specified GenBank file does not exist.
        ValueError: If the file cannot be parsed as a GenBank file.

    Example:
        >>> gbk_file = "path/to/your/file.gbk"
        >>> protein_ids = extract_protein_ids(gbk_file)
        >>> print(protein_ids)
        ['OceanDNA-b29856_00043_1', 'OceanDNA-b29856_00043_2', 'OceanDNA-b29856_00043_3']
    """
    protein_ids = []

    try:
        # Parse the GenBank file
        for record in SeqIO.parse(gbk_file, "genbank"):
            for feature in record.features:
                if feature.type == "CDS":
                    if "locus_tag" in feature.qualifiers:
                        protein_id = "_".join(
                            feature.qualifiers["locus_tag"][0].split("_")[-3:]
                        )
                        protein_ids.append(protein_id)
    except FileNotFoundError:
        raise FileNotFoundError(f"The file {gbk_file} was not found.")
    except ValueError:
        raise ValueError(f"The file {gbk_file} could not be parsed as a GenBank file.")

    return protein_ids


# Check if protein sequences in gbk and KG coincide for protein_ids
from Bio import SeqIO
from typing import Dict


def extract_protein_data_from_gbk(gbk_file: str) -> Dict[str, str]:
    """
    Extract protein IDs and their corresponding sequences from CDS features in a GenBank file.

    This function parses a GenBank file and extracts protein IDs from the
    'locus_tag' qualifier and protein sequences from the 'translation' qualifier
    of each CDS feature. The protein ID is the entire locus_tag value,
    typically in the form "OceanDNA-b29856_00043_18".

    Args:
        gbk_file (str): Path to the GenBank file.

    Returns:
        Dict[str, str]: A dictionary where keys are protein IDs and values are
                        the corresponding protein sequences.

    Raises:
        FileNotFoundError: If the specified GenBank file does not exist.
        ValueError: If the file cannot be parsed as a GenBank file.

    Example:
        >>> gbk_file = "path/to/your/file.gbk"
        >>> protein_data = extract_protein_data(gbk_file)
        >>> for protein_id, sequence in list(protein_data.items())[:2]:
        ...     print(f"{protein_id}: {sequence[:20]}...")
        OceanDNA-b29856_00043_1: MARIVKIAVIGSGHVNSV...
        OceanDNA-b29856_00043_2: MNDTLYNQLKSYFDNHPV...
    """
    protein_data: Dict[str, str] = {}

    try:
        # Parse the GenBank file
        for record in SeqIO.parse(gbk_file, "genbank"):
            for feature in record.features:
                if feature.type == "CDS":
                    if (
                        "locus_tag" in feature.qualifiers
                        and "translation" in feature.qualifiers
                    ):
                        protein_id = protein_id = "_".join(
                            feature.qualifiers["locus_tag"][0].split("_")[-3:]
                        )
                        protein_sequence = feature.qualifiers["translation"][0]
                        protein_data[protein_id] = protein_sequence
    except FileNotFoundError:
        raise FileNotFoundError(f"The file {gbk_file} was not found.")
    except ValueError:
        raise ValueError(f"The file {gbk_file} could not be parsed as a GenBank file.")

    return protein_data


def has_smiles_for_all_compounds(
    reaction: Dict[str, any], compounds_data: Dict[str, Dict[str, any]]
) -> bool:
    """
    Check if a reaction has SMILES for all its compounds (substrates and products).
    """
    compound_ids = reaction["compound_ids"].split(";")
    for compound_id in compound_ids:
        if (
            compound_id not in compounds_data
            or "smiles" not in compounds_data[compound_id]
            or not compounds_data[compound_id]["smiles"]
        ):
            return False
    return True


def filter_reactions(
    reaction_ids: List[str],
    reactions_data: Dict[str, Dict[str, any]],
    compounds_data: Dict[str, Dict[str, any]],
) -> List[str]:
    """
    Filter reactions to keep only those with SMILES for all compounds.
    """
    return [
        rid
        for rid in reaction_ids
        if rid in reactions_data
        and has_smiles_for_all_compounds(reactions_data[rid], compounds_data)
    ]


def get_enzymes_with_complete_smiles(
    df: pd.DataFrame,
    reactions_data: List[Dict[str, any]],
    compounds_data: List[Dict[str, any]],
) -> pd.DataFrame:
    """
    Process the dataframe to filter rows based on reaction SMILES availability.
    """
    # Convert reactions_data and compounds_data to dictionaries for faster lookup
    reactions_dict = {r["id"]: r for r in reactions_data}
    compounds_dict = {c["id"]: c for c in compounds_data}

    # Ensure associated_reaction_ids is a list
    df["associated_reaction_ids"] = df["associated_reaction_ids"].apply(
        lambda x: x if isinstance(x, list) else eval(x) if isinstance(x, str) else []
    )

    # Filter reactions and create a new column with filtered reaction IDs
    df["filtered_reaction_ids"] = df["associated_reaction_ids"].apply(
        lambda x: filter_reactions(x, reactions_dict, compounds_dict)
    )

    # Keep only rows with at least one remaining reaction
    df_filtered = df[df["filtered_reaction_ids"].apply(len) > 0].copy()

    # Update the associated_reaction_ids column with the filtered list
    df_filtered["associated_reaction_ids"] = df_filtered["filtered_reaction_ids"]

    # Drop the temporary column
    df_filtered = df_filtered.drop(columns=["filtered_reaction_ids"])

    return df_filtered
