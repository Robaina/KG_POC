from __future__ import annotations
import json
import itertools
from rdkit import Chem, RDLogger
from rdkit.Chem import DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols

RDLogger.DisableLog("rdApp.*")


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


def extract_data(
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
