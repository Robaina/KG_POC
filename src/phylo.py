import subprocess
from typing import Optional
from Bio import SeqIO, Phylo, AlignIO
from Bio.Align import MultipleSeqAlignment
from Bio.Phylo.BaseTree import Tree
from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceTreeConstructor
import matplotlib.pyplot as plt
import pylab
import os


def align_sequences(faa_file: str) -> MultipleSeqAlignment:
    """
    Align sequences using MUSCLE.

    Args:
        faa_file (str): Path to the input FAA file.

    Returns:
        MultipleSeqAlignment: The aligned sequences.

    Raises:
        subprocess.CalledProcessError: If MUSCLE alignment fails.
    """
    input_file = faa_file
    output_file = faa_file + ".aln"

    # Try MUSCLE5 command format
    muscle_command = f"muscle -align {input_file} -output {output_file}"
    try:
        result = subprocess.run(
            muscle_command,
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        print("MUSCLE alignment completed successfully.")
    except subprocess.CalledProcessError:
        # If MUSCLE5 format fails, try MUSCLE3 format
        muscle_command = f"muscle -in {input_file} -out {output_file}"
        try:
            result = subprocess.run(
                muscle_command,
                shell=True,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            print("MUSCLE alignment completed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"MUSCLE alignment failed with error: {e}")
            print(f"STDOUT: {e.stdout}")
            print(f"STDERR: {e.stderr}")
            raise

    # Read the aligned sequences
    alignment = AlignIO.read(output_file, "fasta")
    return alignment


def reconstruct_phylogenetic_tree(faa_file: str, output_tree_file: str) -> Tree:
    """
    Reconstruct a phylogenetic tree from a FAA file and save it to a file.

    Args:
        faa_file (str): Path to the input FAA file.
        output_tree_file (str): Path to save the output tree file.

    Returns:
        Tree: The reconstructed phylogenetic tree.
    """
    alignment = align_sequences(faa_file)
    calculator = DistanceCalculator("identity")
    dm = calculator.get_distance(alignment)
    constructor = DistanceTreeConstructor(calculator, "upgma")
    tree = constructor.build_tree(alignment)

    # Save the tree to the specified output file
    Phylo.write(tree, output_tree_file, "newick")
    print(f"Tree saved in Newick format as '{output_tree_file}'")

    return tree


def visualize_tree(
    tree_file: str, method: str = "matplotlib", output_file: str = "phylogenetic_tree"
) -> None:
    """
    Visualize a phylogenetic tree from a file.

    Args:
        tree_file (str): Path to the input tree file in Newick format.
        method (str): Visualization method ('matplotlib', 'ascii', or 'pylab').
        output_file (str): Path to save the output image file (for 'matplotlib' and 'pylab' methods).
    """
    tree = Phylo.read(tree_file, "newick")

    if method == "matplotlib":
        fig, ax = plt.subplots(figsize=(10, 8))
        Phylo.draw(tree, axes=ax, do_show=False)
        plt.savefig(f"{output_file}_matplotlib.png")
        plt.close()
        print(f"Tree visualization saved as '{output_file}_matplotlib.png'")
    elif method == "ascii":
        Phylo.draw_ascii(tree)
        print("ASCII representation of the tree printed above")
    elif method == "pylab":
        pylab.figure(figsize=(10, 8))
        Phylo.draw(tree, do_show=False)
        pylab.savefig(f"{output_file}_pylab.png")
        pylab.close()
        print(f"Tree visualization saved as '{output_file}_pylab.png'")
    else:
        print("Invalid visualization method. Choose 'matplotlib', 'ascii', or 'pylab'.")
