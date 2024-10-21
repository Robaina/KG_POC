import os
from typing import Dict
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import re

import networkx as nx
import numpy as np
from collections import defaultdict


def extract_phylum(classification_string):
    try:
        match = re.search(r'p__([^;]+)', classification_string)
        return match.group(1) if match else 'Unknown'
    except:
        return 'Unknown'

def process_results(input_dir: str) -> Dict[str, pd.DataFrame]:
    """
    Process TSV files containing EC number results.

    Args:
        input_dir (str): Directory containing the TSV files.

    Returns:
        Dict[str, pd.DataFrame]: A dictionary where keys are EC numbers and values are processed DataFrames.
    """
    processed_data = {}
    input_path = Path(input_dir)

    for file in input_path.glob('KG_hits_*.tsv'):
        # Extract EC number from filename
        ec_number = file.stem.split('_')[-1]
        
        # Read TSV file
        df = pd.read_csv(file, sep='\t')
        
        # Extract phylum from gtdb_classification if the column exists
        if 'gtdb_classification' in df.columns:
            df['phylum'] = df['gtdb_classification'].apply(extract_phylum)
        else:
            df['phylum'] = 'Unknown'
        
        # Convert numeric columns to float, if they exist
        numeric_columns = ['sample_temperature', 'sample_depth', 'sample_latitude', 'sample_longitude']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        processed_data[ec_number] = df
    
    return processed_data

def plot_ec_number_statistics(
    processed_data: Dict[str, pd.DataFrame],
    shp_file_path: str,
    enzymes_dict: Dict[str, str],
    market_sizes: Dict[str, str],
    output_dir: str,
    output_prefix: str = "",
    base_font_size: int = 12
) -> None:
    """
    Plot statistics for EC numbers including a world map, phylum distribution, and summary text.

    Args:
        processed_data (Dict[str, pd.DataFrame]): Dictionary of processed DataFrames for each EC number.
        shp_file_path (str): Path to the shapefile for world map plotting.
        enzymes_dict (Dict[str, str]): Dictionary mapping EC numbers to enzyme names.
        market_sizes (Dict[str, str]): Dictionary of market sizes for enzymes.
        output_dir (str): Directory to save the output figures.
        output_prefix (str, optional): Prefix to append to output filenames. Defaults to "".
        base_font_size (int, optional): Base font size for the plots. Defaults to 12.

    Returns:
        None

    Raises:
        FileNotFoundError: If the shapefile or output directory doesn't exist.
        ValueError: If the processed_data is empty.
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Read world shapefile
    world = gpd.read_file(shp_file_path)

    for ec_number, df in processed_data.items():

        # Extract phylum from gtdb_classification if the column exists
        if 'gtdb_classification' in df.columns:
            df['phylum'] = df['gtdb_classification'].apply(extract_phylum)
        else:
            df['phylum'] = 'Unknown'

        fig = plt.figure(figsize=(24, 8))  # Adjusted for single row layout
        gs = fig.add_gridspec(2, 3, height_ratios=[0.1, 1])

        enzyme_name = enzymes_dict.get(ec_number, "Unknown Enzyme")
        market_size = market_sizes.get(enzyme_name, "N/A")

        # Title
        fig.add_subplot(gs[0, :])
        plt.axis('off')
        plt.text(0.5, 0.5, f'Statistics for EC Number: {ec_number} - {enzyme_name}\nTotal Hits: {len(df)} | Market Size (2023): ${market_size} million', 
                 fontsize=base_font_size*1.8, ha='center', va='center', fontweight='bold')

        # Summary text
        ax_summary = fig.add_subplot(gs[1, 0])
        summary_text = f"""
Enzyme: {enzyme_name}
EC Number: {ec_number}
Number of Hits: {len(df)}
Market Size (2023): ${market_size} million
Temperature Range: {df['sample_temperature'].min():.2f} to {df['sample_temperature'].max():.2f} °C
Depth Range: {df['sample_depth'].min():.2f} to {df['sample_depth'].max():.2f} m
Latitude Range: {df['sample_latitude'].min():.2f} to {df['sample_latitude'].max():.2f}
Longitude Range: {df['sample_longitude'].min():.2f} to {df['sample_longitude'].max():.2f}
Top 5 Phyla:
{df['phylum'].value_counts().head().to_string()}
"""
        ax_summary.text(0.05, 0.95, summary_text, verticalalignment='top', fontsize=base_font_size*1.1, 
                        transform=ax_summary.transAxes, ha='left', va='top', linespacing=1.5)
        ax_summary.axis('off')

        # World map with sample locations
        ax_map = fig.add_subplot(gs[1, 1])
        world.plot(ax=ax_map, color='lightgrey', edgecolor='black')
        ax_map.scatter(df['sample_longitude'], df['sample_latitude'], c='red', s=10, alpha=0.7)
        ax_map.set_title('Sample Locations', fontsize=base_font_size*1.5)
        ax_map.set_xlabel('Longitude', fontsize=base_font_size*1.2)
        ax_map.set_ylabel('Latitude', fontsize=base_font_size*1.2)
        ax_map.set_xlim(-180, 180)
        ax_map.set_ylim(-90, 90)
        ax_map.tick_params(labelsize=base_font_size)

        # Phylum distribution
        ax_pie = fig.add_subplot(gs[1, 2])
        phylum_counts = df['phylum'].value_counts()
        other_threshold = 0.02
        other_mask = phylum_counts / phylum_counts.sum() < other_threshold
        other_count = phylum_counts[other_mask].sum()
        phylum_counts_grouped = phylum_counts[~other_mask]
        phylum_counts_grouped['Other'] = other_count
        wedges, texts, autotexts = ax_pie.pie(phylum_counts_grouped.values, 
                                              labels=phylum_counts_grouped.index, 
                                              autopct='%1.1f%%', 
                                              startangle=90, 
                                              textprops={'fontsize': base_font_size*0.8})
        ax_pie.set_title('Phylum Distribution', fontsize=base_font_size*1.5)
        
        # Adjust legend for pie chart
        plt.setp(autotexts, size=base_font_size*0.8, weight="bold")
        ax_pie.legend(wedges, phylum_counts_grouped.index,
                      title="Phyla",
                      loc="center left",
                      bbox_to_anchor=(1, 0, 0.5, 1),
                      fontsize=base_font_size*0.8)

        plt.tight_layout()
        output_file = output_path / f'{output_prefix}ec_number_{ec_number}_statistics.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

    print(f"All figures have been saved to {output_path}")

    
def plot_compound_statistics(
    processed_data: Dict[str, pd.DataFrame],
    shp_file_path: str,
    output_dir: str,
    output_prefix: str = "",
    base_font_size: int = 12
) -> None:
    """
    Plot statistics for compound IDs including a world map, phylum distribution, and summary text.

    Args:
        processed_data (Dict[str, pd.DataFrame]): Dictionary of processed DataFrames for each compound ID.
        shp_file_path (str): Path to the shapefile for world map plotting.
        output_dir (str): Directory to save the output figures.
        output_prefix (str, optional): Prefix to append to output filenames. Defaults to "".
        base_font_size (int, optional): Base font size for the plots. Defaults to 12.

    Returns:
        None

    Raises:
        FileNotFoundError: If the shapefile or output directory doesn't exist.
        ValueError: If the processed_data is empty.
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Read world shapefile
    world = gpd.read_file(shp_file_path)

    for compound_id, df in processed_data.items():
        # Filter rows with valid latitude and longitude
        df = df.dropna(subset=['sample_latitude', 'sample_longitude'])
        
        if df.empty:
            print(f"Skipping compound {compound_id} due to lack of valid coordinates.")
            continue

        fig = plt.figure(figsize=(24, 8))  # Adjusted for single row layout
        gs = fig.add_gridspec(2, 3, height_ratios=[0.1, 1])
        
        compound_name = df['compound_name'].iloc[0]
        reaction_name = df['reaction_name'].iloc[0]

        # Title
        fig.add_subplot(gs[0, :])
        plt.axis('off')
        plt.text(0.5, 0.5, f'Statistics for Compound ID: {compound_id} - {compound_name}\nReaction: {reaction_name}\nTotal Hits: {len(df)}', 
                 fontsize=base_font_size*1.8, ha='center', va='center', fontweight='bold')

        # Summary text
        ax_summary = fig.add_subplot(gs[1, 0])
        summary_text = f"""
Compound: {compound_name}
Compound ID: {compound_id}
Number of Hits: {len(df)}
Temperature Range: {df['sample_temperature'].min():.2f} to {df['sample_temperature'].max():.2f} °C
Depth Range: {df['sample_depth'].min():.2f} to {df['sample_depth'].max():.2f} m
Latitude Range: {df['sample_latitude'].min():.2f} to {df['sample_latitude'].max():.2f}
Longitude Range: {df['sample_longitude'].min():.2f} to {df['sample_longitude'].max():.2f}
Top 5 Phyla:
{df['gtdb_classification'].apply(lambda x: x.split(';')[1].split('__')[1]).value_counts().head().to_string()}
"""
        ax_summary.text(0.05, 0.95, summary_text, verticalalignment='top', fontsize=base_font_size*1.1, 
                        transform=ax_summary.transAxes, ha='left', va='top', linespacing=1.5)
        ax_summary.axis('off')

        # World map with sample locations
        ax_map = fig.add_subplot(gs[1, 1])
        world.plot(ax=ax_map, color='lightgrey', edgecolor='black')
        ax_map.scatter(df['sample_longitude'], df['sample_latitude'], c='red', s=10, alpha=0.7)
        ax_map.set_title('Sample Locations', fontsize=base_font_size*1.5)
        ax_map.set_xlabel('Longitude', fontsize=base_font_size*1.2)
        ax_map.set_ylabel('Latitude', fontsize=base_font_size*1.2)
        ax_map.set_xlim(-180, 180)
        ax_map.set_ylim(-90, 90)
        ax_map.tick_params(labelsize=base_font_size)

        # Phylum distribution
        ax_pie = fig.add_subplot(gs[1, 2])
        phylum_counts = df['gtdb_classification'].apply(lambda x: x.split(';')[1].split('__')[1]).value_counts()
        other_threshold = 0.02
        other_mask = phylum_counts / phylum_counts.sum() < other_threshold
        other_count = phylum_counts[other_mask].sum()
        phylum_counts_grouped = phylum_counts[~other_mask]
        phylum_counts_grouped['Other'] = other_count
        wedges, texts, autotexts = ax_pie.pie(phylum_counts_grouped.values, 
                                              labels=phylum_counts_grouped.index, 
                                              autopct='%1.1f%%', 
                                              startangle=90, 
                                              textprops={'fontsize': base_font_size*0.8})
        ax_pie.set_title('Phylum Distribution', fontsize=base_font_size*1.5)
        
        # Adjust legend for pie chart
        plt.setp(autotexts, size=base_font_size*0.8, weight="bold")
        ax_pie.legend(wedges, phylum_counts_grouped.index,
                      title="Phyla",
                      loc="center left",
                      bbox_to_anchor=(1, 0, 0.5, 1),
                      fontsize=base_font_size*0.8)

        plt.tight_layout()
        output_file = output_path / f'{output_prefix}compound_{compound_id}_statistics.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

    print(f"All figures have been saved to {output_path}")
    
    
def plot_bgc_statistics(
    processed_data: Dict[str, pd.DataFrame],
    shp_file_path: str,
    output_dir: str,
    output_prefix: str = "",
    base_font_size: int = 12
) -> None:
    """
    Plot statistics for BGC classes including a world map, phylum distribution, and summary text.

    Args:
        processed_data (Dict[str, pd.DataFrame]): Dictionary of processed DataFrames for each BGC class.
        shp_file_path (str): Path to the shapefile for world map plotting.
        output_dir (str): Directory to save the output figures.
        output_prefix (str, optional): Prefix to append to output filenames. Defaults to "".
        base_font_size (int, optional): Base font size for the plots. Defaults to 12.

    Returns:
        None

    Raises:
        FileNotFoundError: If the shapefile or output directory doesn't exist.
        ValueError: If the processed_data is empty.
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Read world shapefile
    world = gpd.read_file(shp_file_path)

    for bgc_class, df in processed_data.items():

        # Extract phylum from gtdb_classification
        df['phylum'] = df['gtdb_classification'].apply(extract_phylum)

        fig = plt.figure(figsize=(24, 8))  # Adjusted for single row layout
        gs = fig.add_gridspec(2, 3, height_ratios=[0.1, 1])

        # Title
        fig.add_subplot(gs[0, :])
        plt.axis('off')
        plt.text(0.5, 0.5, f'Statistics for BGC Class: {bgc_class}\nTotal BGCs: {len(df)}', 
                 fontsize=base_font_size*1.8, ha='center', va='center', fontweight='bold')

        # Summary text
        ax_summary = fig.add_subplot(gs[1, 0])
        summary_text = f"""
BGC Class: {bgc_class}
Number of BGCs: {len(df)}
Temperature Range: {df['sample_temperature'].min():.2f} to {df['sample_temperature'].max():.2f} °C
Depth Range: {df['sample_depth'].min():.2f} to {df['sample_depth'].max():.2f} m
Latitude Range: {df['sample_latitude'].min():.2f} to {df['sample_latitude'].max():.2f}
Longitude Range: {df['sample_longitude'].min():.2f} to {df['sample_longitude'].max():.2f}
Top 5 Phyla:
{df['phylum'].value_counts().head().to_string()}
"""
        ax_summary.text(0.05, 0.95, summary_text, verticalalignment='top', fontsize=base_font_size*1.1, 
                        transform=ax_summary.transAxes, ha='left', va='top', linespacing=1.5)
        ax_summary.axis('off')

        # World map with sample locations
        ax_map = fig.add_subplot(gs[1, 1])
        world.plot(ax=ax_map, color='lightgrey', edgecolor='black')
        ax_map.scatter(df['sample_longitude'], df['sample_latitude'], c='red', s=10, alpha=0.7)
        ax_map.set_title('Sample Locations', fontsize=base_font_size*1.5)
        ax_map.set_xlabel('Longitude', fontsize=base_font_size*1.2)
        ax_map.set_ylabel('Latitude', fontsize=base_font_size*1.2)
        ax_map.set_xlim(-180, 180)
        ax_map.set_ylim(-90, 90)
        ax_map.tick_params(labelsize=base_font_size)

        # Phylum distribution
        ax_pie = fig.add_subplot(gs[1, 2])
        phylum_counts = df['phylum'].value_counts()
        other_threshold = 0.02
        other_mask = phylum_counts / phylum_counts.sum() < other_threshold
        other_count = phylum_counts[other_mask].sum()
        phylum_counts_grouped = phylum_counts[~other_mask]
        phylum_counts_grouped['Other'] = other_count
        wedges, texts, autotexts = ax_pie.pie(phylum_counts_grouped.values, 
                                              labels=phylum_counts_grouped.index, 
                                              autopct='%1.1f%%', 
                                              startangle=90, 
                                              textprops={'fontsize': base_font_size*0.8})
        ax_pie.set_title('Phylum Distribution', fontsize=base_font_size*1.5)
        
        # Adjust legend for pie chart
        plt.setp(autotexts, size=base_font_size*0.8, weight="bold")
        ax_pie.legend(wedges, phylum_counts_grouped.index,
                      title="Phyla",
                      loc="center left",
                      bbox_to_anchor=(1, 0, 0.5, 1),
                      fontsize=base_font_size*0.8)

        plt.tight_layout()
        output_file = output_path / f'{output_prefix}bgc_class_{bgc_class}_statistics.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

    print(f"All figures have been saved to {output_path}")


def parse_reaction_equation(equation):
    # Split the equation into left and right sides
    sides = re.split(r'\s*<*=*>\s*', equation)
    
    # Function to extract compounds from a side of the equation
    def extract_compounds(side):
        return re.findall(r'cpd\d{5}', side)
    
    # Determine directionality
    if '<=>' in equation:
        direction = 'bidirectional'
    elif '=>' in equation:
        direction = 'forward'
    else:
        direction = 'reverse'
    
    # Extract compounds
    if len(sides) == 2:
        left_compounds = extract_compounds(sides[0])
        right_compounds = extract_compounds(sides[1])
    else:
        left_compounds = right_compounds = extract_compounds(sides[0])
    
    return left_compounds, right_compounds, direction

def select_representative_reactions(df):
    ec_to_reactions = defaultdict(list)
    for _, row in df.iterrows():
        for ec, reaction_id, equation, deltag in zip(row['ec_numbers'], row['reaction_ids'], row['reaction_equations'], row['reaction_deltags']):
            ec_to_reactions[ec].append((reaction_id, equation, float(deltag), row['protein_id']))
    
    representative_reactions = {}
    for ec, reactions in ec_to_reactions.items():
        # Select the reaction with the smallest absolute delta G
        representative_reactions[ec] = min(reactions, key=lambda x: abs(x[2]))
    
    return representative_reactions

def build_minimal_bipartite_reaction_graph(representative_reactions, compound_data=None):
    G = nx.DiGraph()
    
    compound_dict = {item['id']: item['name'] for item in compound_data} if compound_data else {}
    
    for ec, (reaction_id, equation, _, protein_id) in representative_reactions.items():
        left, right, direction = parse_reaction_equation(equation)
        
        # Add reaction node with EC number and protein ID
        node_label = f"{reaction_id}\n{ec}\n{protein_id}"
        G.add_node(node_label, bipartite=1, ec_number=ec, reaction_id=reaction_id, protein_id=protein_id)
        
        # Add compound nodes and edges
        for compound in set(left + right):
            if compound in compound_dict:
                compound_label = f"{compound}\n{compound_dict[compound]}"
            else:
                compound_label = compound
            G.add_node(compound_label, bipartite=0, compound_id=compound)
        
        if direction in ['bidirectional', 'forward']:
            for substrate in left:
                substrate_label = f"{substrate}\n{compound_dict[substrate]}" if substrate in compound_dict else substrate
                G.add_edge(substrate_label, node_label)
            for product in right:
                product_label = f"{product}\n{compound_dict[product]}" if product in compound_dict else product
                G.add_edge(node_label, product_label)
        
        if direction in ['reverse']:
            for product in right:
                product_label = f"{product}\n{compound_dict[product]}" if product in compound_dict else product
                G.add_edge(product_label, node_label)
            for substrate in left:
                substrate_label = f"{substrate}\n{compound_dict[substrate]}" if substrate in compound_dict else substrate
                G.add_edge(node_label, substrate_label)
    
    return G

def find_connected_subnetworks(G):
    return list(nx.weakly_connected_components(G))

def analyze_minimal_metabolic_network(df, compound_data=None):
    # Select representative reactions
    representative_reactions = select_representative_reactions(df)
    
    # Build the minimal bipartite reaction graph
    G = build_minimal_bipartite_reaction_graph(representative_reactions, compound_data)
    
    # Find connected subnetworks
    subnetworks = find_connected_subnetworks(G)
    
    # Analyze subnetworks
    results = []
    for i, subnetwork in enumerate(subnetworks, 1):
        subG = G.subgraph(subnetwork)
        compounds = [data['compound_id'] for node, data in subG.nodes(data=True) if 'compound_id' in data]
        reactions = [data['reaction_id'] for _, data in subG.nodes(data=True) if 'reaction_id' in data]
        ec_numbers = list(set(data['ec_number'] for _, data in subG.nodes(data=True) if 'ec_number' in data))
        protein_ids = list(set(data['protein_id'] for _, data in subG.nodes(data=True) if 'protein_id' in data))
        results.append({
            'subnetwork_id': i,
            'num_compounds': len(compounds),
            'num_reactions': len(reactions),
            'num_ec_numbers': len(ec_numbers),
            'num_proteins': len(protein_ids),
            'compounds': compounds,
            'reactions': reactions,
            'ec_numbers': ec_numbers,
            'protein_ids': protein_ids
        })
    
    return pd.DataFrame(results), G

def create_concentric_layout(G):
    # Separate nodes into reaction and compound nodes
    reaction_nodes = [node for node, data in G.nodes(data=True) if data['bipartite'] == 1]
    compound_nodes = [node for node, data in G.nodes(data=True) if data['bipartite'] == 0]
    
    # Calculate positions for reaction nodes (inner circle)
    reaction_radius = 0.5
    reaction_angles = np.linspace(0, 2*np.pi, len(reaction_nodes), endpoint=False)
    reaction_pos = {node: (reaction_radius * np.cos(angle), reaction_radius * np.sin(angle)) 
                    for node, angle in zip(reaction_nodes, reaction_angles)}
    
    # Calculate positions for compound nodes (outer circle)
    compound_radius = 1.0
    compound_angles = np.linspace(0, 2*np.pi, len(compound_nodes), endpoint=False)
    compound_pos = {node: (compound_radius * np.cos(angle), compound_radius * np.sin(angle)) 
                    for node, angle in zip(compound_nodes, compound_angles)}
    
    # Combine positions
    pos = {**reaction_pos, **compound_pos}
    return pos

def visualize_largest_subnetwork(G, save_path=None, circular_layout=False, title: str = None):
    if not title:
        title = "Largest Minimal Metabolic Subnetwork (Bipartite Graph)"
    largest_subnetwork = max(find_connected_subnetworks(G), key=len)
    subG = G.subgraph(largest_subnetwork)
    
    if circular_layout:
        pos = create_concentric_layout(subG)
    else:
        pos = nx.spring_layout(subG, k=0.5, iterations=50)
    
    plt.figure(figsize=(20, 20))  # Square figure for better circular layout
    
    # Draw compound nodes (outer circle)
    nx.draw_networkx_nodes(subG, pos, 
                           nodelist=[node for node, data in subG.nodes(data=True) if data['bipartite'] == 0],
                           node_color='lightblue', node_size=3000, alpha=0.8)
    
    # Draw reaction nodes (inner circle)
    nx.draw_networkx_nodes(subG, pos, 
                           nodelist=[node for node, data in subG.nodes(data=True) if data['bipartite'] == 1],
                           node_color='lightgreen', node_size=5000, alpha=0.8, node_shape='s')
    
    # Draw edges
    nx.draw_networkx_edges(subG, pos, edge_color='gray', arrows=True, 
                           arrowsize=20, width=1.5, alpha=0.7)
    
    # Draw labels
    label_pos = {k: (v[0]*1.0, v[1]*1.0) for k, v in pos.items()}  # Adjust label positions
    nx.draw_networkx_labels(subG, label_pos, font_size=8, font_weight='bold')
    
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, format=save_path.split('.')[-1], dpi=300, bbox_inches='tight')
        print(f"Graph saved to {save_path}")
    
    plt.show()
    
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
from typing import Optional, List, Literal, Union
import matplotlib.patches as mpatches

def visualize_advanced_metabolic_network(
    G: Union[nx.Graph, nx.DiGraph],
    focus_node: Optional[str] = None,
    layout: Literal['spring', 'circular', 'shell'] = 'spring',
    highlight_path: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    title: Optional[str] = None
) -> None:
    """
    Visualize a metabolic network with advanced features, curved edges, and improved spring layout.

    This function creates an advanced visualization of a metabolic network,
    represented as a bipartite graph with compounds and reactions. It works
    with both directed and undirected graphs, includes node labels, and uses
    curved edges for better visibility.

    Parameters:
    -----------
    G : Union[nx.Graph, nx.DiGraph]
        The NetworkX graph representing the metabolic network. Nodes should have
        a 'bipartite' attribute (0 for compounds, 1 for reactions) and optionally
        a 'name' attribute for labels.

    focus_node : Optional[str], default=None
        If provided, the visualization will focus on this node and its neighbors
        within 2 steps. If None, the entire network is visualized.

    layout : Literal['spring', 'circular', 'shell'], default='spring'
        The layout algorithm to use for node positioning:
        - 'spring': Force-directed layout (improved version)
        - 'circular': Nodes arranged in a circle
        - 'shell': Nodes arranged in concentric circles

    highlight_path : Optional[List[str]], default=None
        A list of node identifiers representing a path to highlight in the network.

    save_path : Optional[str], default=None
        If provided, the visualization will be saved to this file path.

    title : Optional[str], default=None
        The title for the visualization. If None, a default title is used.

    Returns:
    --------
    None
        This function doesn't return any value but displays the plot and
        optionally saves it to a file.
    """
    if not title:
        title = "Advanced Metabolic Network Visualization"

    # Use the largest connected component
    if G.is_directed():
        largest_cc = max(nx.weakly_connected_components(G), key=len)
    else:
        largest_cc = max(nx.connected_components(G), key=len)
    subG = G.subgraph(largest_cc)

    if focus_node:
        # Create a subgraph of nodes within 2 steps of the focus node
        subG = nx.ego_graph(subG, focus_node, radius=2, undirected=True)

    # Separate compound and reaction nodes
    compound_nodes = [node for node, data in subG.nodes(data=True) if data['bipartite'] == 0]
    reaction_nodes = [node for node, data in subG.nodes(data=True) if data['bipartite'] == 1]

    # Create layout
    if layout == 'spring':
        pos = nx.spring_layout(subG, k=0.5, iterations=50, scale=2)
    elif layout == 'circular':
        pos = nx.circular_layout(subG)
    elif layout == 'shell':
        pos = nx.shell_layout(subG)
    else:
        pos = nx.spring_layout(subG, k=0.5, iterations=50, scale=2)

    # Set up the plot
    plt.figure(figsize=(20, 20))
    
    # Color palette
    palette = sns.color_palette("pastel")
    
    # Compute node sizes based on degree
    node_sizes = [3000 * (1 + subG.degree(node) / 10) for node in compound_nodes]
    
    # Draw compound nodes
    nx.draw_networkx_nodes(subG, pos, 
                           nodelist=compound_nodes,
                           node_color=palette[0], 
                           node_size=node_sizes, 
                           alpha=0.8,
                           edgecolors='black',
                           linewidths=1)
    
    # Draw reaction nodes
    nx.draw_networkx_nodes(subG, pos, 
                           nodelist=reaction_nodes,
                           node_color=palette[1], 
                           node_size=2000, 
                           alpha=0.8,
                           node_shape='s',
                           edgecolors='black',
                           linewidths=1)
    
    # Draw curved edges
    for edge in subG.edges():
        start = pos[edge[0]]
        end = pos[edge[1]]
        rad = 0.2
        arrow = mpatches.FancyArrowPatch(start, end,
                                         connectionstyle=f"arc3,rad={rad}",
                                         color='gray',
                                         arrowstyle="->",
                                         mutation_scale=20,
                                         linewidth=2,
                                         alpha=0.6)
        plt.gca().add_patch(arrow)
    
    # Highlight path if provided
    if highlight_path:
        path_edges = list(zip(highlight_path, highlight_path[1:]))
        for edge in path_edges:
            start = pos[edge[0]]
            end = pos[edge[1]]
            rad = 0.2
            arrow = mpatches.FancyArrowPatch(start, end,
                                             connectionstyle=f"arc3,rad={rad}",
                                             color='red',
                                             arrowstyle="->",
                                             mutation_scale=20,
                                             linewidth=1,
                                             alpha=1)
            plt.gca().add_patch(arrow)

    # Add labels
    labels = {}
    for node in subG.nodes():
        if 'name' in subG.nodes[node]:
            labels[node] = subG.nodes[node]['name']
        else:
            labels[node] = str(node)
    nx.draw_networkx_labels(subG, pos, labels, font_size=8, font_weight='bold')
    
    # Add a legend
    plt.plot([], [], 'o', color=palette[0], label='Compound', markersize=15)
    plt.plot([], [], 's', color=palette[1], label='Reaction', markersize=15)
    if highlight_path:
        plt.plot([], [], '-r', label='Highlighted Path')
    plt.legend(fontsize=12, loc='upper left', bbox_to_anchor=(1, 1))

    plt.title(title, fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, format=save_path.split('.')[-1], dpi=300, bbox_inches='tight')
        print(f"Graph saved to {save_path}")
    
    plt.show()