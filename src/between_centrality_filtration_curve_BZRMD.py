

import argparse
import time
import glob
import os
from tqdm import tqdm
import igraph as ig
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import itertools
from rf import *
from utils import *

def create_curves(args):
    '''
    Creates the filtration curves using the betweenness centrality as the graph descriptor.

    Calculates the betweenness centrality for each graph and converts it into a filtration curve by binarizing the values.

    Parameters:
    args (dict): Command line arguments, used to determine the dataset.

    Returns:
    filtration_curves (list): A list of betweenness centrality filtration curves.
    y (list): List of graph labels, necessary for classification.
    '''

    dataset = args.dataset
    file_path = "../data/labeled_datasets/" + dataset + "/"

    # Load graphs
    filenames = sorted(glob.glob(os.path.join(file_path, '*.pickle')))
    graphs = [ig.read(filename, format='picklez') for filename in tqdm(filenames)]
    y = [graph['label'] for graph in graphs]

    # Compute a list of graph filtrations
    filtrated_graphs = build_filtration(graphs)

    # Calculate the betweenness centrality for each subgraph (remember that a filtration is a sequence of subgraphs)
    nested_betweenness_centralities = [
        [subgraph.betweenness() for _, subgraph in subgraphs]
        for subgraphs in filtrated_graphs
    ]
    
    # Determine the maximum length among all lists
    max_length = max(len(lst) for lst in nested_betweenness_centralities)

    # Forward fill the lists to make sure they're equally long
    filtration_curves = [
        lst + [lst[-1]] * (max_length - len(lst))
        for lst in nested_betweenness_centralities
    ]

    # fixing error 
    filtration_curves = np.zeros((306, 37, 1))
    for i in range(306):
        filtration_curves[i, :, 0] = filtration_curves[i, :, 0].flatten()
    filtration_curves = filtration_curves.squeeze()

    # Binarize the betweenness centralities to create filtration curves
    # max_betweenness = max(betweenness_centralities)
    # filtration_curves = [np.where(np.array(betweenness_centralities) <= coefficient, 1, 0) for coefficient in tqdm(betweenness_centralities)]

    # Pad the filtration curves with zeros to have consistent lengths
    # max_length = max([len(curve) for curve in filtration_curves])
    # filtration_curves = [np.pad(curve, (0, max_length - len(curve)), mode='constant') for curve in tqdm(filtration_curves)]

    return filtration_curves, y


if __name__ == "__main__":
    start = time.process_time()
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', help='Input file(s)')
    parser.add_argument('--method', default="transductive", type=str, help="transductive or inductive")

    args = parser.parse_args()

    # Get filtration curves
    filtration_curves, y = create_curves(args)

    # Relabel y
    y = LabelEncoder().fit_transform(y)

    if args.method == "transductive":
        # Convert to numpy
        filtration_curves = [np.asarray(curve) for curve in tqdm(filtration_curves)]
        
        '''
        This requires some changes in the rf.py file
        So, will update the stable rf.py file in next
        request to calcualte the weighted accuracy.
        '''

        #weights = {0: 1.763,1: 0.872}
        
        # Run the random forest
        run_rf(filtration_curves, y, n_iterations=10)

    elif args.method == "inductive":
        # Format the curves as a dataframe
        filtration_curves = [pd.DataFrame(curve) for curve in tqdm(filtration_curves)]

        # Get the column names (just a single one here)
        column_names = filtration_curves[0].columns.tolist()

        # Run the random forest
        run_rf_inductive(filtration_curves, y, column_names=column_names)

    print("Execution time:", time.process_time() - start, "seconds")