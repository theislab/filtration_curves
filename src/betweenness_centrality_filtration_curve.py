''' Script to run the filtration curves using betweenness centrality as the graph descriptor function. '''

import argparse
import time
import glob
import os
from tqdm import tqdm
import igraph as ig
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from rf import *
from utils import *


def create_curves(args):
    '''
    Creates the betweenness centrality filtration curves.

    Calculates the betweenness centrality for each graph and converts it into
    a filtration curve by binarizing the values.

    Parameters
    ----------
    args: dict
        Command line arguments, used to determine the dataset

    Returns
    -------
    filtration_curves: list
        A list of betweenness centrality filtration curves.
    y: list
        List of graph labels, necessary for classification.

    '''
    dataset = args.dataset
    file_path = "../data/unlabeled_datasets/" + dataset + "/"

    # This section is the normal section to load the graphs.
    filenames = sorted(glob.glob(os.path.join(file_path, '*.pickle')))

    graphs = [ig.read(filename, format='picklez') for filename in tqdm(filenames)]
    y = [graph['label'] for graph in graphs]

    # Calculate betweenness centrality for each graph
    betweenness_centralities = [graph.betweenness() for graph in tqdm(graphs)]

    # Binarize the betweenness centrality values to create filtration curves
    max_betweenness_centrality = max(max(bc) for bc in betweenness_centralities)
    filtration_curves = [np.where(np.array(bc) <= max_betweenness_centrality, 1, 0) for bc in tqdm(betweenness_centralities)]

    # Pad the filtration curves with zeros to have consistent lengths
    max_length = max(len(curve) for curve in filtration_curves)
    filtration_curves = [np.pad(curve, (0, max_length - len(curve)), mode='constant') for curve in tqdm(filtration_curves)]
    
    return filtration_curves, y


if __name__ == "__main__":
    start = time.process_time()
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', help='Input file(s)')
    parser.add_argument('--method', default="transductive", type=str, help="transductive or inductive")

    args = parser.parse_args()

    # get filtration curves
    filtration_curves, y = create_curves(args)

    # relabel y
    y = LabelEncoder().fit_transform(y)

    if args.method == "transductive":
        # convert to numpy
        filtration_curves = [np.asarray(i) for i in tqdm(filtration_curves)]
        weights = {0: 1.763, 1: 0.872}
        # run the random forest
        run_rf(filtration_curves, y, n_iterations=10)#weights=weights)

    elif args.method == "inductive":
        # format the curves as a dataframe
        filtration_curves = [pd.DataFrame(i) for i in tqdm(filtration_curves)]

        # get the column names (just a single one here)
        column_names = filtration_curves[0].columns.tolist()
        #weights = {0: 1.763, 1: 0.872}
        # run the random forest
        run_rf_inductive(filtration_curves, y, column_names=column_names)#weights=weights)

    print("Execution time:", time.process_time() - start, "seconds")
