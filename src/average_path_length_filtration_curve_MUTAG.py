''' Script to run the filtration curves using average path length as the graph descriptor function. ''' 

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
    Creates the average path length filtration curves.

    Calculates the average path length for each graph and converts it into
    a filtration curve by binarizing the values.

    Parameters
    ----------
    args: dict 
        Command line arguments, used to determine the dataset 

    Returns
    -------
    filtration_curves: list
        A list of average path length filtration curves.
    y: list
        List of graph labels, necessary for classification.

    '''
    dataset = args.dataset
    file_path = "../data/unlabeled_datasets/" + dataset + "/"
    
    # This section is the normal section to load the graphs.
    filenames = sorted(glob.glob(os.path.join(file_path, '*.pickle')))
    
    graphs = [ig.read(filename, format='picklez') for filename in tqdm(filenames)]
    y = [graph['label'] for graph in graphs]
    
    # Calculate average path length for each graph
    avg_path_lengths = [graph.average_path_length() for graph in tqdm(graphs)]
    
    # Binarize the average path lengths to create filtration curves
    max_path_length = max(avg_path_lengths)
    filtration_curves = [np.where(np.array(avg_path_lengths) <= length, 1, 0) for length in tqdm(avg_path_lengths)]
    
    # Pad the filtration curves with zeros to have consistent lengths
    max_length = max([len(curve) for curve in filtration_curves])
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
    
        # run the random forest
        run_rf(filtration_curves, y, n_iterations=10)

    elif args.method == "inductive":
        # format the curves as a dataframe
        filtration_curves = [pd.DataFrame(i) for i in tqdm(filtration_curves)]

        # get the column names (just a single one here)
        column_names = filtration_curves[0].columns.tolist()

        # run the random forest
        run_rf_inductive(filtration_curves, y, column_names=column_names)

    print("Execution time:", time.process_time() - start, "seconds")


