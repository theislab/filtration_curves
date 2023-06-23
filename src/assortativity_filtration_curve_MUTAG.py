''' using assortativity coeff. as the graph discriptor function'''
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
    Creates the filtration curves using the assortativity coefficient as the graph descriptor.

    Calculates the assortativity coefficient for each graph and converts it into a filtration curve by binarizing the values.

    Parameters:
    args (dict): Command line arguments, used to determine the dataset.

    Returns:
    filtration_curves (list): A list of assortativity coefficient filtration curves.
    y (list): List of graph labels, necessary for classification.
    '''

    dataset = args.dataset
    file_path = "../data/unlabeled_datasets/" + dataset + "/"

    # Load graphs
    filenames = sorted(glob.glob(os.path.join(file_path, '*.pickle')))
    graphs = [ig.read(filename, format='picklez') for filename in tqdm(filenames)]
    y = [graph['label'] for graph in graphs]

    # Calculate assortativity coefficient for each graph
    assortativity_coefficients = [graph.assortativity_degree() for graph in tqdm(graphs)]

    # Binarize the assortativity coefficients to create filtration curves
    max_assortativity = max(assortativity_coefficients)
    filtration_curves = [np.where(np.array(assortativity_coefficients) <= coefficient, 1, 0) for coefficient in tqdm(assortativity_coefficients)]

    # Pad the filtration curves with zeros to have consistent lengths
    max_length = max([len(curve) for curve in filtration_curves])
    filtration_curves = [np.pad(curve, (0, max_length - len(curve)), mode='constant') for curve in tqdm(filtration_curves)]
    #breakpoint()
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


