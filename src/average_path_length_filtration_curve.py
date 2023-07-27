# #!/usr/bin/env python
#
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
import matplotlib.pyplot as plt

from rf import *
from utils import *


def create_curves(args):
    '''
    Creates the filtration curves using the average path length as the graph descriptor.

    Calculates the average path length for each graph and converts it into a filtration curve by binarizing the values.

    Parameters:
    args (dict): Command line arguments, used to determine the dataset.

    Returns:
    filtration_curves (list): A list of average path length filtration curves.
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

    # Calculate the average path length for each subgraph (remember that a filtration is a sequence of subgraphs)
    average_path_lengths = []
    for subgraphs in filtrated_graphs:
        average_path_lengths.append([subgraph.average_path_length() for _, subgraph in subgraphs])

    # Determine the maximum length among all lists
    max_length = max(len(lst) for lst in average_path_lengths)

    # Forward fill the lists to make sure they're equally long
    filtration_curves = [
        lst + [lst[-1]] * (max_length - len(lst))
        for lst in average_path_lengths
    ]

    # Get the number of classes
    classes = list(set(y))
    n_classes = len(classes)

    #colors = plt.cm.tab10(np.linspace(0, 1, len(classes)))
    class_colors = ['red', 'green']
    class_color_dict = {1.0: class_colors[0], -1.0: class_colors[1]}
    for i, curve in enumerate(filtration_curves):
        plt.plot(curve, color=class_color_dict[y[i]], alpha=0.7)

    # Add labels and title
    plt.xlabel('Time')
    plt.ylabel('Average Path length')
    plt.title('Filtration Curves')

    # Show the plot
    plt.show()
    
    # Plot the filtration curves for each class separately
    class_colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for c in classes:
        class_indices = [i for i, label in enumerate(y) if label == c]
        class_curves = [filtration_curves[i] for i in class_indices]
        class_colors_subset = [class_colors[i % len(class_colors)] for i in range(len(class_indices))]
        for i, curve in enumerate(class_curves):
            plt.plot(curve, color=class_colors_subset[i], alpha=0.7)
        plt.xlabel('Time')
        plt.ylabel('Average Path Length')
        plt.title(f'Filtration Curves for Class {c}')
        plt.show()

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

