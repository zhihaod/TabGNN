from tqdm import tqdm
import pandas as pd
import argparse
import numpy as np
import math
from collections import Counter

def calculate_entropy(probabilities):
    return -np.sum([p * np.log2(p) for p in probabilities if p > 0])

def calculate_conditional_entropy(x, y):
    joint_prob = Counter(zip(x, y))
    total_count = len(x)
    joint_prob = {k: v / total_count for k, v in joint_prob.items()}
    x_prob = Counter(x)
    x_prob = {k: v / total_count for k, v in x_prob.items()}
    
    cond_entropy = 0.0
    for (x_val, y_val), p_xy in joint_prob.items():
        p_x = x_prob[x_val]
        p_y_given_x = p_xy / p_x
        cond_entropy += p_xy * np.log2(p_y_given_x)
    
    return -cond_entropy

def Fun_H_C(n_ctl, n_case):
    if n_ctl == 0 or n_case == 0:
        return 0
    n_ctl_p = n_ctl / (n_ctl + n_case)
    n_case_p = n_case / (n_ctl + n_case)
    if n_ctl_p == 0 or n_case_p == 0:
        return 0
    return n_ctl_p * np.log2(1 / n_ctl_p) + n_case_p * np.log2(1 / n_case_p)

def Fun_H_CA(arr_ctl, arr_case):
    s_ctl = sum(arr_ctl)
    s_case = sum(arr_case)
    total = s_ctl + s_case
    r = 0.0
    for i in range(len(arr_ctl)):
        if arr_ctl[i] == 0 or arr_case[i] == 0:
            continue
        r += (arr_ctl[i] / total) * np.log2(1 / (arr_ctl[i] / (arr_ctl[i] + arr_case[i])))
    for i in range(len(arr_case)):
        if arr_ctl[i] == 0 or arr_case[i] == 0:
            continue
        r += (arr_case[i] / total) * np.log2(1 / (arr_case[i] / (arr_ctl[i] + arr_case[i])))
    return r

def Fun_I_AC(arr_ctl, arr_case):
    return Fun_H_C(sum(arr_ctl), sum(arr_case)) - Fun_H_CA(arr_ctl, arr_case)

def Fun_IG_ABC(m_ctl, m_case):
    mutual = Fun_I_AC(m_ctl.flatten(), m_case.flatten())
    main_1 = Fun_I_AC(np.sum(m_ctl, axis=0), np.sum(m_case, axis=0))
    main_2 = Fun_I_AC(np.sum(m_ctl, axis=1), np.sum(m_case, axis=1))
    return mutual - main_1 - main_2, mutual, main_1, main_2

def bin_continuous_variable(var, bins):
    return np.digitize(var, bins=np.linspace(min(var), max(var), bins))

def ComprehensiveNet_1(v1, v2, f1, f2, l, numOfPermutation, bins=20):
    # Discretize the continuous variables v1 and v2 into specified number of bins
    v1_binned = bin_continuous_variable(v1, bins)
    v2_binned = bin_continuous_variable(v2, bins)

    # Values and indices for f1, f2, and l
    f1_values = range(bins + 1)
    f1_values_index = [[i for i, e in enumerate(v1_binned) if e == val] for val in f1_values]

    f2_values = range(bins + 1)
    f2_values_index = [[i for i, e in enumerate(v2_binned) if e == val] for val in f2_values]

    l_values = [0, 1]
    l_values = l.unique()

    l_values_index = [[i for i in range(len(l)) if l[i] == val] for val in l_values]

    # Initializing matrices
    m_ctl = np.zeros((len(f1_values), len(f2_values)))
    m_case = np.zeros((len(f1_values), len(f2_values)))
    
    for i in range(len(f1_values)):
        for j in range(len(f2_values)):
            m_ctl[i, j] = len(set(f1_values_index[i]).intersection(f2_values_index[j], l_values_index[0]))
            m_case[i, j] = len(set(f1_values_index[i]).intersection(f2_values_index[j], l_values_index[1]))

    # Adjusting m_ctl based on the ratio of cases to controls
    ratio = np.sum(m_case) / np.sum(m_ctl)
    m_ctl *= ratio

    # Calculate information gain and mutual information
    ig, mutual, main1, main2 = Fun_IG_ABC(m_ctl, m_case)

    # Formatting results
    ig = "{:.9f}".format(ig)
    mutual = "{:.9f}".format(mutual)

    return f"{f1},{f2},{ig},{mutual}\n"


def processDF(df,col):
    labels = df[col]
    del df[col]
    features = df.columns
    return labels, features, df



def main(names):
    path = './data'

    ratios = [1.0]
    for name in names:
        for ratio in ratios:

            df = pd.read_csv(path + '/' + name + '/' + 'processed_data' + '.csv')
            labels, snps, data = processDF(df,'label')
            labels = labels.astype('int')

            edges = [[i,j]for i in range(0, len(snps)-1) for j in range(i+1, len(snps))]
            res = []
            for e in tqdm(edges, desc="Processing Edges"):
                res.append(ComprehensiveNet_1(data[:int(data.shape[0]*ratio)][snps[e[0]]], data[:int(data.shape[0]*ratio)][snps[e[1]]], snps[e[0]], snps[e[1]], labels, 0,20))
    
            split_data = [line.strip().split(',') for line in res]
            df = pd.DataFrame(split_data, columns=['Feature1', 'Feature2', 'IG', 'MutualInfo'])
            df['IG'] = df['IG'].astype('float64')
            df['MutualInfo'] = df['MutualInfo'].astype('float64')
            csv_file_path = path + '/' + name + '/' +'comprehensive_network'+'.csv'
            df.to_csv(csv_file_path, index=False)
            print(f"DataFrame saved to {csv_file_path}")


if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Run GNN model training with a specific config")
    parser.add_argument('names', nargs='+', type=str, help='The name(s) of datasets')
    
    
    # Parse the command-line arguments
    args = parser.parse_args()

    # Pass the argument to the main function
    main(args.names)