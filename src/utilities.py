import pandas as pd
import numpy as np
from scipy.stats import entropy
from collections import Counter
import matplotlib.pyplot as plt 
import networkx as nx
import torch
import csv
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,SAGEConv,GATConv,RGCNConv
from torch_geometric.nn import global_mean_pool,global_add_pool,global_max_pool,max_pool_x


def load_dataset_tabtransformer(path, data_name, label_name):
    
    feature_file = 'processed_data.csv'
    feature_path = path + '/' + data_name + '/' + feature_file        
    x_categ,x_cont,labels = read_features_labels_tabtransformer(feature_path, label_name)
    labels = torch.tensor(labels, dtype=torch.long)
    num_unique_values_per_categ = [torch.unique(x_categ[:, col]).numel() for col in range(x_categ.shape[1])]
    num_continuous = x_cont.shape[1]
    dataset = TabtransformerDataset(x_categ,x_cont,labels)
    return dataset,num_unique_values_per_categ,num_continuous


class TabtransformerDataset(Dataset):
    def __init__(self,x_categ,x_cont,labels):
        # You can load your data here
        self.x_categ = x_categ  # A simple tensor dataset
        self.x_cont = x_cont 
        self.labels =labels

    def __len__(self):
        # This defines the total number of samples in your dataset
        return len(self.labels)

    def __getitem__(self, idx):
        # This returns a single sample (data, label) for a given index
        sample_cat = self.x_categ[idx]
        sample_cont = self.x_cont[idx]
        label = self.labels[idx]
        return sample_cat,sample_cont, label


def load_dataset_mlp(path, data_name, label_name):
    
    feature_file = 'processed_data.csv'
    feature_path = path + '/' + data_name + '/' + feature_file        
    features,labels = read_features_labels_baselines(feature_path, label_name)
    labels = torch.tensor(labels, dtype=torch.long)
    dataset = MLPDataset(features,labels)
    return dataset,features.shape[1]

class MLPDataset(Dataset):
    def __init__(self,x,labels):
        # You can load your data here
        self.x = x  # A simple tensor dataset
        self.labels =labels

    def __len__(self):
        # This defines the total number of samples in your dataset
        return len(self.labels)

    def __getitem__(self, idx):
        # This returns a single sample (data, label) for a given index
        sample = self.x[idx]
        label = self.labels[idx]
        return sample, label



    
def load_dataset(path, data_name, label_name, threshold):
    
    feature_file = 'processed_data.csv'
    full_network_file = 'comprehensive_network.csv'

    feature_path = path + '/' + data_name + '/' + feature_file
    df_features = pd.read_csv(feature_path)
    feature_names = list(df_features.drop(columns=[label_name]).columns)    
    
    df = pd.read_csv(path + '/' + data_name + '/' + full_network_file)
    df_filter = (df[df['IG'] >= threshold])
    df_filter = df_filter.drop(columns=['IG', 'MutualInfo'])
    df_filter.to_csv(path + '/' + data_name + '/comprehensive_network_filtered.csv', index=False)
    #G = nx.from_pandas_edgelist(df_filter, 'Feature1', 'Feature2')
    with open(path + '/' + data_name + '/comprehensive_network_filtered.csv', 'r') as file:
        edges = [line.strip().split(',')[0:2] for line in file]
    G = nx.Graph()
    G.add_nodes_from(feature_names)
    G.add_edges_from(edges[1:])
    
    relabel_mapping = {node: i for i, node in enumerate(feature_names)}
    G = nx.relabel_nodes(G, relabel_mapping)
    edge_index = graph_to_coo_tensor(G)
    # print(edge_index)
    
    features,labels = read_features_labels(feature_path, label_name)
    dataset = create_dataset(edge_index,features,labels)
    
    return dataset


def load_dataset_ratio(path, data_name, label_name, ratio):
    
    feature_file = 'processed_data.csv'
    full_network_file = 'comprehensive_network.csv'

    feature_path = path + '/' + data_name + '/' + feature_file
    df_features = pd.read_csv(feature_path)
    feature_names = list(df_features.drop(columns=[label_name]).columns)    
    
    df = pd.read_csv(path + '/' + data_name + '/' + full_network_file)
    df = df.sort_values(by='IG', ascending=False)
    if int(df.shape[0] * ratio) < 1:
        df_filter = df[:1]
    else:
        
        df_filter = df[:int(df.shape[0] * ratio)]
    #df_filter = (df[df['IG'] >= threshold])
    df_filter = df_filter.drop(columns=['IG', 'MutualInfo'])
    df_filter.to_csv(path + '/' + data_name + '/comprehensive_network_filtered_ratio.csv', index=False)
    #G = nx.from_pandas_edgelist(df_filter, 'Feature1', 'Feature2')
    with open(path + '/' + data_name + '/comprehensive_network_filtered_ratio.csv', 'r') as file:
        edges = [line.strip().split(',')[0:2] for line in file]
    G = nx.Graph()
    G.add_nodes_from(feature_names)
    G.add_edges_from(edges[1:])
    
    relabel_mapping = {node: i for i, node in enumerate(feature_names)}
    print(relabel_mapping)
    G = nx.relabel_nodes(G, relabel_mapping)
    print(G.nodes)

    edge_index = graph_to_coo_tensor(G)
    # print(edge_index)
    
    features,labels = read_features_labels(feature_path, label_name)
    dataset = create_dataset(edge_index,features,labels)
    
    return dataset






def load_features_labels_baselines(path, data_name, label_name):
    feature_file = 'processed_data.csv'
    feature_path = path + '/' + data_name + '/' + feature_file
    return read_features_labels_baselines(feature_path, label_name)








def read_and_relabel_edgelist(file_path,relabel_mapping,node_order):
    # Read the file
    with open(file_path, 'r') as file:
        edges = [line.strip().split(',')[0:2] for line in file]
    # Create a graph from the edges
    G = nx.Graph()
    G.add_nodes_from(node_order)
    G.add_edges_from(edges[1:])


    # Relabel nodes
    G = nx.relabel_nodes(G, relabel_mapping)

    

    return G


def graph_to_coo_tensor(graph):
    # Get a list of edges from the graph
    edges = list(graph.edges())

    # Convert edges to COO format
    # Unzip the edges into two tuples, then convert them into numpy arrays
    row, col = zip(*edges)
    coo_matrix = np.array([row, col]).astype('float')

    # Convert the numpy array to a PyTorch tensor
    coo_tensor = torch.tensor(coo_matrix, dtype=torch.long)

    return coo_tensor



def read_csv_as_list(file_path):
    data_list = []
    
    with open(file_path, 'r', newline='') as csvfile:
        csvreader = csv.reader(csvfile)
        
        for row in csvreader:
            # Convert strings to numbers for each element in the row
            numeric_row = [float(cell) if cell.replace('.', '', 1).replace('-', '', 1).isdigit() else cell for cell in row]
            #numeric_row = [float(cell) for cell in row]

            data_list.append(numeric_row)
    return data_list


def create_dataset(edge_index,features,labels):
    dataset = []
    for i in range(len(labels)):
        x = features[i]
        edge_index = edge_index
        y = labels[i]
        data = Data(x=x, edge_index=edge_index,y=y)
        dataset.append(data)
    return dataset


def read_features_labels(file_path,label_index):
    df = pd.read_csv(file_path)   
    df = df.astype('float')
    labels = df[label_index]
    labels = labels.to_numpy()
    df = df.drop(columns=[label_index])
# Distinction is based on the number of different values in the column
    columns = list(df.columns)
    categoric_columns = []
    numeric_columns = []

    for i in columns:
        if len(df[i].unique()) > 5:
            numeric_columns.append(i)
        else:
            categoric_columns.append(i)
    # print('Numerical features: ', numeric_columns)
    # print('Categorical features: ', categoric_columns)    
    
# Convert numeric columns to float64
    df[numeric_columns] = df[numeric_columns].astype('float64')    

# Initialize LabelEncoder
    label_encoder = LabelEncoder()

# Encode categorical features
    df = df.copy()
    for column in df[categoric_columns]:  
        df[column] = label_encoder.fit_transform(df[column])

# Standardize numerical features
    scaler = StandardScaler()
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    features = df.to_numpy()
    features = features.reshape((features.shape[0], features.shape[1],1))

    features = torch.tensor(features, dtype=torch.float)
    features = features.type(torch.FloatTensor)
    labels = labels.reshape(labels.shape[0],1)
    labels = torch.tensor(labels, dtype=torch.long)
    return features,labels

def read_features_labels_baselines(file_path,label_index):
    df = pd.read_csv(file_path)   
    df = df.astype('float')
    labels = df[label_index]
    labels = labels.to_numpy()
    df = df.drop(columns=[label_index])
# Distinction is based on the number of different values in the column
    columns = list(df.columns)
    categoric_columns = []
    numeric_columns = []

    for i in columns:
        if len(df[i].unique()) > 5:
            numeric_columns.append(i)
        else:
            categoric_columns.append(i)
    # print('Numerical features: ', numeric_columns)
    # print('Categorical features: ', categoric_columns)    
    
# Convert numeric columns to float64
    df[numeric_columns] = df[numeric_columns].astype('float64')    

# Initialize LabelEncoder
    label_encoder = LabelEncoder()

# Encode categorical features
    df = df.copy()
    for column in df[categoric_columns]:  
        df[column] = label_encoder.fit_transform(df[column])

# Standardize numerical features
    scaler = StandardScaler()
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    features = df.to_numpy()
    features = torch.tensor(features, dtype=torch.float)
    return features,labels



def read_features_labels_tabtransformer(file_path,label_index):
    df = pd.read_csv(file_path)   
    df = df.astype('float')
    labels = df[label_index]
    labels = labels.to_numpy()
    df = df.drop(columns=[label_index])
# Distinction is based on the number of different values in the column
    columns = list(df.columns)
    categoric_columns = []
    numeric_columns = []

    for i in columns:
        if len(df[i].unique()) > 5:
            numeric_columns.append(i)
        else:
            categoric_columns.append(i)
    # print('Numerical features: ', numeric_columns)
    # print('Categorical features: ', categoric_columns)    
    
# Convert numeric columns to float64
    df[numeric_columns] = df[numeric_columns].astype('float64')    

# Initialize LabelEncoder
    label_encoder = LabelEncoder()

# Encode categorical features
    df = df.copy()
    for column in df[categoric_columns]:  
        df[column] = label_encoder.fit_transform(df[column])

# Standardize numerical features
    scaler = StandardScaler()
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    # features = df.to_numpy()
    # features = torch.tensor(features, dtype=torch.float)
    
    features_C = df[categoric_columns].to_numpy()
    features_C = torch.tensor(features_C, dtype=torch.int)

    features_N = df[numeric_columns].to_numpy()
    features_N = torch.tensor(features_N, dtype=torch.float)
    return features_C,features_N,labels




def get_edge_index(file_path,node_order):
    relabel_mapping = {node: i for i, node in enumerate(node_order)}
    G = read_and_relabel_edgelist(file_path,relabel_mapping,node_order)
    edge_index = graph_to_coo_tensor(G)
    return edge_index


def draw_graph(G):
    """
    Draws the given NetworkX graph G using matplotlib.
    
    Args:
    G (networkx.Graph): A NetworkX graph
    """
    # Draw the graph
    pos = nx.spring_layout(G)  # positions for all nodes - can be random, circular, etc.
    pos = nx.nx_agraph.graphviz_layout(G, prog="neato")
    nx.draw(G, pos, with_labels=True, node_color='skyblue', edge_color='black', node_size=300, font_size=10)

    # Show the plot
    plt.show()
def add_gaussian_noise(X, mean=0, std=0.1):
    noise = np.random.normal(mean, std, X.shape)
    return (X + noise).to(torch.float32)



