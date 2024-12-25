import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, global_add_pool



class GNNModel(torch.nn.Module):
    def __init__(self, config):
        super(GNNModel, self).__init__()
        torch.manual_seed(12345)
      #  torch.manual_seed(42)
        self.config = config
        
        # Choose convolution type based on config
        if config['conv_type'] == 'GCNConv':
            conv_layer = GCNConv
        elif config['conv_type'] == 'SAGEConv':
            conv_layer = SAGEConv
        elif config['conv_type'] == 'GATConv':
            conv_layer = GATConv
        else:
            raise ValueError(f"Unsupported conv layer: {config['conv_type']}")
        
        # Input layer
        self.layers = torch.nn.ModuleList()
        self.layers.append(conv_layer(config['input_dim'], config['hidden_dims'][0],aggr='add'))
        
        # Hidden layers
        for i in range(len(config['hidden_dims']) - 1):
            self.layers.append(conv_layer(config['hidden_dims'][i], config['hidden_dims'][i+1],aggr='add'))
        
        # Output layer
        self.layers.append(Linear(config['hidden_dims'][-1], config['output_dim']))

        # Dropout
        #self.dropout = torch.nn.Dropout(config['dropout'])

    def forward(self, x, edge_index,batch):
        # Apply layers
        for layer in self.layers[:-2]:
            x = layer(x, edge_index)
            x = F.relu(x)
            # x = self.dropout(x)
        
        # Final layer (without activation)
        x = self.layers[-2](x, edge_index)
        x = F.relu(x)

        x = global_add_pool(x, batch)
        
        x = F.dropout(x, p = self.config['dropout'], training = self.training)
        x = self.layers[-1](x)
        return x 