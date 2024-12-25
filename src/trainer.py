import torch
import torch.nn as nn
import numpy as np
from torch_geometric.loader import DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_fscore_support



class Trainer:
    def __init__(self, dataset, model_class, config, num_folds=5, patience=10, max_epochs=300, save_path='./', save_name='model'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset = dataset
        self.model_class = model_class
        self.config = config
        self.num_folds = num_folds
        self.patience = patience
        self.max_epochs = max_epochs
        self.save_path = save_path
        self.save_name = save_name
        self.results = []
        self.kfold = KFold(n_splits=self.num_folds, shuffle=True, random_state=42)

    def test(self, loader, model):
        """Evaluate the model on the test loader."""
        model.eval()
        y_true, y_pred = [], []
        for data in loader:
            
            data = data.to(self.device)
            out = model(data.x, data.edge_index, data.batch)
            preds = out.argmax(dim=1).cpu().numpy()  # Get predicted classes
            labels = data.y.cpu().numpy()  # Get true labels
            y_pred.extend(preds)
            y_true.extend(labels)
        
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted',zero_division=0)
        accuracy = np.sum(np.array(y_true) == np.array(y_pred)) / len(y_true)
        return accuracy, precision, recall, f1

    def train_fold(self, fold, train_ids, test_ids):
        """Train the model for a given fold."""
        print(f'Fold {fold + 1}/{self.num_folds}')
        
        # Prepare data loaders
        train_subset = torch.utils.data.Subset(self.dataset, train_ids)
        test_subset = torch.utils.data.Subset(self.dataset, test_ids)
        train_loader = DataLoader(train_subset, batch_size=self.config["batch_size"], shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=self.config["batch_size"], shuffle=False)
        
        # Initialize model, optimizer, and loss function
        model = self.model_class(self.config).to(self.device)
        # print(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config["lr"])
        criterion = torch.nn.CrossEntropyLoss()

        best_val_f1 = None
        trigger_times = 0

        for epoch in range(self.max_epochs):
            model.train()
            running_loss = 0.0

            for data in train_loader:
                data = data.to(self.device)
                optimizer.zero_grad()
                out = model(data.x, data.edge_index, data.batch)
                loss = criterion(out, data.y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            # Early stopping and validation check
            _, _, _, val_f1 = self.test(test_loader, model)
            if best_val_f1 is None or val_f1 > best_val_f1:
                best_val_f1 = val_f1
                trigger_times = 0
                model_path = self.config['path'] + '/' + self.config['name'] + '/' + f'best_model_fold_{fold}.pth'
                torch.save(model.state_dict(), model_path)
            #    print(f'Validation F1 improved at epoch {epoch}, model saved')
            else:
                trigger_times += 1
                print(f'Early stopping counter: {trigger_times}/{self.patience}')
                if trigger_times >= self.patience:
                    print(f'Early stopping at epoch {epoch} for fold {fold + 1}')
                    break

        return model, test_loader

    def cross_validate(self):
        """Perform K-fold cross-validation."""
        for fold, (train_ids, test_ids) in enumerate(self.kfold.split(self.dataset)):
            model, test_loader = self.train_fold(fold, train_ids, test_ids)
            model_path = self.config['path'] + '/' + self.config['name'] + '/' + f'best_model_fold_{fold}.pth'
            model.load_state_dict(torch.load(model_path,weights_only=True))  # Load the best model for this fold
            accuracy, precision, recall, f1 = self.test(test_loader, model)
            self.results.append((accuracy, precision, recall, f1))
            print(f'Fold {fold + 1} - Test Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')

    def print_average_results(self):
        """Print the average results across all folds."""
        average_results = np.mean(self.results, axis=0)
        print(f'\nAverage across folds - Accuracy: {average_results[0]:.4f}, Precision: {average_results[1]:.4f}, Recall: {average_results[2]:.4f}, F1: {average_results[3]:.4f}')
        return list(average_results)



class Trainer_mlp:
    def __init__(self, dataset, model_class, config, in_dim,num_folds=5, patience=10, max_epochs=300, save_path='./', save_name='model'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset = dataset
        self.model_class = model_class
        self.config = config
        self.config['input_dim']  = in_dim
        self.num_folds = num_folds
        self.patience = patience
        self.max_epochs = max_epochs
        self.save_path = save_path
        self.save_name = save_name
        self.results = []
        self.kfold = KFold(n_splits=self.num_folds, shuffle=True, random_state=42)

    def test(self, loader, model):
        """Evaluate the model on the test loader."""
        model.eval()
        y_true, y_pred = [], []
        for data in loader:
            x = data[0].to(self.device)
            out = model(x)
            preds = out.argmax(dim=1).cpu().numpy()  # Get predicted classes
            labels =  data[1].cpu().numpy()  # Get true labels
            y_pred.extend(preds)
            y_true.extend(labels)
        
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted',zero_division=0)
        accuracy = np.sum(np.array(y_true) == np.array(y_pred)) / len(y_true)
        return accuracy, precision, recall, f1

    def train_fold(self, fold, train_ids, test_ids):
        """Train the model for a given fold."""
        print(f'Fold {fold + 1}/{self.num_folds}')
        
        # Prepare data loaders
        train_subset = torch.utils.data.Subset(self.dataset, train_ids)
        test_subset = torch.utils.data.Subset(self.dataset, test_ids)
        train_loader = DataLoader(train_subset, batch_size=self.config["batch_size"], shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=self.config["batch_size"], shuffle=False)
        
        # Initialize model, optimizer, and loss function
        model = self.model_class(self.config).to(self.device)
        # print(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config["lr"])
        criterion = torch.nn.CrossEntropyLoss()

        best_val_f1 = None
        trigger_times = 0

        for epoch in range(self.max_epochs):
            model.train()
            running_loss = 0.0
            

            for data in train_loader:
                x = data[0].to(self.device)      
                optimizer.zero_grad()
                out = model(x)
                loss = criterion(out, data[1].to(self.device))
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            # Early stopping and validation check
            _, _, _, val_f1 = self.test(test_loader, model)
            if best_val_f1 is None or val_f1 > best_val_f1:
                best_val_f1 = val_f1
                trigger_times = 0
                model_path = self.config['path'] + '/' + self.config['name'] + '/' + f'best_model_fold_{fold}.pth'
                torch.save(model.state_dict(), model_path)
            #    print(f'Validation F1 improved at epoch {epoch}, model saved')
            else:
                trigger_times += 1
                print(f'Early stopping counter: {trigger_times}/{self.patience}')
                if trigger_times >= self.patience:
                    print(f'Early stopping at epoch {epoch} for fold {fold + 1}')
                    break

        return model, test_loader

    def cross_validate(self):
        """Perform K-fold cross-validation."""
        for fold, (train_ids, test_ids) in enumerate(self.kfold.split(self.dataset)):
            model, test_loader = self.train_fold(fold, train_ids, test_ids)
            model_path = self.config['path'] + '/' + self.config['name'] + '/' + f'best_model_fold_{fold}.pth'
            model.load_state_dict(torch.load(model_path,weights_only=True))  # Load the best model for this fold
            accuracy, precision, recall, f1 = self.test(test_loader, model)
            self.results.append((accuracy, precision, recall, f1))
            print(f'Fold {fold + 1} - Test Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')

    def print_average_results(self):
        """Print the average results across all folds."""
        average_results = np.mean(self.results, axis=0)
        print(f'\nAverage across folds - Accuracy: {average_results[0]:.4f}, Precision: {average_results[1]:.4f}, Recall: {average_results[2]:.4f}, F1: {average_results[3]:.4f}')
        return list(average_results)


class Trainer_tabtransformer:
    def __init__(self, dataset, model_class, config, num_unique_values_per_categ,num_continuous,num_folds=5, patience=10, max_epochs=300, save_path='./', save_name='model'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset = dataset
        self.model_class = model_class
        self.config = config
        self.num_unique_values_per_categ = num_unique_values_per_categ
        self.num_continuous = num_continuous
      #  self.config['input_dim']  = in_dim
        self.num_folds = num_folds
        self.patience = patience
        self.max_epochs = max_epochs
        self.save_path = save_path
        self.save_name = save_name
        self.results = []
        self.kfold = KFold(n_splits=self.num_folds, shuffle=True, random_state=42)

    def test(self, loader, model):
        """Evaluate the model on the test loader."""
        model.eval()
        y_true, y_pred = [], []
        for data in loader:            
            _x_categ = data[0].to(self.device)
            _x_cont = data[1].to(self.device)
            out = model(_x_categ, _x_cont)
            preds = out.argmax(dim=1).cpu().numpy()  # Use the class with the highest probability.
            labels = data[2].cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(labels)
        
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted',zero_division=0)
        accuracy = np.sum(np.array(y_true) == np.array(y_pred)) / len(y_true)
        return accuracy, precision, recall, f1

    def train_fold(self, fold, train_ids, test_ids):
        """Train the model for a given fold."""
        print(f'Fold {fold + 1}/{self.num_folds}')
        
        # Prepare data loaders
        train_subset = torch.utils.data.Subset(self.dataset, train_ids)
        test_subset = torch.utils.data.Subset(self.dataset, test_ids)
        train_loader = DataLoader(train_subset, batch_size=self.config["batch_size"], shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=self.config["batch_size"], shuffle=False)
        
        # Initialize model, optimizer, and loss function
        # model = self.model_class(self.config).to(self.device)
        # print(model)
        model = self.model_class(
    categories = self.num_unique_values_per_categ,      # tuple containing the number of unique values within each category
    num_continuous = self.num_continuous,                # number of continuous values
    dim = 32,                           # dimension, paper set at 32
    dim_out = self.config['output_dim'],                        # binary prediction, but could be anything
    depth = 6,                          # depth, paper recommended 6
    heads = 8,                          # heads, paper recommends 8
    attn_dropout = 0.1,                 # post-attention dropout
    ff_dropout = 0.1,                   # feed forward dropout
    mlp_hidden_mults = (4, 2),          # relative multiples of each hidden dimension of the last mlp to logits
    mlp_act = nn.ReLU(),                # activation for final mlp, defaults to relu, but could be anything else (selu etc)
 #   continuous_mean_std = cont_mean_std # (optional) - normalize the continuous values before layer norm
).to(self.device)
        
        
        
        
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config["lr"])
        criterion = torch.nn.CrossEntropyLoss()

        best_val_f1 = None
        trigger_times = 0

        for epoch in range(self.max_epochs):
            model.train()
            running_loss = 0.0
            

            for data in train_loader:
              #  x = data[0].to(self.device)  
                _x_categ = data[0].to(self.device)
                _x_cont = data[1].to(self.device)
                optimizer.zero_grad()
                out = model(_x_categ, _x_cont)
                loss = criterion(out, data[2].to(self.device))
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            # Early stopping and validation check
            _, _, _, val_f1 = self.test(test_loader, model)
            if best_val_f1 is None or val_f1 > best_val_f1:
                best_val_f1 = val_f1
                trigger_times = 0
                model_path = self.config['path'] + '/' + self.config['name'] + '/' + f'best_model_fold_{fold}.pth'
                torch.save(model.state_dict(), model_path)
            #    print(f'Validation F1 improved at epoch {epoch}, model saved')
            else:
                trigger_times += 1
                print(f'Early stopping counter: {trigger_times}/{self.patience}')
                if trigger_times >= self.patience:
                    print(f'Early stopping at epoch {epoch} for fold {fold + 1}')
                    break

        return model, test_loader

    def cross_validate(self):
        """Perform K-fold cross-validation."""
        for fold, (train_ids, test_ids) in enumerate(self.kfold.split(self.dataset)):
            model, test_loader = self.train_fold(fold, train_ids, test_ids)
            model_path = self.config['path'] + '/' + self.config['name'] + '/' + f'best_model_fold_{fold}.pth'
            model.load_state_dict(torch.load(model_path,weights_only=True))  # Load the best model for this fold
            accuracy, precision, recall, f1 = self.test(test_loader, model)
            self.results.append((accuracy, precision, recall, f1))
            print(f'Fold {fold + 1} - Test Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')

    def print_average_results(self):
        """Print the average results across all folds."""
        average_results = np.mean(self.results, axis=0)
        print(f'\nAverage across folds - Accuracy: {average_results[0]:.4f}, Precision: {average_results[1]:.4f}, Recall: {average_results[2]:.4f}, F1: {average_results[3]:.4f}')
        return list(average_results)

    
class Trainer_fttransformer:
    def __init__(self, dataset, model_class, config, num_unique_values_per_categ,num_continuous,num_folds=5, patience=10, max_epochs=300, save_path='./', save_name='model'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset = dataset
        self.model_class = model_class
        self.config = config
        self.num_unique_values_per_categ = num_unique_values_per_categ
        self.num_continuous = num_continuous
      #  self.config['input_dim']  = in_dim
        self.num_folds = num_folds
        self.patience = patience
        self.max_epochs = max_epochs
        self.save_path = save_path
        self.save_name = save_name
        self.results = []
        self.kfold = KFold(n_splits=self.num_folds, shuffle=True, random_state=42)

    def test(self, loader, model):
        """Evaluate the model on the test loader."""
        model.eval()
        y_true, y_pred = [], []
        for data in loader:            
            _x_categ = data[0].to(self.device)
            _x_cont = data[1].to(self.device)
            out = model(_x_categ, _x_cont)
            preds = out.argmax(dim=1).cpu().numpy()  # Use the class with the highest probability.
            labels = data[2].cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(labels)
        
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted',zero_division=0)
        accuracy = np.sum(np.array(y_true) == np.array(y_pred)) / len(y_true)
        return accuracy, precision, recall, f1

    def train_fold(self, fold, train_ids, test_ids):
        """Train the model for a given fold."""
        print(f'Fold {fold + 1}/{self.num_folds}')
        
        # Prepare data loaders
        train_subset = torch.utils.data.Subset(self.dataset, train_ids)
        test_subset = torch.utils.data.Subset(self.dataset, test_ids)
        train_loader = DataLoader(train_subset, batch_size=self.config["batch_size"], shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=self.config["batch_size"], shuffle=False)
        
        # Initialize model, optimizer, and loss function
        # model = self.model_class(self.config).to(self.device)
        # print(model)
        model = self.model_class(
    categories = self.num_unique_values_per_categ,      # tuple containing the number of unique values within each category
    num_continuous = self.num_continuous,                # number of continuous values
    dim = 32,                           # dimension, paper set at 32
    dim_out = self.config['output_dim'],                        # binary prediction, but could be anything
    depth = 6,                          # depth, paper recommended 6
    heads = 8,                          # heads, paper recommends 8
    attn_dropout = 0.1,                 # post-attention dropout
    ff_dropout = 0.1,                   # feed forward dropout
).to(self.device)
        
        
        
        
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config["lr"])
        criterion = torch.nn.CrossEntropyLoss()

        best_val_f1 = None
        trigger_times = 0

        for epoch in range(self.max_epochs):
            model.train()
            running_loss = 0.0
            

            for data in train_loader:
              #  x = data[0].to(self.device)  
                _x_categ = data[0].to(self.device)
                _x_cont = data[1].to(self.device)
                optimizer.zero_grad()
                out = model(_x_categ, _x_cont)
                loss = criterion(out, data[2].to(self.device))
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            # Early stopping and validation check
            _, _, _, val_f1 = self.test(test_loader, model)
            if best_val_f1 is None or val_f1 > best_val_f1:
                best_val_f1 = val_f1
                trigger_times = 0
                model_path = self.config['path'] + '/' + self.config['name'] + '/' + f'best_model_fold_{fold}.pth'
                torch.save(model.state_dict(), model_path)
            #    print(f'Validation F1 improved at epoch {epoch}, model saved')
            else:
                trigger_times += 1
                print(f'Early stopping counter: {trigger_times}/{self.patience}')
                if trigger_times >= self.patience:
                    print(f'Early stopping at epoch {epoch} for fold {fold + 1}')
                    break

        return model, test_loader

    def cross_validate(self):
        """Perform K-fold cross-validation."""
        for fold, (train_ids, test_ids) in enumerate(self.kfold.split(self.dataset)):
            model, test_loader = self.train_fold(fold, train_ids, test_ids)
            model_path = self.config['path'] + '/' + self.config['name'] + '/' + f'best_model_fold_{fold}.pth'
            model.load_state_dict(torch.load(model_path,weights_only=True))  # Load the best model for this fold
            accuracy, precision, recall, f1 = self.test(test_loader, model)
            self.results.append((accuracy, precision, recall, f1))
            print(f'Fold {fold + 1} - Test Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')

    def print_average_results(self):
        """Print the average results across all folds."""
        average_results = np.mean(self.results, axis=0)
        print(f'\nAverage across folds - Accuracy: {average_results[0]:.4f}, Precision: {average_results[1]:.4f}, Recall: {average_results[2]:.4f}, F1: {average_results[3]:.4f}')
        return list(average_results)