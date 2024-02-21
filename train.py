import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric.transforms as T
import torch.nn.functional as F
from torch_geometric.nn import SGConv
from sklearn.metrics.pairwise import haversine_distances
from math import radians
import numpy as np
import pickle
import gzip
import pdb
import os
import scipy.sparse as sp
from haversine import haversine
# Function to evaluate accuracy on a dataset
from data import DataLoader



# def evaluate(model, combined_features, y_true, user_info, city_info, edge_index):
#     model.eval()
#     with torch.no_grad():
#         # social_features = sgc_model(user_indices, edge_index)
#         # semantic_features = nn_model(x_tf_idf)
#         # combined_features = torch.cat([social_features, semantic_features], dim=1)
#         outputs = classifier(combined_features)
#         _, y_pred = outputs.max(1)
#         correct = 0
#         for idx, city_idx in enumerate(y_pred.tolist()):
#             pred_city = user_indices[idx]
#             pred_lat_lon = np.array([city_info[city_idx][0], city_info[city_idx][1]])
#             true_lat_lon = np.array([user_info[pred_city]['lat'], user_info[pred_city]['lon']])
#             distance = haversine_distances([radians(x) for x in pred_lat_lon], [radians(x) for x in true_lat_lon]).item()
#             miles_distance = distance * 3958.8 # Convert from radians to miles
#             if miles_distance <= 100:
#                 correct += 1
#         accuracy = correct / len(y_true)
#     return accuracy



# Assuming you have loaded your data into the following variables
# user_info = ...  # User information with latitude and longitude
# y = ...          # Class assignments for users
# x = ...          # TF-IDF features
# city_info = ...  # City information with latitude and longitude
# adjacency_matrix = ...  # Adjacency matrix of the graph
# data_loader = DataLoader(data_home='./data/cmu', encoding='latin1')
# preprocessed_data = data_loader.load_obj('D:\KnowledgeGeo\data\cmu\dumped_data2.pkl')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
def load_obj(filename):
    # Check if the file exists
    if not os.path.exists(filename):
        raise FileNotFoundError(f"The file {filename} was not found.")

    # Attempt to open and load the file
    try:
        with gzip.open(filename, 'rb') as fin:
            obj = pickle.load(fin)
        return obj
    except Exception as e:
        # Raise an exception with a more descriptive message
        raise IOError(f"An error occurred while loading the file: {e}")

# Load the pre-processed data
try:
    file_path = r'/data/cmu/dumped_data.pkl'  # Replace with your absolute path
    preprocessed_data = load_obj(file_path)
except Exception as e:
    print(e)

preprocessed_data=load_obj('data/cmu/dumped_data.pkl')
# Access different parts of the data
user_info = preprocessed_data['user_info']  # Dictionary with keys 'train', 'dev', 'test'
y = preprocessed_data['y']
x = preprocessed_data['x']
city_info = preprocessed_data['city_info']
adjacency_matrix = preprocessed_data['adjacency_matrix']

# Convert adjacency matrix to PyTorch format
edge_index = torch.tensor(np.array(adjacency_matrix.nonzero()), dtype=torch.long).to(device)

# Initialize random embeddings for each user
num_users = adjacency_matrix.shape[0]
print("num users:", num_users)
embedding_dim = 56  # Choose a suitable embedding dimension
# embedding_dim_semantic=x['train'].shape[1]
embedding_dim_semantic=500
# user_embeddings = torch.rand((num_users, embedding_dim), dtype=torch.float).to(device)

# SGC Model with Embeddings
def sgc_precompute(features, adj, degree):
    # # Ensure the adjacency matrix is in CSR format
    # adj_csr = sp.csr_matrix(adj)
    #
    # # Convert the CSR matrix to COO format
    # adj_coo = adj_csr.tocoo()
    #
    # # Create indices and values for the PyTorch sparse tensor
    # indices = np.vstack((adj_coo.row, adj_coo.col))
    # values = adj_coo.data
    #
    # # Create the PyTorch sparse tensor
    # indices = torch.LongTensor(indices).to(device)
    # values = torch.FloatTensor(values).to(device)
    # shape = torch.Size(adj_coo.shape)
    # adj_torch = torch.sparse.FloatTensor(indices, values, shape).to(device)

    # Perform SGC precomputation
    for i in range(degree):
        # features = torch.spmm(adj_torch, features).to(device)
        features = torch.spmm(adj, features).to(device)
    return features

# precompute= sgc_precompute(user_embeddings,edge_index,degree=3)
#
# class SGC(nn.Module):
#     """
#     A Simple PyTorch Implementation of Logistic Regression.
#     Assuming the features have been preprocessed with k-step graph propagation.
#     """
#
#     def __init__(self, nfeat, nclass):
#         super(SGC, self).__init__()
#
#         self.W = nn.Linear(nfeat, nclass)
#
#     def forward(self, x):
#         h1 = self.W(x)
#         return h1

class SGC(nn.Module):
    def __init__(self, num_users, embedding_dim, out_channels, K=3, dropout_rate=0.3):
        super(SGC, self).__init__()
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.conv1 = SGConv(embedding_dim, out_channels, K=K, cached=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.conv2 = SGConv(out_channels, out_channels, K=K, cached=True)  # Second layer
        self.conv3 = SGConv(out_channels, out_channels, K=K, cached=True)  # Third layer
        self.init_embeddings()

    def init_embeddings(self):
        nn.init.normal_(self.user_embeddings.weight, std=0.2)

    def forward(self, edge_index):
        x = self.user_embeddings.weight
        x = self.conv1(x, edge_index)
        x = self.dropout(x)  # Apply dropout after the first convolution
        x = self.conv2(x, edge_index)  # Apply  the second convolution
        x = self.dropout(x)  # Apply dropout after the second convolution
        x = self.conv3(x, edge_index)  # Apply the third convolution
        return x
class SGC4Content(nn.Module):
    def __init__(self, embedding_dim, out_channels, output= len(city_info),K=3, dropout_rate=0.3):
        super(SGC4Content, self).__init__()
        self.conv1 = SGConv(embedding_dim, out_channels, K=K, cached=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.conv2 = SGConv(out_channels, out_channels, K=K, cached=True)  # Second layer
        self.conv3 = SGConv(out_channels, out_channels, K=K, cached=True)  # Third layer
        self.fc4=nn.Linear(out_channels,output)
        # self.init_embeddings()

    # def init_embeddings(self):
    #     nn.init.normal_(self.user_embeddings.weight, std=0.2)

    def forward(self, x,edge_index):
        # x = self.user_embeddings.weight
        x = self.conv1(x, edge_index)
        x = self.dropout(x)  # Apply dropout after the first convolution
        x = self.conv2(x, edge_index)  # Apply  the second convolution
        x = self.dropout(x)  # Apply dropout after the second convolution
        x = self.conv3(x, edge_index)  # Apply the third convolution
        x=self.fc4(x)
        return x

# Simple Neural Network for TF-IDF features
class SimpleNN(nn.Module):
    def __init__(self, in_features, out_features):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.fc2 = nn.Linear(out_features, out_features)
        self.fc3 = nn.Linear(out_features, out_features)
        self.prelu = torch.nn.PReLU()
    def forward(self, x):
        x= self.fc1(x)
        F.dropout(input=x, p=0.5, training=self.training)
        x=self.prelu(x)
        # x=self.fc2(x)
        # F.dropout(input=x, p=0.5, training=self.training)
        # x = self.prelu(x)
        # x = self.fc3(x)
        # F.dropout(input=x, p=0.5, training=self.training)
        # x = self.prelu(x)
        return x

# Initialize models
# sgc_model=SGC(embedding_dim,embedding_dim).to(device)
sgc_model = SGC(num_users,embedding_dim, embedding_dim).to(device)  # Keeping the same dimension for simplicity
sgc_model4content = SGC4Content(embedding_dim, embedding_dim).to(device)  # Keeping the same dimension for simplicity
nn_model = SimpleNN(x['train'].shape[1], embedding_dim_semantic).to(device)  # Assuming the TF-IDF feature size is known

# Concatenate and classify
class Classifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)

classifier = Classifier(
    embedding_dim+ embedding_dim_semantic,
                        len(city_info)).to(device)  # Output size = number of cities

# Optimization
# optimizer = optim.Adam(list(sgc_model.parameters()) + list(nn_model.parameters()) + list(classifier.parameters()), lr=0.01, weight_decay=5e-4)

optimizer = optim.Adam(sgc_model4content.parameters(), lr=0.01, weight_decay=5e-4)

# Prepare data
def prepare_data(x):
    x = torch.tensor(x.todense(), dtype=torch.float)  # Convert sparse matrix to dense
    return x

x_train_tf_idf= prepare_data(x['train']).to(device)
y_train = torch.tensor(y['train'], dtype=torch.long).to(device)
x_test_tf_idf = prepare_data(x['test']).to(device)
y_test = torch.tensor(y['test'], dtype=torch.long).to(device)
x_dev_tf_idf = prepare_data(x['dev']).to(device)
y_dev = torch.tensor(y['dev'], dtype=torch.long).to(device)
# Calculate indices ranges for each set
train_end_idx = len(y['train'])
print("train length:",train_end_idx)
dev_end_idx = len(y['dev'])
test_end_idx=len(y['test'])

def evaluate(model, combined_features, y_true, user_info, city_info):
        model.eval()
        with torch.no_grad():
            outputs = classifier(combined_features)
            _, y_pred = outputs.max(1)
            count=0
            correct = 0
            city_correct = 0
            # print("Shape of y_true:", y_true.shape)
            for idx, pred_city in enumerate(y_pred):
                true_city = y_true[idx].item()  # Assuming y_true contains the true city indices
                count+=1

                # Check if the predicted city is correct
                if pred_city == true_city:
                    city_correct += 1

                pred_lat_lon = (city_info[pred_city][0], city_info[pred_city][1])
                true_lat_lon = (user_info[idx]['lat'], user_info[idx]['lon'])
                distance = haversine(pred_lat_lon, true_lat_lon)

                if distance <= 161:  # Threshold in km
                    correct += 1

            # overall_accuracy = correct / len(y_true)
            # city_accuracy = city_correct / len(y_true)
            overall_accuracy = correct / count
            city_accuracy = city_correct / count
            return overall_accuracy, city_accuracy

    # model.eval()
    # with torch.no_grad():
    #     outputs = classifier(combined_features)
    #     _, y_pred = outputs.max(1)
    #     correct = 0
    #     for idx, pred_city in enumerate(y_pred):
    #
    #         # pred_lat_lon = np.array([city_info[pred_city][0], city_info[pred_city][1]]).reshape(1,-1)
    #         # true_lat_lon = np.array([user_info[idx]['lat'], user_info[idx]['lon']]).reshape(1,-1)
    #         pred_lat_lon = np.array([city_info[pred_city][0], city_info[pred_city][1]])
    #         true_lat_lon = np.array([user_info[idx]['lat'], user_info[idx]['lon']])
    #         # distance = haversine_distances([radians(x) for x in pred_lat_lon], [radians(x) for x in true_lat_lon]).item()
    #         # print("predi shape:",pred_lat_lon.shape)
    #         # Convert to tuple if they are not already
    #         pred_lat_lon = tuple(pred_lat_lon) if isinstance(pred_lat_lon, list) else pred_lat_lon
    #         true_lat_lon = tuple(true_lat_lon) if isinstance(true_lat_lon, list) else true_lat_lon
    #         # distance = haversine_distances(pred_lat_lon, true_lat_lon).item()
    #         distance = haversine(pred_lat_lon, true_lat_lon)
    #         # miles_distance = distance * 3958.8 # Convert from radians to miles
    #         if distance <= 161:
    #             correct += 1
    #     accuracy = correct / len(y_true)
    # return accuracy


# Training loop
y_train = y_train.argmax(dim=1)  # Convert from one-hot encoding to class indices
y_dev_covert=y_dev.argmax(dim=1)
y_test=y_test.argmax(dim=1)
def custom_loss_function(outputs, y_true, user_info, city_info):
    _, y_pred = outputs.max(1)  # Get predicted city indices

    # Initialize tensors to store coordinates
    batch_size = y_pred.size(0)
    predicted_coords = torch.zeros(batch_size, 2, dtype=torch.float32, device=outputs.device)
    true_coords = torch.zeros(batch_size, 2, dtype=torch.float32, device=outputs.device)

    # Ensure gradients are tracked for predicted coordinates
    predicted_coords.requires_grad_(True)

    # Collect predicted and true coordinates
    for idx, pred_city in enumerate(y_pred):
        true_city_idx = y_true[idx].item()  # True city index

        # Retrieve predicted coordinates
        predicted_coords[idx] = torch.tensor([city_info[pred_city.item()][0], city_info[pred_city.item()][1]],
                                             dtype=torch.float32, device=outputs.device)

        # Retrieve true coordinates
        true_coords[idx] = torch.tensor([user_info[true_city_idx]['lat'], user_info[true_city_idx]['lon']],
                                        dtype=torch.float32, device=outputs.device)

    # Compute Mean Squared Error
    mse_loss = nn.MSELoss()
    loss = mse_loss(predicted_coords, true_coords)
    return loss
def evaluate_label_accuracy(y_true, user_info, city_info):
    total_distance = 0
    count = 0

    for idx, city_idx in enumerate(y_true):
        true_city_idx =y_true[idx].item()  # Convert city index to Python integer

        # Check if the city index and user index are valid
        if true_city_idx < len(city_info) and idx < len(user_info):
            # Retrieve city coordinates
            city_coords = (city_info[true_city_idx][0], city_info[true_city_idx][1])

            # Retrieve user coordinates
            user_coords = (user_info[idx]['lat'], user_info[idx]['lon'])

            # Calculate distance
            distance = haversine(city_coords, user_coords)
            # print(f"Distance for user {idx} to city {true_city_idx}: {distance} miles")  # Debug print

            total_distance += distance
            count += 1
        else:
            print(f"Skipping invalid city index {true_city_idx} or user index {idx}")  # Debug print

    # Check for division by zero
    average_distance = total_distance / count if count > 0 else float('inf')
    return average_distance

# Usage example
y_train_labels = y_train.argmax(dim=1) if y_train.ndim > 1 else y_train
average_distance_train = evaluate_label_accuracy(y_train_labels, user_info['train'], city_info)
print("Average distance for training set:", average_distance_train)

y_dev_labels = y_dev.argmax(dim=1) if y_dev.ndim > 1 else y_dev
average_distance_dev = evaluate_label_accuracy(y_dev_labels, user_info['dev'], city_info)
print("Average distance for development set:", average_distance_dev)

y_test_labels = y_test.argmax(dim=1) if y_test.ndim > 1 else y_test
average_distance_test = evaluate_label_accuracy(y_test_labels, user_info['test'], city_info)
print("Average distance for test set:", average_distance_test)
pdb.set_trace()

for epoch in range(2000):
    # Train on training data
    sgc_model4content.train()
    # sgc_model.train()
    # nn_model.train()
    # classifier.train()
    from scipy.sparse import vstack
    optimizer.zero_grad()
    features = sgc_model4content(torch.cat([x_train_tf_idf, x_dev_tf_idf, x_test_tf_idf], dim=0),edge_index)
    features_train=features[:train_end_idx].to(device)
    loss=F.cross_entropy(features_train,y_train)
    print(
        f'Epoch {epoch + 1}: Training Loss: {loss:.4f}')
    train_overall_accuracy, train_city_accuracy = evaluate(SGC4Content, features_train, y_train,
                                                           user_info['train'], city_info)
    print(
        f'Epoch {epoch + 1}: Train Overall Accuracy: {train_overall_accuracy:.4f}, Train City Accuracy: {train_city_accuracy:.4f}')
    features_dev=features_train[train_end_idx:train_end_idx+dev_end_idx].to(device)
    dev_overall_accuracy, dev_city_accuracy = evaluate(SGC4Content, features_dev, y_dev_covert,
                                                       user_info['dev'], city_info)
    print(
        f'Epoch {epoch + 1}: Dev Overall Accuracy: {dev_overall_accuracy:.4f}, Dev City Accuracy: {dev_city_accuracy:.4f}')
    SGC4Content.step()
    # all_social_features = sgc_model(edge_index)
    # # all_social_features = sgc_model(precompute)
    # all_semantic_features = nn_model(torch.cat([x_train_tf_idf, x_dev_tf_idf, x_test_tf_idf], dim=0)).to(device)
    # # all_semantic_features = nn_model(torch.cat([x_train_tf_idf, x_dev_tf_idf], dim=0)).to(device)
    #
    # # Extract features for the training set
    # social_features_train = all_social_features[:train_end_idx].to(device)
    # semantic_features_train = all_semantic_features[:train_end_idx].to(device)
    # combined_features_train = torch.cat([social_features_train, semantic_features_train], dim=1).to(device)
    # # combined_features_train = semantic_features_train
    # out = classifier(combined_features_train).to(device)
    # # print("Output shape:", out.shape)
    # # print("Output type:", out.dtype)
    # # print("Target shape:", y_train.shape)
    # # print("Target type:", y_train.dtype)
    #
    # loss = F.cross_entropy(out, y_train)
    # print(
    #     f'Epoch {epoch + 1}: Training Loss: {loss:.4f}')
    # train_overall_accuracy, train_city_accuracy = evaluate(classifier, combined_features_train, y_train,
    #                                                    user_info['train'], city_info)
    # print(
    #     f'Epoch {epoch + 1}: Train Overall Accuracy: {train_overall_accuracy:.4f}, Train City Accuracy: {train_city_accuracy:.4f}')
    # # loss = custom_loss_function(out, y_train, user_info['train'], city_info)
    # loss.backward()
    # optimizer.step()
    # sgc_model.eval()
    # nn_model.eval()
    # classifier.eval()
    # # Extract features for the dev set and evaluate
    # social_features_dev = all_social_features[train_end_idx:train_end_idx+dev_end_idx].to(device)
    # semantic_features_dev = all_semantic_features[train_end_idx:train_end_idx+dev_end_idx].to(device)
    # dev_combined_features = torch.cat([social_features_dev, semantic_features_dev], dim=1).to(device)
    # # dev_combined_features = semantic_features_dev
    # # dev_accuracy = evaluate(classifier, dev_combined_features, y_dev, user_info['dev'], city_info)
    # # print(f'Epoch {epoch + 1}: Dev Accuracy: {dev_accuracy:.4f}')
    # social_features_test = all_social_features[train_end_idx + dev_end_idx:train_end_idx + dev_end_idx+test_end_idx].to(device)
    # semantic_features_test = all_semantic_features[train_end_idx + dev_end_idx:train_end_idx + dev_end_idx+test_end_idx].to(device)
    # test_combined_features=torch.cat([social_features_test,semantic_features_test],dim=1).to(device)
    # # accuracy = evaluate(classifier, test_combined_features, y_test, user_info['test'], city_info)
    # # print(f'Accuracy within 100 miles: {accuracy:.4f}')
    # dev_overall_accuracy, dev_city_accuracy = evaluate(classifier, dev_combined_features, y_dev_covert,
    #                                                    user_info['dev'], city_info)
    # print(
    #     f'Epoch {epoch + 1}: Dev Overall Accuracy: {dev_overall_accuracy:.4f}, Dev City Accuracy: {dev_city_accuracy:.4f}')

    # Extract features for the test set and evaluate
    # test_overall_accuracy, test_city_accuracy = evaluate(classifier, test_combined_features, y_test, user_info['test'],city_info)
    # print(
    #     f'Epoch {epoch + 1}: Test Overall Accuracy: {test_overall_accuracy:.4f}, Test City Accuracy: {test_city_accuracy:.4f}')
# Evaluate the model using the previously defined 'evaluate' function
# accuracy = evaluate(classifier, x_tes