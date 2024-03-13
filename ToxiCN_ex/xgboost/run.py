import sys
import os

# Get the parent directory of the current script
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the parent directory to the Python path
sys.path.append(parent_dir)
print(parent_dir)

import xgboost as xgb
import numpy as np
import torch
import time
from importlib import import_module
import os
from src.datasets import *
from src.Models import *
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import numpy as np

def custom_eval(preds, dmatrix):
    labels = dmatrix.get_label()
    # Convert probabilities to binary predictions assuming 0.5 as threshold
    # Adjust the threshold as necessary
    binary_preds = np.where(preds > 0.5, 1, 0)
    f1 = f1_score(labels, binary_preds)
    precision = precision_score(labels, binary_preds)
    recall = recall_score(labels, binary_preds)
    # Note that XGBoost requires the return of a list of tuples
    return [('f1', f1), ('precision', precision), ('recall', recall)]

# Function to save encoded data
def save_encoded_data(filepath, X, y):
    torch.save({'X': X, 'y': y}, filepath)

# Function to load encoded data
def load_encoded_data(filepath):
    data = torch.load(filepath)
    return data['X'], data['y']


def encode_data(data_iter, embed_model, device):
    embed_model.eval()  # Set the model to evaluation mode
    embed_model.to(device)  # Ensure the model is on the correct device
    X = []
    y = []

    with torch.no_grad():  # No need to track gradients
        for batch in tqdm(data_iter, desc="Encoding", colour="MAGENTA"):
            # Since batch is a list of dicts, iterate over each dict
            # Process each item's data through the embed_model
            # Assuming embed_model returns the embedding directly. Adjust as necessary.
            args = to_tensor(batch)
            att_input, pooled_emb = embed_model(**args)  
            # print(pooled_emb.shape)        
            # Append the embedding and label to the lists
            X.append(pooled_emb.cpu().numpy())
            for item in batch:
                # Extract input_ids, attention_mask, and labels for each item
                input_ids = torch.tensor(item['text_idx']).unsqueeze(0).to(device)  # Add batch dimension
                attention_mask = torch.tensor(item['text_mask']).unsqueeze(0).to(device)  # Add batch dimension
                

                y.append(item['toxic'][0] == 1)  # Assuming 'toxic' is the label

    # After processing all items, concatenate to form the final datasets
    X = np.concatenate(X, axis=0)
    y = np.array(y)  # Convert list of labels to numpy array
    return X, y

if __name__ == '__main__':
    
    # dataset = 'TOC'  # 数据集
    dataset = "ToxiCN"
    # model_name = "bert-base-chinese"
    model_name = "hfl/chinese-roberta-wwm-ext"
    # model_name = "junnyu/ChineseBERT-base"

    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()    
    print("Loading data...")

    x = import_module('model.' + "Config_base")
    config = x.Config_base(model_name, dataset)  # 引入Config参数，包括Config_base和各私有Config

    if not os.path.exists(config.data_path): 
        trn_data = Datasets(config, config.train_path)
        dev_data = Datasets(config, config.dev_path)
        test_data = Datasets(config, config.test_path)
        torch.save({
            'trn_data' : trn_data,
            'dev_data' : dev_data,
            'test_data' : test_data,
            }, config.data_path)
    else:
        checkpoint = torch.load(config.data_path)
        trn_data = checkpoint['trn_data']
        dev_data = checkpoint['dev_data']
        test_data = checkpoint['test_data']
        print('The size of the Training dataset: {}'.format(len(trn_data)))
        print('The size of the Validation dataset: {}'.format(len(dev_data)))
        print('The size of the Test dataset: {}'.format(len(test_data)))

    train_iter = Dataloader(trn_data,  batch_size=int(config.batch_size), SEED=config.seed)
    dev_iter = Dataloader(dev_data,  batch_size=int(config.batch_size), shuffle=False)
    test_iter = Dataloader(test_data,  batch_size=int(config.batch_size), shuffle=False)


    embed_model = Bert_Layer(config).to(config.device)


    # File paths for the encoded datasets
    encoded_train_path = os.path.join(config.save_path, 'encoded_train_data.pth')
    encoded_dev_path = os.path.join(config.save_path, 'encoded_dev_data.pth')
    encoded_test_path = os.path.join(config.save_path, 'encoded_test_data.pth')

 

    # Check if encoded data already exists, if not, encode and save
    if not os.path.exists(encoded_train_path) or not os.path.exists(encoded_dev_path) or not os.path.exists(encoded_test_path):
        print("Encoding datasets...")
        X_train, y_train = encode_data(train_iter, embed_model, config.device)
        X_dev, y_dev = encode_data(dev_iter, embed_model, config.device)
        X_test, y_test = encode_data(test_iter, embed_model, config.device)
        
        print("Saving encoded datasets...")
        save_encoded_data(encoded_train_path, X_train, y_train)
        save_encoded_data(encoded_dev_path, X_dev, y_dev)
        save_encoded_data(encoded_test_path, X_test, y_test)
    else:
        print("Loading encoded datasets...")
        X_train, y_train = load_encoded_data(encoded_train_path)
        X_dev, y_dev = load_encoded_data(encoded_dev_path)
        X_test, y_test = load_encoded_data(encoded_test_path)

        print(f'Size of encoded Training dataset: {X_train.shape[0]}')
        print(f'Size of encoded Validation dataset: {X_dev.shape[0]}')
        print(f'Size of encoded Test dataset: {X_test.shape[0]}')

    print(X_train.shape, y_train.shape)
    # Convert your dataset to DMatrix format which is optimized for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    ddev = xgb.DMatrix(X_dev, label=y_dev)
    dtest = xgb.DMatrix(X_test, label=y_test)



    # Set up your configuration
    param = {
        'max_depth': 30,  # the maximum depth of each tree
        'eta': 0.3,  # the training step for each iteration
        'silent': 1,  # logging mode - quiet
        'objective': 'binary:logistic',  # binary classification loss function
        'eval_metric': 'auc',  # evaluation metric
    }
    num_round = 500  # the number of training iterations

    # Assuming dtrain, ddev are XGBoost DMatrix objects for training and validation, respectively
    evals = [(dtrain, 'train'), (ddev, 'dev')]
    bst = xgb.train(param, dtrain, num_boost_round=1000, evals=evals, feval=custom_eval, verbose_eval=True)

    # Save the model
    bst.save_model('xgb_model.model')

    # Predictions
    preds = bst.predict(dtest)

    # Get the labels from the test DMatrix
    labels = dtest.get_label()

    # Convert probabilities to binary predictions assuming 0.5 as threshold
    # Adjust the threshold as necessary
    binary_preds = np.where(preds > 0.5, 1, 0)

    # Calculate metrics
    accuracy = accuracy_score(labels, binary_preds)
    f1 = f1_score(labels, binary_preds)
    precision = precision_score(labels, binary_preds)
    recall = recall_score(labels, binary_preds)

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test F1 Score: {f1:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")

