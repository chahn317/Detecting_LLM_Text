# Import necessary libraries
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from tqdm import tqdm
import pickle
import gc
import numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from collections import defaultdict



def generate_embeddings(path, file_name):
    """
    Generate and save embeddings and optionally KL divergences from a model's hidden states.
    """
    print("generate embedding for " + path + file_name)
    def last_token_pool(last_hidden_states, attention_mask):
        """
        Extract embedding for the last valid token in the sequence.
        Handles left-padding cases.
        """
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device='cpu'), sequence_lengths]

    def get_kl_and_embeddings(input_texts):
        """
        Tokenize text, pass through model, extract embeddings and optionally compute KL divergence.
        """
        # Get token length for debugging
        num_tokens = tokenizer(input_texts, return_tensors='pt', truncation=False, padding=False)['input_ids'].shape[1]

        # Tokenize with padding and truncation
        max_length = 300
        batch_dict = tokenizer(input_texts, max_length=max_length, padding=True, truncation=True, return_tensors='pt')

        with torch.no_grad():
            outputs = model(**batch_dict, output_hidden_states=True)
            # Get logits from first and last hidden layers
            last_logits = model.lm_head(outputs.hidden_states[-1]).squeeze()
            first_logits = model.lm_head(outputs.hidden_states[0]).squeeze()

        # Concatenate embeddings from all layers
        all_embed = [last_token_pool(outputs.hidden_states[i].cpu(), batch_dict['attention_mask']) for i in range(len(outputs.hidden_states))]
        all_embed_concated = torch.concat(all_embed, 1).cpu()

        # Compute KL divergence between intermediate layers and first/last layers
        kls = []
        for i in range(1, len(outputs.hidden_states)-1):
            with torch.no_grad():
                middle_logits = model.lm_head(outputs.hidden_states[i]).squeeze()
            kls.append(
                F.kl_div(F.log_softmax(middle_logits, dim=-1), F.softmax(first_logits, dim=-1), reduction='batchmean').item() +
                F.kl_div(F.log_softmax(middle_logits, dim=-1), F.softmax(last_logits, dim=-1), reduction='batchmean').item()
            )
        return kls, all_embed_concated

    # Load data
    with open(path, 'r') as f:
        data = json.load(f)
    if file_name == 'HC3_en_train':
        data = data[:160]
    elif file_name == 'HC3_en_valid':
        data = data[:20]

    # Process each entry in the dataset
    kls = []
    embeddings = []
    for text_info in tqdm(data):
        text = text_info['text']
        kl, embedding = get_kl_and_embeddings([text])
        if kl is not None:
            kls.append(kl)
            embeddings.append(embedding)


    # Save KL divergences and embeddings
    save_embed_dir = 'save/embeddings/'
    save_kl_dir = 'save/kl_divergence/all_tokens/'
    # Ensure directories exist
    os.makedirs(save_embed_dir, exist_ok=True)
    os.makedirs(save_kl_dir, exist_ok=True)
    pickle.dump(kls, open(save_kl_dir + file_name + '.pkl', 'wb'))
    embeddings = torch.cat(embeddings, dim=0)
    torch.save(embeddings, save_embed_dir + file_name + '.pt')



def get_embeddings_and_labels(file_name, data_path, device, layer_num):
    """
    Loads saved embeddings and labels. If embeddings not found, generates them.
    Handles selection of embeddings based on specific layer or KL divergence.
    """
    labels = torch.load(f'dataset/labels/' + file_name + '.pt').to(device)
    try:
        embeddings = torch.load(f'save/embeddings/' + file_name + '.pt')
    except FileNotFoundError:
        generate_embeddings(data_path, file_name)
        embeddings = torch.load(f'save/embeddings/' + file_name + '.pt')

    # Select specific layer embedding or use max-KL layer
    if layer_num != -1:
        embeddings = embeddings[:, embedding_dim * layer_num: embedding_dim * (layer_num + 1)].to(device)
    else:
        with open(f'save/kl_divergence/all_tokens/' + file_name + '.pkl', 'rb') as f:
            kl = np.array(pickle.load(f))
            idx = kl.argmax(axis=1)
            embeddings = torch.tensor([
                row[(i+1)*embedding_dim:(i+2)*embedding_dim].tolist()
                for row, i in zip(embeddings, idx)
            ]).to(device)

    return embeddings, labels

    

def get_train_eval_data(layer_num, device, train_num=160, valid_num=20):
    """
    Get train and validation data from HC3 dataset.
    """
    train_embeddings, train_labels = get_embeddings_and_labels('HC3_en_train', 'dataset/processed_data/train_valid_data/HC3_en_train.json', device, layer_num)
    valid_embeddings, valid_labels = get_embeddings_and_labels('HC3_en_valid', 'dataset/processed_data/train_valid_data/HC3_en_valid.json', device, layer_num)
    return train_embeddings[:train_num], train_labels[:train_num], valid_embeddings[:valid_num], valid_labels[:valid_num]

def get_test_data(device, dataset_names, layer_num) :
    testset_embeddings, testset_labels, data_and_model_names = [], [], []
    for dataset_name in dataset_names: 
        for model_name in ['gpt3.5', 'gpt4', 'claude3']:
            test_embeddings, test_labels = get_embeddings_and_labels(dataset_name + '_' + model_name, 
                'dataset/processed_data/test_data/' + dataset_name + '_' + model_name + '.json', device, layer_num)
            testset_embeddings.append(test_embeddings)
            testset_labels.append(test_labels)
            data_and_model_names.append(dataset_name + "-" + model_name)
    return testset_embeddings, testset_labels, data_and_model_names


class BinaryClassifier(nn.Module):
    """
    A feedforward neural network for binary classification.
    """
    def __init__(self, input_size, hidden_sizes=[1024, 512], num_labels=2, dropout_prob=0.2):
        super(BinaryClassifier, self).__init__()
        self.num_labels = num_labels
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Dropout(dropout_prob),
                nn.Linear(prev_size, hidden_size),
                nn.Tanh(),
            ])
            prev_size = hidden_size
        self.dense = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_size, num_labels)

    def forward(self, x):
        x = self.dense(x)
        return self.classifier(x)


def train(layer_num, device, hidden_sizes=[1024, 512], droprate=0.4, num_epochs=10, learning_rate=0.003):
    """
    Train a binary classifier on embeddings.
    """
    train_embeddings, train_labels, valid_embeddings, valid_labels = get_train_eval_data(layer_num, device)
    input_size = train_embeddings.shape[1]
    
    model = BinaryClassifier(input_size, hidden_sizes=hidden_sizes, dropout_prob=droprate).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    batch_size = 16

    for epoch in range(num_epochs):
        for i in range(0, len(train_embeddings), batch_size):
            model.train()
            batch_embeddings = train_embeddings[i:i+batch_size].to(device)
            batch_labels = train_labels[i:i+batch_size].to(device)
            outputs = model(batch_embeddings)
            loss = criterion(outputs, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            outputs = model(valid_embeddings)
            _, predicted = torch.max(outputs.data, 1)
            accuracy = (predicted == valid_labels).sum().item() / len(valid_labels)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Validation Accuracy: {accuracy:.4f}")
            
    return model


def group_and_average(name_to_auroc):
    """
    Groups AUROC scores by model type (gpt3.5, gpt4, claude3) and computes the average AUROC for each group.
    """
    grouped = defaultdict(list)
    for name, auroc in name_to_auroc.items():
        name_lower = name.lower()
        if 'gpt3' in name_lower or 'chatgpt' in name_lower:
            grouped['gpt3.5'].append(auroc)
        elif 'gpt4' in name_lower:
            grouped['gpt4'].append(auroc)
        elif 'claude' in name_lower:
            grouped['claude3'].append(auroc)
        else:
            print(f"Warning: could not classify {name}")
    
    avg_aurocs = {}
    for model_type in ['gpt3.5', 'gpt4', 'claude3']:
        if grouped[model_type]:
            avg_aurocs[model_type] = sum(grouped[model_type]) / len(grouped[model_type])
        else:
            avg_aurocs[model_type] = None
    
    return avg_aurocs


def run_single_test(model,test_set,test_label,test_acc,testset_name):
    """
    Runs a single test using a trained model and computes AUROC for the binary classification.
    """
    with torch.no_grad():
        outputs = model(test_set)
        probabilities = torch.softmax(outputs, dim=1)[:, 1]
        auroc = roc_auc_score(test_label.cpu().numpy(), probabilities.cpu().numpy())
        test_acc.append(auroc)
    return auroc

def run_all_tests(model, layer_num, device, dataset_names) :
    """
    Runs the trained model on all test datasets (across models like GPT-3.5, GPT-4, Claude) and computes AUROC.
    """
    testset_embeddings, testset_labels, data_and_model_names = get_test_data(device, dataset_names, layer_num)
    with torch.no_grad():
        name_to_auroc = {}
        for test_embed, test_label, data_and_model_name in zip(testset_embeddings, testset_labels, data_and_model_names):
            auroc = run_single_test(model, test_embed, test_label, [], ' ')
            name_to_auroc[data_and_model_name] = auroc
    return name_to_auroc

def train_and_test_one_layer(layer_num, device, dataset_names = ['pub', 'writing', 'xsum']):
    """
    Trains a classifier using embeddings from a specific model layer, tests it on multiple datasets,
    and computes average AUROC scores grouped by model type.
    """
    model = train(layer_num, device)
    name_to_auroc = run_all_tests(model, layer_num, device, dataset_names)
    avg_aurocs = group_and_average(name_to_auroc)
    return avg_aurocs



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train and evaluate a binary classifier using LLM embeddings.")
    parser.add_argument('--layer_num', type=int, default=-1,
                        help='Layer number to use for embeddings (-1 for KL-divergence-based layer, -2 for all layers)')
    parser.add_argument('--datasets', nargs='+', default=['pub', 'writing', 'xsum'],
                        help='List of dataset names to test on (default: pub, writing, xsum)')
    # parser.add_argument('--epochs', type=int, default=10,
    #                     help='Number of training epochs')
    # parser.add_argument('--dropout', type=float, default=0.4,
    #                     help='Dropout probability')
    # parser.add_argument('--lr', type=float, default=0.003,
    #                     help='Learning rate')
    # parser.add_argument('--train_num', type=int, default=160,
    #                     help='Number of training examples to use')
    # parser.add_argument('--valid_num', type=int, default=20,
    #                     help='Number of validation examples to use')
    args = parser.parse_args()

    embedding_dim = 4096  # Dimensionality of hidden layers
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Choose GPU if available

    # Load the tokenizer and language model from HuggingFace
    pretrained_model_name_or_path = '/storage/huggingface_model/gte-Qwen1.5-7B-instruct'
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True, device_map='auto')
    print(model.hf_device_map)  # Display the model’s device allocation


    if (args.layer_num >= -1) :
        avg_aurocs = train_and_test_one_layer(args.layer_num, device, args.datasets)
        print(avg_aurocs)
    else:
        layer_to_aurocs = {'gpt3.5': [], 'gpt4': [], 'claude3': []}
        layer_nums = []

        for layer_num in range(33):
            print("layer_num =", layer_num)
            layer_nums.append(layer_num)
            avg_aurocs = train_and_test_one_layer(layer_num, device, args.datasets)
            for model_type in ['gpt3.5', 'gpt4', 'claude3']:
                layer_to_aurocs[model_type].append(avg_aurocs[model_type])

        plt.figure(figsize=(8, 6))
        for model_type, aucs in layer_to_aurocs.items():
            plt.plot(layer_nums, aucs, label=model_type)

        plt.xlabel('Layer Number')
        plt.ylabel('Average AUROC')
        plt.title('Layer vs AUROC for Different Models')
        plt.legend()
        plt.grid(True)
        try:
            plt.show()
        except:
            plt.savefig('figures/layer_vs_auroc_plot.png')
            print("Plot saved as figures/layer_vs_auroc_plot.png (GUI not available)")
