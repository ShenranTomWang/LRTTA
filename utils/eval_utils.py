import pandas as pd
import torch
import logging
import numpy as np
from typing import Union
import pickle
import os
from collections import deque

logger = logging.getLogger(__name__)


def split_results_by_domain(domain_dict: dict, labels : list, domains : list, predictions: torch.tensor, confs: torch.tensor):
    """
    Separates the label prediction pairs by domain
    Input:
        domain_dict: Dictionary, where the keys are the domain names and the values are lists with pairs [[label1, prediction1], ...]
        data: List containing [images, labels, domains, ...]
        predictions: Tensor containing the predictions of the model
        confs: Tensor containing confidence of predictions
    Returns:
        domain_dict: Updated dictionary containing the domain seperated label prediction pairs
    """
    assert predictions.shape[0] == labels.shape[0], "The batch size of predictions and labels does not match!"

    for i in range(labels.shape[0]):
        if domains[i] in domain_dict.keys():
            domain_dict[domains[i]].append([labels[i].item(), predictions[i].item(), confs[i].item()])
        else:
            domain_dict[domains[i]] = [[labels[i].item(), predictions[i].item(), confs[i].item()]]

    return domain_dict


def eval_domain_dict(domain_dict: dict):
    """
    Print detailed results for each domain. This is useful for settings where the domains are mixed
    Input:
        domain_dict: Dictionary containing the labels and predictions for each domain
        domain_seq: Order to print the results (if all domains are contained in the domain dict)
    """
    result_dict = {"ACC" : {}, "ECE" : {}}
    logger.info(f"Splitting the results by domain...")
    for key in domain_dict:
        label_prediction_arr = np.array(domain_dict[key])  # rows: samples, cols: (label, prediction)
        labels = label_prediction_arr[:, 0]
        preds = label_prediction_arr[:, 1]
        correct = (labels == preds).sum()
        num_samples = label_prediction_arr.shape[0]
        accuracy = correct / num_samples
        result_dict["ACC"][key] = accuracy

    result_dict["ACC"]["avg"] = sum(list(result_dict["ACC"].values())) / len(result_dict["ACC"].keys())
    return result_dict


def flatten_dict(dict_):
    new_dict = {}
    for key, subdict in dict_.items():
        for key2, values in subdict.items():
            new_dict[f"{key}_{key2}"] = values
    return new_dict


def load_error_dict(exp_dir: str, result_file : str = "result.pkl"):
    records = []    
    for seed_folder in os.listdir(exp_dir):
        seed_result_file = os.path.join(exp_dir, seed_folder, result_file)
        if os.path.exists(seed_result_file):
            with open(seed_result_file, "rb") as f:
                result_dict = pickle.load(f)
                result_dict = eval_domain_dict(result_dict)
                result_dict = flatten_dict(result_dict)
                records.append(result_dict)
        else:
            print(f"Warning ! {seed_result_file} does not exist")
    df = pd.DataFrame.from_records(records).mean(axis=0)
    return df


def get_accuracy(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    dataset_name: str,
    domain_name: str,
    print_every: int,
    device: Union[str, torch.device],
    is_source: bool = False
):
    
    num_correct = 0.
    num_samples = 0
    domain_dict = {}

    with torch.no_grad():
        for i, data in enumerate(data_loader):
            imgs, labels = data[0], data[1]
            labels = labels.to(device, dtype=torch.int64)
    
            output = model([img.to(device) for img in imgs], is_source=is_source) if isinstance(imgs, list) else model(imgs.to(device), is_source=is_source)
            predictions = output.argmax(1)
            confs = output.softmax(1).amax(1)

            current_num_correct = (predictions == labels.to(device)).float().sum()
            num_correct += current_num_correct
            current_num_samples = imgs[0].shape[0] if isinstance(imgs, list) else imgs.shape[0]
            num_samples += current_num_samples

            if len(data) >= 3:
                domain_dict = split_results_by_domain(domain_dict, data[1], data[2], predictions, confs)
            else:
                domain_dict = split_results_by_domain(domain_dict, labels, [domain_name]*len(imgs), predictions, confs)

            # track progress
            if print_every > 0 and (i+1) % print_every == 0:
                message = f"domain={domain_name} #batches={i+1:<6} #samples={num_samples:<9} running error = {1 - num_correct/num_samples:.2%}"
                logger.info(message)
        
            if dataset_name == "ccc" and num_samples >= 7500000:
                break

    accuracy = num_correct.item() / num_samples
    return accuracy, domain_dict, num_samples
