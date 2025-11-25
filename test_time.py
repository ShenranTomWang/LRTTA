import os
import pickle
import torch
import logging
import numpy as np

from models.model import get_model
from utils.misc import print_memory_info
from utils.eval_utils import get_accuracy, eval_domain_dict
from utils.registry import ADAPTATION_REGISTRY
from datasets.data_loading import get_test_loader, get_source_loader
from conf import cfg, load_cfg_from_args, get_num_classes
from methods import *

logger = logging.getLogger(__name__)


def evaluate(description):
    load_cfg_from_args(description)
    valid_settings = [
                      "continual",                  # train on sequence of domain shifts without knowing when a shift occurs
                      "continual_cdc",
                      ]
    assert cfg.SETTING in valid_settings, f"The setting '{cfg.SETTING}' is not supported! Choose from: {valid_settings}"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = get_num_classes(dataset_name=cfg.CORRUPTION.DATASET)

    # setup number of recurrencies
    if cfg.CORRUPTION.DATASET == "ccc":
        num_recur = 1
    else:
        num_recur = cfg.CORRUPTION.RECUR

    if os.path.exists(os.path.join(cfg.SAVE_DIR,f"results_r={num_recur}.pkl")):
        print("Experiment already Done! No Overwriting possible!")
        return
    
    # get the base model and its corresponding input pre-processing (if available)
    base_model, model_preprocess = get_model(cfg, num_classes, device)

    # append the input pre-processing to the base model
    base_model.model_preprocess = model_preprocess

    # setup test-time adaptation method
    available_adaptations = ADAPTATION_REGISTRY.registered_names()
    assert cfg.MODEL.ADAPTATION in available_adaptations, \
        f"The adaptation '{cfg.MODEL.ADAPTATION}' is not supported! Choose from: {available_adaptations}"
    model = ADAPTATION_REGISTRY.get(cfg.MODEL.ADAPTATION)(cfg=cfg, model=base_model, num_classes=num_classes)
    logger.info(f"Successfully prepared test-time adaptation method: {cfg.MODEL.ADAPTATION}")

    # Get domain sequence
    domain_sequence = ["continual_cdc"] if cfg.SETTING == "continual_cdc" else cfg.CORRUPTION.TYPE
    domain_names_all = cfg.CORRUPTION.TYPE
    logger.info(f"Using {cfg.CORRUPTION.DATASET} with the following domain sequence: {domain_sequence}")

    # setup the severities
    severities = cfg.CORRUPTION.SEVERITY
    
    # start evaluation
    logger.info(f"Run {num_recur} recurs")
    all_errs = []
    
    _, source_data_loader = get_source_loader(
        dataset_name=cfg.CORRUPTION.DATASET.replace("_c", ""),
        data_root_dir=cfg.DATA_DIR,
        batch_size=cfg.TEST.BATCH_SIZE,
        train_split=False,
        workers=cfg.TEST.NUM_WORKERS
    )
    acc_source, domain_dict_source, num_samples_source = get_accuracy(
        model,
        data_loader=source_data_loader,
        dataset_name=cfg.CORRUPTION.DATASET.replace("_c", ""),
        domain_name="source",
        print_every=cfg.PRINT_EVERY,
        device=device,
        is_source=True
    )
    torch.cuda.empty_cache()
    logger.info(f"{cfg.CORRUPTION.DATASET.replace("_c", "")} error % [source][before start][#samples={num_samples_source}]: {1-acc_source:.2%}")
    result_dict = {}
    for domain, values in domain_dict_source.items():
            key = f"source_#0"
            result_dict[key] = values
    avg_acc = eval_domain_dict(result_dict)["ACC"]['avg']
    all_errs.append(1-avg_acc)
    logger.info(f"#recur: 0, mean avg error: {1-avg_acc:.2%}\n")
    with open(os.path.join(cfg.SAVE_DIR, f"results_0.pkl"), 'wb') as f:
        pickle.dump(result_dict, f)

    for i_dom, domain_name in enumerate(domain_sequence):
        logger.info(f"Start Domain={domain_name}")
        result_dict = {}

        for r in range(num_recur):
            for severity in severities:
                test_data_loader = get_test_loader(
                    setting=cfg.SETTING,
                    dataset_name=cfg.CORRUPTION.DATASET,
                    data_root_dir=cfg.DATA_DIR,
                    domain_name=domain_name,
                    domain_names_all=domain_names_all,
                    severity=severity,
                    num_examples=cfg.CORRUPTION.NUM_EX,
                    rng_seed=cfg.RNG_SEED+r*len(domain_sequence)+i_dom,
                    batch_size=cfg.TEST.BATCH_SIZE,
                    shuffle=False,
                    workers=cfg.TEST.NUM_WORKERS,
                    preprocess=model_preprocess,
                )

                if cfg.CORRUPTION.DATASET != "ccc" and (r == 0 and i_dom == 0):
                    # Note that the input normalization is done inside of the model
                    logger.info(f"Using the following data transformation:\n{test_data_loader.dataset.transform}")

                # evaluate the model
                acc, domain_dict, num_samples = get_accuracy(
                    model,
                    data_loader=test_data_loader,
                    dataset_name=cfg.CORRUPTION.DATASET,
                    domain_name=domain_name,
                    print_every=cfg.PRINT_EVERY,
                    device=device,
                )
                torch.cuda.empty_cache()

                # Log results
                logger.info(f"{cfg.CORRUPTION.DATASET} error % [{domain_name}][#severity={severity}][#recur={r+1}][#samples={num_samples}]: {1-acc:.2%}")
                for domain, values in domain_dict.items():
                    key = f"{domain}_{severity}_#{r+1}"
                    result_dict[key] = values
        
        _, source_data_loader = get_source_loader(
            dataset_name=cfg.CORRUPTION.DATASET.replace("_c", ""),
            data_root_dir=cfg.DATA_DIR,
            batch_size=cfg.TEST.BATCH_SIZE,
            train_split=False,
            workers=cfg.TEST.NUM_WORKERS
        )
        acc_source, domain_dict_source, num_samples_source = get_accuracy(
            model,
            data_loader=source_data_loader,
            dataset_name=cfg.CORRUPTION.DATASET.replace("_c", ""),
            domain_name="source",
            print_every=cfg.PRINT_EVERY,
            device=device,
            is_source=True
        )
        torch.cuda.empty_cache()
        logger.info(f"{cfg.CORRUPTION.DATASET.replace("_c", "")} error % [source][after domain={domain_name}][#samples={num_samples_source}]: {1-acc_source:.2%}")
        for domain, values in domain_dict_source.items():
            key = f"{domain}_{severity}_#{r+1}"
            result_dict[key] = values

        # Save results for current round
        avg_acc = eval_domain_dict(result_dict)["ACC"]['avg']
        all_errs.append(1-avg_acc)
        logger.info(f"#domain: {domain_name}, mean avg error: {1-avg_acc:.2%}\n")
        with open(os.path.join(cfg.SAVE_DIR, f"results_domain={domain_name}.pkl"), 'wb') as f:
            pickle.dump(result_dict, f)

    # Log final results
    if cfg.TEST.DEBUG:
        print_memory_info()
    logger.info(f"Mean avg error rate: {np.mean(np.array(all_errs)):.2%}\n")


if __name__ == '__main__':
    evaluate('"Evaluation.')
