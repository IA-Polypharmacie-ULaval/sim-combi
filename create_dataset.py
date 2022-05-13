import argparse
import json
import logging
import os
import random

import numpy as np
import pandas as pd
import scipy as sp
import torch

logging.basicConfig(level=logging.DEBUG)


def parse_args():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "--config", default="config.json", help="Path to the configuration file (JSON)"
    )
    parser.add_argument(
        "--seed",
        help="Set seed of rng. Overrides config file's seed.",
    )
    args = parser.parse_args()
    return args


def set_seed(args, config):
    if args.seed is not None:
        seed = args.seed
    else:
        seed = config["seed"]

    logging.info(f"Setting seed to {seed}")

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def check_gpu():
    if torch.cuda.is_available():
        logging.info("Torch detected a CUDA device. Using GPU for data generation...")
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    else:
        logging.info("Torch did not detect a CUDA device. Won't be using GPU...")


def read_config(path_to_config):
    logging.info(f"Loading configuration file: {path_to_config}")
    with open(path_to_config) as file:
        config = json.load(file)
    return config


def generate_patterns(config):
    """Generate patterns from the loaded config

    Args:
        config (dict (JSON)): configuration

    Returns:
        tuple (torch.Tensor, torch.Tensor): patterns and risks tensors
    """
    n_patterns = config["patterns"]["n_patterns"]
    min_rr = config["patterns"]["min_rr"]
    max_rr = config["patterns"]["max_rr"]
    n_rx = config["n_rx"]
    average_rx = config["average_rx"]

    logging.info("Generating patterns and their risks...")

    p = average_rx / n_rx
    size_pattern = (n_patterns, n_rx)

    prob_matrix = torch.full(size_pattern, p)

    patterns = torch.bernoulli(prob_matrix)

    patterns = regen_bad_combis(patterns, p)

    # Generate risks in [min_rr, max_rr] for each pattern
    risks = (min_rr - max_rr) * torch.rand(n_patterns) + max_rr

    return patterns, risks


def save_patterns(patterns, risks, config):
    """Save pattern to file. File path is dictated by output_dir, file_identifier and seed* in config. Seed can be overwritten from arguments.


    Args:
        patterns (torch.Tensor):  tensor of generated patterns
        risks (torch.Tensor): tensor of generated risks
        config (dict): dictionnary containing configuration parameters
    """

    directory = config["output_dir"]
    filename = f"patterns/{config['file_identifier']}_{config['seed']}.json"
    out_path = f"{directory}/{filename}"

    dict_ = {
        f"pattern_{i}": {
            "pattern": patterns[i].tolist(),
            "risk": round(risks[i].item(), 2),
        }
        for i in range(len(patterns))
    }

    logging.info(f"Saving patterns at {out_path}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(dict_, f)


def regen_bad_combis(combinations, p):
    """Regenerate combinations which are either duplicates of others or are filled with 0s.
    Show off recursion skills by adding it to a function. ¯\_(ツ)_/¯

    Args:
        combinations (torch.Tensor): tensor of combinations
        p (p): p in a binomial distribution

    Returns:
        torch.Tensor : tensor of correctly formed combinations
    """

    assert combinations.shape[0] <= (
        2 ** combinations.shape[1]
    ), "Number of requested combinations is bigger than the powerset of Rx"

    logging.info("Regenerating bad combinations...")

    uniques = torch.unique(combinations, dim=0)
    not_zero_idx = torch.where(uniques.sum(dim=1) != 0)
    uniques = uniques[not_zero_idx]

    if uniques.shape == combinations.shape:
        logging.info("Ending recursion for regeneration!")
        return combinations
    else:
        # Find out how many combis need regenerating
        num_to_regen = len(combinations) - len(uniques)
        n_rx = combinations.shape[1]

        # Regenerate new patterns
        prob_matrix = torch.full((num_to_regen, n_rx), p)
        patterns_to_add = torch.bernoulli(prob_matrix)

        # Cat them and recheck for duplicates
        new_combis = torch.cat((uniques, patterns_to_add), dim=0)
        return regen_bad_combis(new_combis, p)


def generate_combinations(config, patterns, patterns_risks):
    """Generate combinations and risks based on configuration

    Args:
        config (dict): dictionnary containing configuration parameters
        patterns (torch.Tensor): tensor of patterns (2D)
        patterns_risks (torch.Tensor): tensor of pattern risks (1D)

    Returns:
        tuple (torch.Tensor, torch.Tensor): tensor of combinations (2D) and tensor of risks (1D)
    """
    n_rx = config["n_rx"]
    average_rx = config["average_rx"]
    n_combi = config["n_combi"]

    logging.info("Generating combinations and their risks...")

    p = average_rx / n_rx
    size_combi = (n_combi, n_rx)

    prob_matrix = torch.full(size_combi, p)
    combinations = torch.bernoulli(prob_matrix)

    combinations = regen_bad_combis(combinations, p)

    risks = generate_risks(combinations, patterns, patterns_risks, config)

    return combinations, risks


def save_combinations(combinations, risks, config, format_="csv"):
    """Save combinations and their respective risks in the given format

    Args:
        combinations (torch.Tensor): tensor of combinations (2D)
        risks (torch.Tensor): tensor of risks (1D)
        config (dict): dictionnary containing configuration parameters
        format_ (str, optional): Format in which to save the combinations/risks. Defaults to "csv".
    """

    directory = config["output_dir"]
    filename = f"combinations/{config['file_identifier']}_{config['seed']}.csv"
    out_path = f"{directory}/{filename}"

    concat = torch.cat((combinations, risks.unsqueeze(1)), dim=1).cpu().numpy()

    logging.info(f"Saving combinations at {out_path}")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savetxt(out_path, concat, fmt="%.2f", delimiter=",")


def generate_risks(combinations, patterns, patterns_risks, config):
    """Generate risks for generated combinations based on distance to patterns.

    Args:
        combinations (torch.Tensor): tensor of combinations (2D)
        patterns (torch.Tensor): tensor of patterns (2D)
        patterns_risks (torch.Tensor): tensor of pattern risks (1D)
        config (dict): dictionnary containing configuration parameters

    Returns:
        torch.Tensor: tensor of risks for the combinations
    """
    # Get variables from config
    n_patterns = config["patterns"]["n_patterns"]
    n_rx = config["n_rx"]
    n_combi = config["n_combi"]
    similarity_std = bool(config["inter_combinations"]["similarity_std"])

    # Unrelated combis are disjointed from patterns
    disjoint_mean = config["disjoint_combinations"]["mean_rr"]
    disjoint_std = config["disjoint_combinations"]["std_rr"]

    # Related combis have at least one common Rx with patterns
    inter_std = config["inter_combinations"]["std_rr"]

    logging.info("Generating risks for combinations")

    # For each combination, find the nearest pattern
    dists = torch.cdist(combinations, patterns, p=1)
    knn_dist, knn_idx = torch.topk(dists, 1, dim=1, largest=False)

    # Find out where combinations have nothing in common with the nearest pattern
    # If the hamming distance is exactly equal to the sum of both rows, then we have that case.
    disjoint_bool = torch.where(
        knn_dist.squeeze() == (combinations + patterns[knn_idx].squeeze()).sum(dim=1),
        True,
        False,
    )

    # Get the idx of combinations intersecting with their nearest pattern
    inter_bool = torch.logical_not(disjoint_bool)

    # Get the number of disjoint and not disjoint combinations
    n_disjoint = disjoint_bool.sum()
    n_inter = inter_bool.sum()

    # Sample risks for disjoint combinations
    disjoint_risks = torch.normal(
        mean=torch.full((n_disjoint,), disjoint_mean),
        std=torch.full((n_disjoint,), disjoint_std),
    )

    knn_idx_inter = knn_idx[inter_bool].squeeze()

    # If `similarity_std` is True, widen the std around the mean inversely proportional to
    # the similarity between the combination its pattern
    # Else, similarity has no influence on std.
    inter_added_std = similarity_std * knn_dist[inter_bool].squeeze() / n_rx

    inter_risks = torch.normal(
        mean=patterns_risks[knn_idx_inter],
        std=torch.full((n_inter,), inter_std) + inter_added_std,
    )

    risks = torch.empty((n_combi))
    risks[disjoint_bool] = disjoint_risks
    risks[inter_bool] = inter_risks

    return risks


if __name__ == "__main__":
    args = parse_args()
    config = read_config(args.config)
    set_seed(args, config)
    check_gpu()
    patterns, p_risks = generate_patterns(config)
    combinations, c_risks = generate_combinations(config, patterns, p_risks)

    save_patterns(patterns, p_risks, config)
    save_combinations(combinations, c_risks, config)
    logging.info("Finished generating dataset!")
