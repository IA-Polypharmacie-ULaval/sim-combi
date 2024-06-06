import argparse
import json
import logging
import os
import random
import math
import time

import numpy as np
import pandas as pd
import scipy as sp
import torch

logging.basicConfig(level=logging.DEBUG)


def parse_args():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "--config",
        default="configs/config.json",
        help="Path to the configuration file (JSON)",
    )
    parser.add_argument(
        "--seed",
        type=int,
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

    return seed


def check_gpu(config):
    if torch.cuda.is_available() and config["use_gpu"]:
        logging.info("Torch detected a CUDA device. Using GPU for data generation...")
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    else:
        logging.info(
            "Torch did not detect a CUDA device or use_gpu=0 in config. Data generation will not use GPU..."
        )


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
    mean_rx = config["patterns"]["mean_rx"]
    n_rx = config["n_rx"]

    logging.info("Generating patterns and their risks...")

    p = mean_rx / n_rx
    size_pattern = (n_patterns, n_rx)

    prob_matrix = torch.full(size_pattern, p)

    patterns = torch.bernoulli(prob_matrix)

    patterns = regen_bad_combis(patterns, p)

    # Generate risks in [min_rr, max_rr] for each pattern
    risks = (min_rr - max_rr) * torch.rand(n_patterns) + max_rr

    return patterns, risks


def save_patterns(patterns, risks, config, seed):
    """Save pattern to file. File path is dictated by output_dir, file_identifier and seed* in config.
    *Seed can be overwritten from arguments.


    Args:
        patterns (torch.Tensor):  tensor of generated patterns
        risks (torch.Tensor): tensor of generated risks
        config (dict): dictionnary containing configuration parameters
    """

    directory = config["output_dir"]
    filename = f"patterns/{config['file_identifier']}_{seed}.json"
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
    mean_rx = config["mean_rx"]
    n_combi = config["n_combi"]

    logging.info("Generating combinations and their risks...")

    p = mean_rx / n_rx
    size_combi = (n_combi, n_rx)

    prob_matrix = torch.full(size_combi, p)
    combinations = torch.bernoulli(prob_matrix)

    combinations = regen_bad_combis(combinations, p)

    risks, inter_bool, dists = generate_risks(
        combinations, patterns, patterns_risks, config
    )

    return combinations, risks, inter_bool, dists


def save_combinations(
    combinations, risks, config, seed, inter_bool=None, dists=None, format_="csv"
):
    """Save combinations and their respective risks in the given format

    Args:
        combinations (torch.Tensor): tensor of combinations (2D)
        risks (torch.Tensor): tensor of risks (1D)
        config (dict): dictionnary containing configuration parameters
        inter_bool (torch.Tensor, optional): tensor of booleans indicating interesection between combinnations and patterns (1D). Defaults to None.
        dists (torch.Tensor, optional): tensor of distances to nearest pattern (1D). Defaults to None.
        format_ (str, optional): Format in which to save the combinations/risks. Defaults to "csv".
    """

    directory = config["output_dir"]
    filename = f"combinations/{config['file_identifier']}_{seed}.csv"
    out_path = f"{directory}/{filename}"

    header = [f"Rx{i}" for i in range(combinations.shape[1])] + ["risk"]
    concat = torch.cat((combinations, risks.unsqueeze(1)), dim=1)

    if inter_bool is not None:
        concat = torch.cat((concat, inter_bool.unsqueeze(1)), dim=1)
        header += ["inter"]
    if dists is not None:
        concat = torch.cat((concat, dists), dim=1)
        header += ["dist"]

    concat = concat.cpu().numpy()
    logging.info(f"Saving combinations at {out_path}")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savetxt(out_path, concat, fmt="%.2f", delimiter=",", header=",".join(header))


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

    # Unrelated combis are disjointed from patterns
    disjoint_mean = config["disjoint_combinations"]["mean_rr"]
    disjoint_std = config["disjoint_combinations"]["std_rr"]

    # Related combis have at least one common Rx with patterns
    inter_std = config["inter_combinations"]["std_rr"]
    adjust_factor = config["inter_combinations"]["adjust_factor"]

    logging.info("Generating risks for combinations")

    # For each combination, find the nearest pattern
    dists = torch.cdist(combinations, patterns, p=1)
    knn_dist, knn_idx = torch.topk(dists, 1, dim=1, largest=False)
    n_rx_combi_pat = (combinations + patterns[knn_idx].squeeze()).sum(dim=1)
    # Find out where combinations have nothing in common with the nearest pattern
    # If the hamming distance is exactly equal to the sum of both rows, then we have that case.
    disjoint_bool = torch.where(
        knn_dist.squeeze() == n_rx_combi_pat,
        True,
        False,
    )

    # Get the idx of combinations intersecting with their nearest pattern
    inter_bool = torch.logical_not(disjoint_bool)
    # Ids of the closest pattern to each combination that have intersections
    knn_idx_inter = knn_idx[inter_bool].squeeze()

    # Exact pattern match
    exact_match_bool = torch.where(knn_dist.squeeze() == 0)
    # Ids of the pattern that has an exact match with this combination
    knn_idx_exact = knn_idx[exact_match_bool].squeeze()

    # Get the number of disjoint and not disjoint combinations
    n_disjoint = disjoint_bool.sum()
    n_inter = inter_bool.sum()
    logging.info(f"Number of disjoint combinations {n_disjoint}")
    logging.info(f"Number of intersecting combinations {n_inter}")

    # Sample risks for disjoint combinations
    disjoint_risks = torch.normal(
        mean=torch.full((n_disjoint,), disjoint_mean),
        std=torch.full((n_disjoint,), disjoint_std),
    )

    # Adjust expectation for intersecting combinations
    mean_adjust = knn_dist[inter_bool].squeeze() / n_rx_combi_pat[inter_bool]

    inter_mean = patterns_risks[knn_idx_inter] - adjust_factor * mean_adjust
    inter_risks = torch.normal(
        mean=inter_mean,
        std=torch.full((n_inter,), inter_std),
    )

    risks = torch.empty((n_combi,))
    risks[disjoint_bool] = disjoint_risks
    risks[inter_bool] = inter_risks
    risks[exact_match_bool] = patterns_risks[knn_idx_exact]
    # Ensure no negative risks are possible
    risks = torch.clip(risks, min=0)

    return risks, inter_bool, knn_dist


def n_combi_is_reasonable(config):
    mean_rx_combi = round(config["mean_rx"])
    n_rx_total = config["n_rx"]
    n_req_combis = config["n_combi"]

    f = math.factorial
    n_poss_combis_expec = f(n_rx_total) / (
        f(mean_rx_combi) * f(n_rx_total - mean_rx_combi)
    )

    return n_poss_combis_expec >= n_req_combis


def print_warnings(config):
    if not n_combi_is_reasonable(config):
        logging.warning(
            "Number of requested combis is bigger than the expected possible number of combinations"
        )
        logging.warning(
            "This could lead to an infinite/very long loop of regenerating combinations where there are duplicates"
        )
        time.sleep(2)
        logging.warning("Continuing anyway...")


def print_quick_stats(combinations, c_risks, c_inter_bool, thresh):
    risky = torch.where(c_risks > thresh, True, False)
    risky_values = c_risks[risky]
    n_risky = risky.sum()
    n_risky_and_inter = c_inter_bool[risky].sum()

    logging.info("Quick stats:")
    logging.info(f"Total number of combinations: {len(combinations)}")
    logging.info(f"Number of risky combinations ({thresh=}): {n_risky}")
    logging.info(f"That's {100 * n_risky / len(combinations)}% of the combinations")
    logging.info(
        f"Among the risky combinations, {n_risky_and_inter} are intersecting with a dangerous pattern"
    )
    logging.info(f"Minimum risk value: {risky_values.min()}")
    logging.info(f"Maximum risk value: {risky_values.max()}")


if __name__ == "__main__":
    args = parse_args()
    config = read_config(args.config)
    print_warnings(config)
    seed_value = set_seed(args, config)
    check_gpu(config)
    patterns, p_risks = generate_patterns(config)
    combinations, c_risks, c_inter_bool, c_dists = generate_combinations(
        config, patterns, p_risks
    )

    save_patterns(patterns, p_risks, config, seed_value)
    save_combinations(
        combinations,
        c_risks,
        config,
        seed_value,
        c_inter_bool,
        c_dists,
    )
    print_quick_stats(combinations, c_risks, c_inter_bool, 1.1)
    logging.info("Finished generating dataset!")
