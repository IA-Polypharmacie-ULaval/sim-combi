# Data simulator for polypharmacies / drug combinations

## TL;DR
```python create_dataset.py [--config path/to/config.json --seed your_seed]```

Template of `config.json` in `configs/`

## Example end result
| Rx1 | Rx2 | Rx3 | Rx4 | ... | RxN | RR   |
|-----|-----|-----|-----|-----|-----|------|
| 1   | 0   | 0   | 1   | ... | 0   | 3.00 |
| 0   | 1   | 0   | 1   | ... | 1   | 2.67 |
| 0   | 1   | 1   | 1   | ... | 1   | 3.14 |
| ⋮   | ⋮   | ⋮   | ⋮   | ... | ⋮   | ⋮    |
| 1   | 1   | 1   | 1   | ... | 0  | 1.85|


## Abbreviations
* RR = Relative risk

## Configuration
* `file_identifier`: File identifier for the output data
* `output_dir`: Directory identifier for the output data
* `seed`: Random seed
* `n_combi`: Number of unique drug combinations to produce
* `n_rx`: Number of individual drugs (equals the number of columns in the generated dataset)
* `mean_rx`: Mean number of drugs per combination
* `use_gpu`: Indicate whether to use GPU for data generation, if available

* `patterns`: Sub-configuration for the dangerous patterns
    * `n_patterns`: Number of dangerous patterns to generate
    * `min_rr`: Minimal RR for patterns
    * `max_rr`: Maximal RR for patterns
    * `mean_rx`: Mean number of drugs per dangerous patterns

* `disjoint_combinations`: Sub-configuration for drug combinations disjoint from the dangerous patterns 
    * `mean_rr`: Gaussian mean for the RR of these combinations
    * `std_rr`: Gaussian standard deviation of these combinations


* `inter_combinations`: Sub-configurtion for drug combinations which intersect with dangerous patterns
    * `std_rr`: Gaussian standard deviation of these combinations


## Used Distributions
### Patterns
Here, uniform distributions within the interval [`patterns:min_rr`, `patterns:max_rr`] are used to facilitate the creation of datasets of varying difficulty levels.

### Combinations with Intersection with a Pattern
A normal distribution with a standard deviation of `inter_combinations:std_rr` is used, with a mean calculated based on the similarity between combinations and dangerous patterns.

### Disjoint Combinations of Patterns
A normal distribution with a mean of `disjoint_combinations:mean_rr` and a standard deviation of `disjoint_combinations:std_rr` is used. Combinations related to a pattern will thus be closer to an RR predetermined by the configuration.

## General Idea
1. Generate dangerous patterns and associated risks randomly.
2. Generate combinations.
3. Generate risks based on the similarity between combinations and patterns.

This can be seen as a cut that overflows into other cuts, or as a tree. Each pattern is a root from which several combinations stem. A combination is associated with a pattern if the pattern is its nearest neighbor according to the Hamming distance. However, a combination can be placed in a separate set if no medication is shared between the combination and the nearest pattern.

See our [paper](https://arxiv.org/abs/2212.05190) for more details. 

## Troubleshooting
1. If stuck on "Regenerating bad combinations...", it is possible that the average number of "possible" combinations is smaller than the number of combinations being generated. In other words, the average number of Rx per combination should be increased, otherwise, you'll be stuck in an infinite loop.
To ensure a finite loop, it suffices to have:

$$ C_k(n) = {n \choose k} = \frac{n!}{k!(n-k)!} $$

where $n$ is the number of Rx, $k$ is the average number of Rx per combination. This condition is sufficient but not necessary, as we are working in expectation.

