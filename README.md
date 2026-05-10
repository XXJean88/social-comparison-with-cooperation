# Social Comparison and the Evolution of Cooperation

This repository contains the code and empirical network datasets used in our study on how social comparison influences the evolution of cooperation on complex networks. The project investigates how different psychological configurations of social comparison affect cooperative behaviour under evolutionary game dynamics.

## Repository Structure

```text
.
├── data/
├── homo/
├── single/
├── hete/
└── opt/
```

### `data/`

This folder contains the empirical network datasets used throughout the study.

Included datasets:

* `ca-netscience (CN).zip`
* `ia-infect-dublin (ON).zip`
* `pone.0136497.s004 (HN).csv`
* `soc-firm-hi-tech (FN).zip`

These datasets represent real-world interaction networks used for both theoretical calculations and stochastic simulations.

---

### `homo/`

This folder contains the codes for the **homogeneous social comparison** setting, where all individuals share the same social comparison parameter.

Files:

* `Theoretical calculation of homogeneous social comparison.py`

  * Performs theoretical analysis of cooperation dynamics under homogeneous social comparison.

* `Simulation of homogeneous social comparison.py`

  * Runs stochastic evolutionary simulations for the homogeneous case.

---

### `single/`

This folder contains the codes for the **single-individual heterogeneous social comparison** setting.

In this scenario, one focal individual possesses a nonzero social comparison parameter, while all other individuals have neutral social comparison.

Files:

* `Theoretical calculation of single-individual homogeneous social comparison.py`

  * Computes the theoretical results for the single-individual heterogeneous case.

* `Simulation of single-individual homogeneous social comparison.py`

  * Performs stochastic simulations for the single-individual heterogeneous case.

---

### `hete/`

This folder contains the codes for the **population-level heterogeneous social comparison** setting, where social comparison parameters are distributed across individuals according to a specified distribution.

Files:

* `Theoretical calculation of heterogeneous social comparison.py`

  * Computes theoretical predictions under distributed heterogeneity.

* `Simulation of Heterogeneous Social Comparison.py`

  * Runs stochastic simulations for heterogeneous social comparison.

---

### `opt/`

This folder contains the optimisation framework used to identify psychological configurations that most strongly promote cooperation.

Files:

* `SASO.py`

  * Implements the Structure-Adaptive Swarm Optimisation (SASO) algorithm proposed in the paper.
  * The algorithm searches for optimal population-level psychological patterns that maximise the evolutionary advantage of cooperation.

---

## Main Components of the Study

The repository investigates three progressively more general scenarios:

1. **Homogeneous social comparison**

   * All individuals share the same comparison tendency.

2. **Single-individual heterogeneity**

   * Only one individual deviates from the population baseline.

3. **Population-level heterogeneity**

   * Social comparison tendencies vary across the entire population.

In addition, the SASO optimisation framework is used to explore which psychological distributions are most favourable for sustaining cooperation.

---

## Requirements

The codes are written in Python.

Typical dependencies include:

* `numpy`
* `networkx`
* `scipy`
* `matplotlib`
* `pickle`

You can install the required packages via:

```bash
pip install numpy networkx scipy matplotlib
```

---

## Running the Codes

Examples:

```bash
python "homo/Simulation of homogeneous social comparison.py"
```

```bash
python "hete/Theoretical calculation of heterogeneous social comparison.py"
```

```bash
python "opt/SASO.py"
```

---

## Notes

* The theoretical scripts compute analytical quantities related to evolutionary dynamics and cooperation conditions.
* The simulation scripts perform stochastic evolutionary game simulations on empirical networks.
* Different datasets can be substituted by modifying the network input section in the corresponding scripts.

---

## Citation

If you use this repository or build upon this work, please cite the corresponding paper.
