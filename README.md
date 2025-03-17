# Logic Tensor Network Implementation (LTN_Imp)

## Overview

This repository contains an implementation of **Logic Tensor Networks (LTN)** for **Symbolic Knowledge Injection (SKI)** into machine learning models. It is particularly relevant for **neuro-symbolic AI**, where structured domain knowledge is integrated into data-driven models to enhance interpretability, performance, and robustness.

This implementation has been used in the research paper:

> **Neuro-symbolic AI for Supporting Chronic Disease Diagnosis and Monitoring**\
> *Matteo Magnini, Giovanni Ciatto, Ahmet Emre Kuru, Christel Sirocchi, and Sara Montagna*

For full reproducibility of the results in this paper, please refer to the commit:

> [Commit ](https://github.com/emrekuruu/LTN_Imp/tree/d363aff892f417d770b86253aa8011c4754e8987)[`d363aff892f417d770b86253aa8011c4754e8987`](https://github.com/emrekuruu/LTN_Imp/tree/d363aff892f417d770b86253aa8011c4754e8987)

## Features

- **LTN-based Knowledge Injection**: Implements logic-based constraints in neural networks.
- **Medical AI Applications**: Evaluates SKI on medical datasets, focusing on chronic disease prediction.
- **Performance & Robustness Analysis**: Compares LTN with classical ML models in terms of accuracy, recall, and adherence to clinical guidelines.
- **Reproducible Experiments**: Code used to generate results in the referenced paper.

## Requirements

To set up the environment, install the necessary dependencies using Poetry:

```bash
poetry install
```

## Usage

### Running Experiments

To reproduce the experiments from the referenced paper, use the provided Jupyter Notebook:

> [`examples/medical/diabetes/demo.ipynb`](https://github.com/emrekuruu/LTN_Imp/blob/d363aff892f417d770b86253aa8011c4754e8987/examples/medical/diabetes/demo.ipynb)

This notebook walks through the process of training and evaluating the LTN model, comparing it against classical ML baselines while considering:

- **Predictive performance** (Accuracy, Recall, F1-score, MCC)
- **Logical adherence** to clinical rules
- **Robustness** under data perturbation (noise injection, missing data, label flipping)

## Datasets

The primary dataset used in the experiments is:

- **Pima Indians Diabetes (PID) dataset**: A benchmark dataset for diabetes prediction.

## Model Architectures

The following models are implemented and compared:

- **Classic ML Models:**
  - K-Nearest Neighbors (KNN)
  - Decision Tree (DT)
  - Random Forest (RF)
  - Logistic Regression (LR)
  - Multi-Layer Perceptron (MLP)
- **Logic Tensor Network (LTN)**: Incorporates symbolic knowledge via logical constraints.

## Results Summary

The key findings from the experiments include:

- **LTN improves recall and balanced accuracy**, making it more suitable for medical AI applications where false negatives must be minimized.
- **LTN exhibits higher adherence to clinical rules**, ensuring predictions align with medical guidelines.
- **LTN demonstrates robustness to data perturbations**, maintaining stable performance even with missing or noisy data.

For detailed results, refer to the paper.

## Acknowledgments

Special thanks to the authors of the referenced paper and the contributors of the **LTN_Imp** repository.

## License

This repository is released under the Apache 2.0 license.

---

For any issues or questions, feel free to open an [issue](https://github.com/emrekuruu/LTN_Imp/issues).



## Project structure
Overview:

```bash
<root directory>
├── ltn_imp/             # main package (should be named after your project)
│   ├── __init__.py         # python package marker
│   └── __main__.py         # application entry point
│   └── fuzzy_operators     # folder for all of the fuzzy operators
│   └── parsing             # folder for the parser utilizing NLTK Logic and all the needed files
│   
├── test/                   # test package contains unit tests
│   ├── parsing_tests       # folder for unit tests for parser
│   └── learning_tests      # folder for unit tests for fuzzy operators in optimization
│   
├── .github/                # configuration of GitHub CI
│   └── workflows/          # configuration of GitHub Workflows
│       └──  check.yml       # runs tests on multiple OS and versions of Python
│
├── LICENSE                 # license file (Apache 2.0 by default)
├── pyproject.toml          # declares build dependencies
└── poetry.toml             # Poetry settings
```