# ML based Qubit Design

## Contents
## Contents
- [Features](#features)
- [Contributions](#contributions)
- [Acknowledgments](#acknowledgments)
- [SQuADDS Dataset](#squadds-dataset)
- [Quick Start](#quick-start)
  - [Environment Setup](#environment-setup)
  - [Fermilab Elastic Analysis Facility (EAF) setup](#fermilab-elastic-analysis-facility-eaf-setup)
  - [Large Files](#large-files)
  - [Notebooks](#notebooks)
  - [Potential optimization strategy](#potential-optimization-strategy)
- [Structure](#structure)
- [Within each folder there are three scripts](#within-each-folder-there-are-three-scripts)
- [Completed](#completed)
- [Desired Flow](#desired-flow)
- [ML Deployment](#ml-deployment)
- [Agent Integration](#agent-integration)

## Features
This repo has tools/options to experiment with different configurations of a Multi-Layer Perceptron (MLP) for Qiskit Metal parameter prediction, when given a desired set of hamiltonian parameters. You can use the contained notebooks to explore the following options:

1. **Hyperparameter Optimization**  
   - Compare using **Keras Tuner** to automatically search for and return the best hyperparameters vs. using a predefined set.  

2. **Feature Scaling (0–1 normalization)**  
   - Option to scale all features to the range `[0, 1]` or leave them unscaled.  
   - I strongly recommended scaling because input values can vary drastically in magnitude for these Qiskit metal design parameters, and proper scaling often improves prediction accuracy across all parameters.  
   - Scaling amounts are saved, allowing you to safely reverse the scaling later and recover accurate/real-world predicted values.  

3. **Categorical Data Encoding**  
   - Compare **one-hot encoding** vs. **linear encoding** for categorical Qiskit Metal parameters.  
   - Since an MLP predicts numeric values rather than categories directly, categorical features must be converted into numeric form. These experiments help determine which encoding method gives better performance.
   - Initial experimentation says one-hot encoding is better across the board

## Contributions
Please contact Olivia Seidel at **olivias@fnal.gov** with any questions or comments.

**Contributors:**
- Sara Sussman (Fermilab): Ideas Contributor
- Giuseppe Di Guglielmo (Fermilab): ML expert and advice giver
- Firas Abouzahr (Northwestern): Code tester/bug finder

## Acknowledgments

We gratefully acknowledge this collaboration for developing the model: Taylor Patti, Nicola Pancotti, Enectali Figueroa-Feliciano, Sara Sussman, Olivia Seidel, Firas Abouzahr, and Sadman Ahmed Shanto.

Special thanks to Olivia Seidel and Firas Abouzahr, who were the primary trainers of the model.

## SQuADDS Dataset

This project is built around the public SQuADDS dataset hosted on Hugging Face.

Relevant dataset resources:

- SQuADDS dataset page: [https://huggingface.co/datasets/SQuADDS/SQuADDS_DB](https://huggingface.co/datasets/SQuADDS/SQuADDS_DB)
- SQuADDS dataset file tree: [https://huggingface.co/datasets/SQuADDS/SQuADDS_DB/tree/main](https://huggingface.co/datasets/SQuADDS/SQuADDS_DB/tree/main)
- SQuADDS organization datasets page: [https://huggingface.co/SQuADDS/datasets](https://huggingface.co/SQuADDS/datasets)
- SQuADDS homepage: [https://lfl-lab.github.io/SQuADDS/](https://lfl-lab.github.io/SQuADDS/)
- SQuADDS repository: [https://github.com/LFL-Lab/SQuADDS](https://github.com/LFL-Lab/SQuADDS)
- SQuADDS paper: [https://doi.org/10.22331/q-2024-09-09-1465](https://doi.org/10.22331/q-2024-09-09-1465)
- Hugging Face dataset DOI: `10.57967/hf/1582`

The dataset page lists the SQuADDS data as MIT licensed JSON data on Hugging Face. The model folders in this repo are most directly connected to these SQuADDS subsets:

- `cavity_claw-RouteMeander-eigenmode`
- `coupler-NCap-cap_matrix`
- `qubit-TransmonCross-cap_matrix`

The `model_predict_qubit-TransmonCross-Hamiltonian_params` workflow is derived from the `qubit-TransmonCross-cap_matrix` dataset and then transformed into Hamiltonian-space targets using the SQuADDS tooling.

If you use SQuADDS data or this ML workflow in research, please cite the SQuADDS paper:

```bibtex
@article{Shanto2024squaddsvalidated,
  doi = {10.22331/q-2024-09-09-1465},
  url = {https://doi.org/10.22331/q-2024-09-09-1465},
  title = {{SQ}u{ADDS}: {A} validated design database and simulation workflow for superconducting qubit design},
  author = {Shanto, Sadman and Kuo, Andre and Miyamoto, Clark and Zhang, Haimeng and Maurya, Vivek and Vlachos, Evangelos and Hecht, Malida and Shum, Chung Wa and Levenson-Falk, Eli},
  journal = {{Quantum}},
  volume = {8},
  pages = {1465},
  month = sep,
  year = {2024}
}
```

## Quick Start

### Environment Setup

Do this first. See the [ENV_SETUP](ENV_SETUP.md) documentation.

###  Fermilab Elastic Analysis Facility (EAF) setup
A GPU is reccomended. If you have a Fermilab services account, you can use it to run jupyter notebooks on the EAF using the following instructions:

1. Navigate to [https://eafdocs.fnal.gov/master/index.html](https://eafdocs.fnal.gov/master/index.html) and sign in with your credentials
2. Follow the three steps under **Quickstart**. When navigating to the link in step two, enter your credentials again
3. Click on **Add New Server**
4. Scroll to the bottom of server options and chose the one that says **Fermilab generic notebooks**
5. Select the middle of the three options to choose a GPU, and from the drop down menue chose the largest slot
6. Clone the repo there

Please feel free to contribute instructions if you use a GPU elsewhere.

### Large Files
If you aren't planning on rerunning all of the notebooks and parsing data/retraining things, you will need some additional files and saved models that were already trained.

Some model checkpoints and intermediate datasets are too big for GitHub. You can grab everything you need from this Google Drive folder **[here](https://drive.google.com/drive/folders/1WVHR4b4g1M4xdOUghbwNKrABafRz-YaQ?usp=sharing)**.

The drive contains three directories, each containing a zip file—simply unzip them into each respective folder of your cloned repo. They’re already in `.gitignore`, so you won’t accidentally commit them after adding them, and you can dive straight into the notebooks without rerunning model training or the data-processing steps.

If you have any access issues please send me a quick note at **olivias@fnal.gov**.

### Notebooks

If you have set up the environment and data, then move to the data analysis notebook (`00_`), then the model training (`01_`), etc.

To open your JupyterLab and then the notebooks:

```
conda activate qubit-design-env
jupyter-lab
```

### Potential optimization strategy:
In the *parameters.py* file start off with KERAS_TUNER=True, and after doing an automated hyperparameter search, grab the best hyperparameter values and update the *parameters.py* file. Then rerun with KERAS_TUNER=False to see how the model learns over epoch number. You can also add more epochs here and fine tune hyperparameters.

## Structure
The following three folders contain scripts to use Machine Learning to predict Qiskit design parameters based on target hamiltonian values for each specified design part:

- *model_predict_cavity_claw_RouteMeander_eigenmode*

- *model_predict_coupler_NCap_cap_matrix*

- *model_predict_qubit_TransmonCross_cap_matrix*

## Within each folder there are three scripts:

- *ml_00_data_analysis*:loads the data and parses it into a format to use in the next script

- *ml_01_train_keras*: Trains the model using an MLP

- *ml_hyperparameter_search_analysis*: provides plots to visualize the hyperparameter search, so the user can easily see the best values

- *ml_02_print_results*: Loads the model and makes predictions with it


## Completed
- Three models have been trained with optimized hyperparameters from keras-tuner, each model predicting Qiskit metal parameters for various parts of a transmon cross chip/resonator design
- Different encoding values were tested and optimized for the categorical y parameters in Qiskit Metal
- Different scaling techniques were implemented, where both X and Y are scaled
- Training and validation sets were explicitly seperated
  
## Desired Flow

Below is the desired flow for this project. All three models will be stitched together to predict a Qiskit Metal design when given a set of desired "Top_Level_X" Hamiltonian values. This will be done using the "X_2.0" values that are simulated from the 'y' values predicted with each of the 3 individual models:

![Desired Flow](desired_flow.png)

## ML Deployment

This repo now includes deployment tooling for publishing the trained models to the
SQuADDS Hugging Face organization in two forms:

1. A Docker Space with a stable HTTP API for agent and application use
2. Per-model Hugging Face model repos that store the trained artifacts and a
   machine-readable inference manifest

Local build and publish entrypoints:

```bash
python3 scripts/build_hf_space_bundle.py
python3 scripts/publish_hf_space.py

python3 scripts/build_hf_model_repos.py
python3 scripts/publish_hf_model_repos.py
```

## Agent Integration

Agent-facing request contracts, deployment details, and workflow guidance live in
[ML_AGENT_INTEGRATION.md](ML_AGENT_INTEGRATION.md).
