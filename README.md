# A Meta-Generation Framework for Industrial System Generation: Light Review Version

This repository contains a light version of the code and data related to the paper "A Meta-Generation Framework for Industrial System Generation" intended for review purposes.

## Repository Structure

- `models/`: Contains the saved model architectures in scripts.
- `dataset/`: Contains the dataset used in the experiments and the datasets used in training.
- `saves/` Contains four subfolders named after the model types (e.g., meta_vae, vanilla_vae...), each containing five instances of the same model, trained independently with identical hyperparameters and PyTorch's default initialization for increased robustness.
- `notebooks/`: Contains two Jupyter notebooks:
    - `dataset_generation.ipynb`: Used to generate and save the dataset explained in the paper.
    - `experiments.ipynb`: Used to load the models, plot different generations, distributions, error histograms as explained in the paper.

## Getting Started

1. Clone this repository:
```
git clone https://github.com/HiddenContributor/A_Meta_Generation_framework_for_Industrial_System_Generation.git
cd A_Meta_Generation_framework_for_Industrial_System_Generation
```

2. Create a virtual environment and install the required packages:
```
python -m venv venv
source venv/bin/activate # On Windows, use venv\Scripts\activate
pip install -r requirements.txt
```

3. Navigate to the `notebooks` folder and run the Jupyter notebooks:

To generate the datasets, open and run the "dataset_generation.ipynb" notebook. To generate the experiment visuals, open and run the "experiments.ipynb" notebook. Please note that the figures are displayed within the notebooks.

## Dependencies

```
The following libraries are required to run the code:

- numpy==1.24.2
- pandas==1.5.3
- torch==1.13.1
- tqdm==4.65.0
- matplotlib==3.7.1
- scipy==1.10.0
```

