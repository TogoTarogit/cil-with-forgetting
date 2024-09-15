# CIL with Forgetting

This repository contains code for experimenting with Continual Learning (CIL) using Selective Amnesia (SA) methods on Variational Autoencoders (VAEs). The goal is to selectively forget certain classes in a trained VAE model while retaining the ability to generate other classes. We explore different forgetting methods such as Fine-tuning, Elastic Weight Consolidation (EWC), and Selective Amnesia (SA), and evaluate their effectiveness using a classifier.

## Requirements

Set up the environment using the following commands:

```bash
conda create --name cil-with-forgetting python=3.8
conda activate cil-with-forgetting
pip install -r requirements.txt
```

## Datasets

We use the MNIST or Fashion-MNIST datasets for training and evaluation. Set the `--dataset` argument to either `mnist` or `fashion` as needed.

## Experiment Workflow

The typical workflow involves the following steps:

1. **Train the Conditional VAE**: Train a conditional VAE on the dataset, optionally excluding certain classes $C_f$
   .
2. **Calculate the Fisher Information Matrix (FIM)**: Compute the FIM for the trained VAE model, required for EWC.
3. **Apply Forgetting Methods**: Apply different forgetting methods.
4. **Learn with New Classes**: Train the model on new classes, excluding the forgotten classes with EWC and Fine-tuning.
5. **Generate Samples**: Generate samples from the trained or modified VAE models.
6. **Evaluate with Classifier**: Evaluate the performance of the models using a pre-trained classifier.

We provide scripts to automate these steps for different combinations of classes to learn and forget. The `cil-with-forgetting` script automates the entire process, allowing you to run multiple experiments with different configurations.

```bash
bash cil-with-forgetting.sh
```

## Step-by-Step Instructions

### 1. Training the Conditional VAE

Train a conditional VAE on the dataset, excluding any classes you wish to forget later. For example, to exclude class `9`:

```bash
CUDA_VISIBLE_DEVICES="0" python train_cvae.py --remove_label 9 --data_path ./dataset --dataset mnist
```

This command trains the VAE on all classes except class `9`. The trained model will be saved in a directory like `results/yyyy_mm_dd_hhmmss`, where `yyyy_mm_dd_hhmmss` corresponds to the date and time of the training run.

### 2. Calculating the Fisher Information Matrix (FIM)

Compute the FIM for the trained VAE model:

```bash
CUDA_VISIBLE_DEVICES="0" python calculate_fim.py --ckpt_folder results/yyyy_mm_dd_hhmmss
```

The FIM will be saved as `fisher_dict.pkl` in the same directory.

### 3. Applying Forgetting Methods

#### 3.1 Regular Training

##### Fine-tuning

Fine-tune the VAE model without any forgetting method:

```bash
CUDA_VISIBLE_DEVICES="0" python train_finetuning.py --ckpt_folder results/yyyy_mm_dd_hhmmss --removed_label 9 --dataset mnist
```

##### EWC

Apply EWC to the VAE model:

```bash
CUDA_VISIBLE_DEVICES="0" python train_ewc.py --ckpt_folder results/yyyy_mm_dd_hhmmss --removed_label 9 --dataset mnist
```

#### 3.2 Proposed Method : Forgetting with Selective Amnesia (SA)

Apply Selective Amnesia (SA) to the VAE model:

```bash
CUDA_VISIBLE_DEVICES="0" python train_forget.py --ckpt_folder results/yyyy_mm_dd_hhmmss --label_to_drop 9 --lmbda 100 --forgetting_method random --dataset mnist --embedding_label 1
```

- `--label_to_drop`: The label (class) to forget.
- `--forgetting_method`: The method used for forgetting (e.g., `random`).
- `--embedding_label`: The label used for embedding during SA.

##### SA with Fine-tuning

First, apply SA as above, then fine-tune the SA model:

```bash
# Apply SA
CUDA_VISIBLE_DEVICES="0" python train_forget.py --ckpt_folder results/yyyy_mm_dd_hhmmss --label_to_drop 9 --lmbda 100 --forgetting_method random --dataset mnist --embedding_label 1

# and then Fine-tune the SA model
CUDA_VISIBLE_DEVICES="0" python train_finetuning.py --ckpt_folder results/yyyy_mm_dd_hhmmss_sa --removed_label 9 --dataset mnist
```

##### SA with EWC

First, apply SA as above, compute FIM for the SA model, and then apply EWC:

```bash
# Apply SA
CUDA_VISIBLE_DEVICES="0" python train_forget.py --ckpt_folder results/yyyy_mm_dd_hhmmss --label_to_drop 9 --lmbda 100 --forgetting_method random --dataset mnist --embedding_label 1

# Compute FIM for the SA model
CUDA_VISIBLE_DEVICES="0" python calculate_fim.py --ckpt_folder results/yyyy_mm_dd_hhmmss_sa

# Apply EWC to the SA model
CUDA_VISIBLE_DEVICES="0" python train_ewc.py --ckpt_folder results/yyyy_mm_dd_hhmmss_sa --removed_label 9 --dataset mnist
```

### 4. Generating Samples

After training, generate samples from the VAE models for evaluation:

```bash
CUDA_VISIBLE_DEVICES="0" python generate_samples.py --ckpt_folder results/yyyy_mm_dd_hhmmss --label_to_generate 1 --n_samples 10000
```

This command generates 10,000 samples of class `1` and saves them in the model directory.

### 5. Evaluating with a Classifier

First, train a classifier on the dataset:

```bash
CUDA_VISIBLE_DEVICES="0" python train_classifier.py --data_path ./dataset --dataset mnist
```

The classifier model will be saved in the `classifier_ckpts` directory.

Then, evaluate the generated samples:

```bash
CUDA_VISIBLE_DEVICES="0" python evaluate_with_classifier.py --sample_path results/yyyy_mm_dd_hhmmss --label_of_dropped_class 9 --dataset mnist
```

This command outputs metrics such as the average entropy and the average probability of the forgotten class, helping you assess the effectiveness of the forgetting method.

## Automating Experiments

To automate the entire process for various combinations of classes and forgetting methods, you can use the provided shell script `run_experiments.sh`.

### Configuration

Edit the script to set up the experiment parameters:

```bash
# List of classes to train on
list_ewc_learn=(1)
list_forget=(9)

# Experiment settings
cuda_num=1
n_samples=10000
dataset="mnist"  # or "fashion"
forgetting_method="random"
contents_discription="Description of the experiment"
```

- `list_ewc_learn`: Classes to learn.
- `list_forget`: Classes to forget.
- `cuda_num`: GPU device number.
- `n_samples`: Number of samples to generate for evaluation.
- `dataset`: Dataset to use (`mnist` or `fashion`).
- `forgetting_method`: Method used for forgetting (`random`, etc.).
- `contents_discription`: Description of the experiment for logging purposes.

### Running the Script

Make the script executable and run it:

```bash
chmod +x run_experiments.sh
./run_experiments.sh
```

The script will:

- Train VAEs for each class in `list_ewc_learn`, excluding the class to be forgotten.
- Calculate the FIM for each trained VAE.
- For each combination of classes to learn and forget, apply different forgetting methods:
  - Fine-tuning
  - EWC without SA
  - SA only
  - SA with Fine-tuning
  - SA with EWC
- Generate samples and evaluate them using the classifier.
- Save the results and logs in a timestamped file in `results/text_results`.

### Result Logging

The results of the experiments will be saved in `results/text_results/YYYY_MM_DD_HHMMSS_mnist_forget_learn_test.txt`. This file contains:

- The configuration of the experiment.
- The directories of the saved models.
- The evaluation results for each forgetting method.

## Notes

- Replace `mnist` with `fashion` in the `--dataset` argument to use the Fashion-MNIST dataset.
- Adjust the `--label_to_generate` and `--label_of_dropped_class` arguments according to the classes you are focusing on.
- Ensure that you have sufficient computational resources, as generating 10,000 samples and training models can be resource-intensive.

## Project Structure

- `README.md`: The main documentation file providing instructions and details about the project.
- `calculate_fim.py`: Script to compute the Fisher Information Matrix (FIM) for the trained VAE model.
- `cil-with-forgetting.sh`: Shell script to automate experiments involving various forgetting methods and class combinations.
- `evaluate_with_classifier.py`: Script to evaluate generated samples using a pre-trained classifier.
- `fashion.yaml`: Configuration file for training on the Fashion-MNIST dataset.
- `generate_samples.py`: Script to generate samples from the VAE model for evaluation purposes.
- `important_result`: Directory containing important experimental results (could be checkpoints, logs, or evaluations).
- `important_results`: Another directory containing important experimental results; may include different sets of data or analyses.
- `mnist.yaml`: Configuration file for training on the MNIST dataset.
- `model.py`: Python module defining the architecture of the VAE and possibly other neural network models used in the project.
- `requirements.txt`: List of required Python packages needed to set up the project's environment.
- `train_classifier.py`: Script to train a classifier on the dataset, used for evaluating the VAE's sample quality.
- `train_cvae.py`: Script to train the Conditional Variational Autoencoder (CVAE) model.
- `train_ewc.py`: Script to apply Elastic Weight Consolidation (EWC) during training to prevent catastrophic forgetting.
- `train_finetuning.py`: Script to fine-tune the VAE model without any specific forgetting methods.
- `train_forget.py`: Script to apply Selective Amnesia (SA) for intentionally forgetting specific classes in the VAE.
- `utils.py`: Contains utility functions and classes used across various scripts in the project.
- `visualize_10x10.py`: Script to visualize generated samples in a 10x10 grid, useful for inspecting the quality of samples.

---


## Citation

If you use this code in your research, please cite:

```bibtex

```
