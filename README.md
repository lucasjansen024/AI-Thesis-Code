# Predicting VO2 Based on Wearable Sensor Data Using Machine Learning

This repository contains the code and reproducibility materials for the master's thesis:

**Predicting VO2 Based on Wearable Sensor Data Using Machine Learning**

The study evaluates whether wearable-based VO2 prediction generalizes across cycling protocols with different temporal structures. A Temporal Convolutional Network (TCN) and a Gated Recurrent Unit (GRU) network were compared using leave-one-subject-out cross-validation.

## Repository structure

AI Thesis Code Folder/ 
├── src2_training_ready/ 
│ ├── experiments/ 
│ ├── models/ 
│ ├── processing/ 
│ ├── preparation/ 
│ ├── train_testing/ 
│ ├── loaders/ 
│ └── estimators/ 
├── data/ 
│ ├── raw/ 
│ ├── prepared/ 
│ └── protocols/ 
├── README.md 
├── requirements.txt 
└── .gitignore


The main folders contain:

src2_training_ready/: data preparation, processing, model training, evaluation, plotting, and analysis code;
data/raw/: original input data, where sharing is permitted;
data/prepared/: processed files used by the model-training pipeline;

## Main analysis

The cross-protocol experiments are defined in:

src2_training_ready/experiments/run_cross_protocol.py

The script contains experiment configurations for:

pooled protocol-inclusive development;
protocol-inclusive evaluation;
leave-one-protocol-out evaluation;
size-matched protocol-control analyses.

The experiment set executed by default is selected in the if __name__ == "__main__" block of run_cross_protocol.py. Users should verify that the selected experiment specification corresponds to the analysis they intend to reproduce.

The models are evaluated using leave-one-subject-out cross-validation.

## Model inputs

Depending on the selected processor, the model inputs can include:

* Work rate
* Heart rate
* Heart-rate reserve
* Ventilation
* Breathing rate
* Tissue saturation index
* Hemoglobin difference
* Gender
* Skeletal muscle mass

The primary input combination used in the final cross-protocol analysis was:

Work rate + gender + skeletal muscle mass + hemoglobin difference

## Software requirements

The original cross-protocol analyses were run using:

- Python 3.11.3
- TensorFlow 2.15.1
- Keras 2.15.0
- CUDA 12.1.1
- NVIDIA A100 GPU

The complete Python package environment is listed in requirements.txt.

Install the required Python packages with:

python -m pip install -r requirements.txt

## Running the analysis

Open a terminal in the repository root, which is the directory containing both src2_training_ready and data.

Run the cross-protocol script with:

python -m src2_training_ready.experiments.run_cross_protocol

The code uses relative paths. The script should therefore be started from the repository root.

Before running the script, check the following section in run_cross_protocol.py:

if __name__ == "__main__":

This section determines:

the experiment specification;
the processor and input combination;
the models to run;
whether plots and existing results are overwritten.


## Expected outputs

Trained models and evaluation results are written to the m/ directory. The folder structure depends on the model, training protocols, target protocol, participant selection, number of epochs, and input processor.

The analysis produces predictions and result files containing performance metrics such as:

root mean squared error;
mean absolute error;
mean squared error;
repeated-measures correlation;
Pearson correlation;
Bland–Altman bias and standard deviation;
participant-wise results;
model predictions;
residual analyses;
generated figures.

Output files are written to the result directories specified in the project configuration.

## Data availability

The dataset contains physiological measurements from human participants. Participant-level data are not publicly shared when this is restricted by privacy or ethical requirements.

To reproduce the complete analysis, the required data must be placed in the `data` directory using the folder structure and filenames expected by the data-loading code.

Where sharing is permitted, anonymized data or example files may be added to this directory.

## Reproducibility

The TensorFlow random seed was set to 42 before model training. Small numerical differences may still occur between runs because of differences in hardware, operating systems, package versions, or non-deterministic TensorFlow operations.

## Thesis author

Lucas Jansen
Master of Science in Artificial Intelligence
Vrije Universiteit Amsterdam
2026

## Citation

When using this code, please cite the accompanying master's thesis:

> Jansen, L. (2026). *Predicting VO2 Based on Wearable Sensor Data Using Machine Learning*. Master's thesis, Vrije Universiteit Amsterdam.
