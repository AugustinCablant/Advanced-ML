# Efficient Image Classification with Optimal Transport Methods

This repository contains the implementation of a comparative study on image classification using Optimal Transport (OT) methods. The focus of the study is to explore how OT can be used as a novel approach for classifying handwritten digits using the MNIST dataset. We compare the performance of OT-based methods with traditional machine learning algorithms, such as Support Vector Machines (SVM) and Neural Networks, in terms of accuracy, speed, and computational efficiency.


## Project Structure
```plaintext
├── data/                  # Folder containing dataset(s) for training and testing
├── models/                # Directory for storing pretrained models and checkpoints
├── notebooks/             # Jupyter notebooks for exploratory analysis and experimentation
├── src/                   # Source code for data processing, training, and evaluation
│   ├── data_loader.py     # Module for loading and preprocessing text data
│   ├── summarizer.py      # Main module for implementing the summarization models
│   ├── utils.py           # Utility functions for data handling and model operations
├── tests/                 # Unit tests for each component
├── README.md              # Project documentation 
└── requirements.txt       # List of required Python packages
```
## Installation 

### Step 1: Clone the Repository
 ```bash
git clone https://github.com/username/text-summarization.git
cd text-summarization
 ```

### Step 2: Create a virtual environment 
 ```bash
python3 -m venv env
source env/bin/activate  # On Windows, use `env\Scripts\activate`
 ```

### Step 3: Install Dependencies
 ```bash
pip install -r requirements.txt
 ```

### Step 4: Download Pretrained Models
 ```bash
python src/download_pretrained_models.py
 ```

## Models Used

1. Pretrained Encoders
This model architecture leverages pretrained encoders to understand contextual information within a text. These encoders are adept at grasping nuanced information, making them ideal for generating summaries that capture key details accurately.

2. Text-to-Text Transformer
The text-to-text transformer model enables straightforward text manipulation. For summarization, we fine-tune it on our specific dataset, optimizing it to focus on the most relevant details and output coherent summaries.


## Results 

## Contacts 
Authors : Lila Mekki, Théo Moret, Augustin Cablant
Emails : lila.mekki / theo.moret / augustin.cablant dot ensae.fr
