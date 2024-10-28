# Advanced Machine Learning Project : Text Summarization with Pretrained Encoders

## Overview

This project aims to create concise summaries of text, emphasizing key information, using advanced machine learning techniques. Leveraging pretrained encoders and text-to-text transformer models, we focus on identifying and preserving essential details while reducing redundancy. This repository contains code, configurations, and documentation to run, experiment, and evaluate our text summarization pipeline effectively.

## Goals

Develop a robust text summarization pipeline to generate accurate and coherent summaries.
Highlight important elements within texts to improve the relevance of the generated summaries.
Compare and evaluate the performance of Pretrained Encoders and Text-to-Text Transformers for summarization tasks.

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


# Results 
