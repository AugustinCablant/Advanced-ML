# Model Selection in Bandit Algorithms

## Overview
This repository explores model selection in the context of bandit algorithms, focusing on adapting dynamically to the best learning algorithm among a set of candidates. Our goal is to minimize cumulative regret by balancing the exploration and exploitation trade-off, leveraging strategies like regret balancing.

## Motivation 
Bandit algorithms often rely on assumptions about the reward structure. Selecting the wrong model can lead to suboptimal decisions and high regret. For example:
- LinUCB assumes linear reward mappings, which may not hold in all tasks.
- UCB does not assume linearity but may underperform when a linear structure exists.
Through regret balancing, we dynamically evaluate and select the best base algorithm at each step, ensuring regret remains close to optimal, even when the true reward structure is unknown.

## Applications
1. **Bandit Model Selection:**  
   - Example: Deciding whether to use **UCB** (agnostic to reward structure) or **LinUCB** (assumes linear rewards) for a task.  
   - Selecting the wrong algorithm can lead to significant cumulative regret due to mismatched assumptions.  

<picture>
        <source media="(prefers-color-scheme: dark) srcset= "https://github.com/AugustinCablant/Advanced-ML/blob/main/pictures/UCB_RB.png">
        <source media="(prefers-color-scheme: light)" srcset="https://github.com/AugustinCablant/Advanced-ML/blob/main/pictures/UCB_RB.png">
        <img alt="UCB vs linUCB vs RB" src="https://github.com/AugustinCablant/Advanced-ML/blob/main/pictures/UCB_RB.png">
</picture>

2. **Representation Learning for Contextual Bandits:**  
   - Example: Given multiple candidate **linear mappings**, determine which best represents the reward model of a contextual bandit.  
   - The regret balancing strategy dynamically identifies the most suitable mapping, avoiding overcommitment to incorrect models.

## Objectives

The project focuses on:
1. **Theory and Implementation:**
   - Implement classic algorithms: **ε-greedy**, **UCB**, **LinUCB**.
   - Explore the **Regret Balancing Strategy** for model selection.

2. **Use Cases:**
   - **Optimizing Exploration Rate for ε-greedy:**  
     Use regret balancing to determine the best exploration rate.
   - **Representation Learning with Regret Balancing:**  
     Dynamically select the best state-action mapping for contextual bandits using LinUCB variants.

3. **Real-World Testing:**  
   Evaluate the regret balancing strategy on the [Open Bandit Dataset](https://github.com/st-tech/zr-obp), using pre-implemented bandit algorithms provided by ZOZO, Inc.


<picture>
        <source media="(prefers-color-scheme: dark) srcset= "https://github.com/AugustinCablant/Advanced-ML/blob/main/pictures/RealData.png">
        <source media="(prefers-color-scheme: light)" srcset="https://github.com/AugustinCablant/Advanced-ML/blob/main/pictures/RealData.png">
        <img alt="UCB vs linUCB vs RB" src="https://github.com/AugustinCablant/Advanced-ML/blob/main/pictures/RealData.png">
</picture>

## Project Structure
```plaintext
.
├── algorithms/           # Implementation of bandit algorithms
├── experiments/          # Scripts for running experiments
├── data/                 # Dataset and preprocessing scripts
├── config/               # Configuration files for experiments
├── results/              # Output logs and performance metrics
├── README.md             # Project documentation
└── requirements.txt      # Python dependencies

```

## Implementation Details

### 1. Bandit Algorithms
- **Algorithms Implemented:**
  - **ε-greedy:** Balances exploration and exploitation using a tunable exploration rate.
  - **UCB:** Uses confidence intervals to guide exploration.
  - **LinUCB:** Assumes linear relationships between contexts, actions, and rewards.
- **Experiments:**
  - Run numerical experiments in various environments (i.i.d. rewards, fixed actions).
  - Verify theoretical regret bounds for each algorithm.

### 2. Regret Balancing Strategy
- **Settings:**
  - **Known Optimal Regret:** Implement regret balancing when the optimal regret is known.
  - **Candidate Regret Bounds:** Extend the strategy to iteratively verify candidate regret bounds.
- **Experiments:**
  - Illustrate regret balancing as a standalone bandit algorithm.
  - Evaluate performance in optimizing exploration rate and representation learning.

### 3. Use Case 1: Exploration Rate for ε-greedy
- **Objective:**
  - Theoretically explore and experimentally validate the optimal exploration rate in various environments (e.g., i.i.d. actions, fixed action sets).

### 4. Use Case 2: Representation Learning
- **Objective:**
  - Use regret balancing to identify the best linear mapping for contextual bandits.
  - Extend experiments to include non-linear mappings and compare against classical LinUCB.

### 5. Real-World Data Testing
- **Dataset:** [Open Bandit Dataset](https://github.com/st-tech/zr-obp)
- **Goal:** Compare regret balancing performance against standard methods (e.g., ε-greedy, LinUCB, Lin Thompson Sampling).


## Installation 

### Step 1: Clone the Repository
 ```bash
git clone 
cd 
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



## Results 

## License 
This project is licensed under the MIT License.

## Contacts 
Authors : Lila Mekki, Théo Moret, Augustin Cablant
Emails : lila.mekki / theo.moret / augustin.cablant dot ensae.fr
