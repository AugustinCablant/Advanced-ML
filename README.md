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
├── data/                    # A folder containing the data we've used
├── figures/                 # The figures we've drawn
├── intermediate_notebooks/  # Intermediate notebook that we've created for the main notebook  
├── papers/                  # Main papers we've used
├── pictures/                # Some pictures for the Readme
├── src
   ├── Agents                # Multi-armed bandit agents
   ├── environments          # Multi-armed bandit environment
   ├── RealData              # Code for the Real-World Data Testing
   └── utils                 # Usefull functions for the repository
├── README.md                # Project documentation
└── requirements.txt         # Python dependencies

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
- **Goal:** Compare regret balancing performance against standard methods (e.g., ε-greedy, UCB).


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

## Results 

In this project, we explored the foundational theory of multi-armed bandits, starting with an introduction to standard environments and their associated algorithms. We implemented these environments and algorithms to illustrate their properties through numerical simulations. By highlighting the significant assumptions required by each algorithm to achieve optimal regret bounds, we identified the need for a more flexible approach.

To address this, we introduced the Regret Balancing method, which we validated through custom source code for numerical testing. Our results demonstrate its ability to achieve near-optimal regret while relying on fewer assumptions. Furthermore, we tested the method on real-world data, obtaining conclusive and promising outcomes. 

## References

Below are the references cited in this project:

1. Yasin Abbasi-Yadkori, Aldo Pacchiano, and My Phan. **Regret balancing for bandit and RL model selection.** *arXiv preprint arXiv:2006.09479*, 2020. *(The exact preprint URL or journal details need verification if available.)*

2. Aldo Pacchiano, Christoph Dann, Claudio Gentile, and Peter Bartlett. **Regret bound balancing and elimination for model selection in bandits and RL.** *arXiv preprint arXiv:2006.05491*, 2020.

3. Lihong Li, Wei Chu, John Langford, and Xuanhui Wang. **Unbiased offline evaluation of contextual-bandit-based news article recommendation algorithms.** In *Proceedings of the fourth ACM international conference on Web search and data mining, WSDM'11*, page 297–306. ACM, February 2011.


## License 
This project is licensed under the MIT License.

## Contacts 
Authors : Lila Mekki, Théo Moret, Augustin Cablant
Emails : lila.mekki / theo.moret / augustin.cablant dot ensae.fr
