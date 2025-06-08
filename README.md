# Threshold Dialectics: Simulation Code for "Adaptive Lever Management"

This repository contains the Python simulation code accompanying the book chapter, **"Chapter 9: Adaptive Lever Management: From Heuristic Limitations to Learned Resilience,"** from the forthcoming book, \textit{Threshold Dialectics: Understanding Complex Systems and Enabling Active Robustness}.

The simulations in this repository are designed to:
1.  Model a complex adaptive system governed by the principles of Threshold Dialectics (TD).
2.  Train a machine learning classifier to identify the phases of post-collapse recovery (the "Phoenix Loop").
3.  Implement and compare the efficacy of different intervention strategies, including no-intervention, naive heuristics, and Reinforcement Learning (RL) agents.
4.  Demonstrate the emergence of sophisticated adaptive behaviors, such as strategic inaction and the trade-offs between stability and recovery, in learned agents.

This README provides an overview of the conceptual framework, the repository structure, and instructions for running the simulations and reproducing the results from the chapter.

## Table of Contents
- [Conceptual Framework: Threshold Dialectics](#conceptual-framework-threshold-dialectics)
  - [The Three Adaptive Levers](#the-three-adaptive-levers)
  - [The Tolerance Sheet: A Dynamic Viability Boundary](#the-tolerance-sheet-a-dynamic-viability-boundary)
  - [The Phoenix Loop: Dynamics of Recovery](#the-phoenix-loop-dynamics-of-recovery)
  - [Key Diagnostics](#key-diagnostics)
- [Repository Structure](#repository-structure)
- [Setup and Installation](#setup-and-installation)
- [How to Run the Experiments](#how-to-run-the-experiments)
  - [Step 1: Train the ML Phase Classifier](#step-1-train-the-ml-phase-classifier)
  - [Step 2: Run the Main Intervention Study](#step-2-run-the-main-intervention-study)
  - [Step 3: (Optional) Run the Proactive Stabilization Study](#step-3-optional-run-the-proactive-stabilization-study)
- [Expected Outputs](#expected-outputs)
- [Citation](#citation)
- [License](#license)

## Conceptual Framework: Threshold Dialectics

The code in this repository is an implementation of the **Threshold Dialectics (TD)** framework. TD models complex adaptive systems as entities that manage their viability by dynamically balancing a set of core adaptive capacities ("levers") against systemic stress.

### The Three Adaptive Levers

The framework is built around three fundamental, interacting levers:

-   **Perception Gain ($g$):** The system's sensitivity to new information or prediction errors. High gain enhances vigilance but can amplify noise and is energetically costly.
-   **Policy Precision ($\beta$):** The system's confidence in its current strategy or model. High precision leads to efficient, exploitative behavior but can cause rigidity and a failure to adapt. It embodies the exploitation-exploration trade-off.
-   **Energetic Slack ($F_{\text{crit}}$):** The system's reserve capacity (e.g., energy, capital, time) available to absorb shocks, fuel adaptation, or sustain operations during stress.

### The Tolerance Sheet: A Dynamic Viability Boundary

The combined state of the three levers defines the system's **Tolerance Sheet ($\Theta_{\text{T}}$)**, its maximum capacity to withstand systemic strain. Collapse occurs when the system's strain exceeds this dynamic boundary.

$\Theta_{\text{T}} = C \cdot g^{w_1} \beta^{w_2} F_{\text{crit}}^{w_3}$

The exponents ($w_1, w_2, w_3$) represent the system's evolved or designed reliance on each lever for its resilience.

### The Phoenix Loop: Dynamics of Recovery

After a collapse, systems often traverse a four-phase recovery cycle known as the **Phoenix Loop**:

1.  **Phase I: Disintegration:** The initial breakdown and chaotic unraveling post-collapse.
2.  **Phase II: Flaring:** A period of high uncertainty and broad exploration of new strategies and structures.
3.  **Phase III: Pruning:** Consolidation and selection, where successful innovations are reinforced and failures are abandoned.
4.  **Phase IV: Restabilization:** A new, stable operating regime is established.

### Key Diagnostics

TD uses several key diagnostics to monitor the system's dynamic state:

-   **Speed Index ($\mathcal{S}$):** The joint rate of change of the core levers ($\beta$ and $F_{\text{crit}}$). A high Speed Index indicates rapid, potentially destabilizing, structural drift.
-   **Couple Index ($\mathcal{C}$):** The correlation between the lever velocities ($\dot{\beta}$ and $\dot{F}_{\text{crit}}$). Detrimental coupling (e.g., rigidity rising while slack depletes) is a strong sign of increasing fragility.
-   **Exploration Entropy Excess ($\rho_E$):** A measure of the system's exploratory behavior, used to identify the "Flaring" phase of the Phoenix Loop.

## Repository Structure

This repository contains two primary Python scripts and one main study script:

-   "phoenix_loop_classifier_accuracy_ML.py":
    -   Defines the core "TDSystem" simulation and "DiagnosticsCalculator".
    -   Contains the logic to generate simulation data of the Phoenix Loop recovery cycle.
    -   Trains a "RandomForestClassifier" to identify the four phases of the Phoenix Loop based on TD diagnostics.
    -   Saves the trained model ("phoenix_rf_classifier.joblib") and a feature scaler ("phoenix_feature_scaler.joblib") to the "results/" directory.

-   "phoenix_loop_intervention_RL.py":
    -   Imports the "TDSystem" and diagnostics from the first script.
    -   Defines the "PhoenixLoopRLEnv" and "ProactiveStabilizationRLEnv" classes for training Reinforcement Learning agents.
    -   Implements various intervention policies: no-intervention, naive heuristics, and the RL agent.
    -   The "if __name__ == "__main__"" block runs the main comparative study, training/loading a PPO agent, evaluating all policies, and generating analysis plots and summaries.

-   "proactive_stabilization_study.py":
    -   An executable script that uses components from "phoenix_loop_intervention_RL.py".
    -   Trains and evaluates a specialized "proactive" RL agent designed to maintain high stability, using the "ProactiveStabilizationRLEnv".
    -   Allows for focused experimentation on different learning objectives for RL agents.

-   "results/": Directory where all simulation outputs, including plots, summary files, and ML models, are saved.
-   "rl_models/": Directory where trained Reinforcement Learning (PPO) models are saved.

## Setup and Installation

To run the simulations, you need Python 3.8+ and the following packages. It is highly recommended to use a virtual environment.

1.  **Clone the repository:**
    """bash
    git clone https://github.com/your-username/threshold-dialectics.git
    cd threshold-dialectics
    """

2.  **Create and activate a virtual environment:**
    """bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use "venv\Scripts\activate"
    """

3.  **Install the required packages:**
    A "requirements.txt" file can be created with the following content.
    """
    numpy
    scipy
    matplotlib
    scikit-learn
    pandas
    joblib
    stable-baselines3[extra]
    gymnasium
    """
    Install them using pip:
    """bash
    pip install -r requirements.txt
    """

## How to Run the Experiments

The experiments are designed to be run in a specific order, as the intervention study depends on the output of the ML classifier.

### Step 1: Train the ML Phase Classifier

First, you must train the Random Forest model that identifies the Phoenix Loop phases. This model is used by some of the heuristic intervention agents.

"""bash
python phoenix_loop_classifier_accuracy_ML.py
"""

This script will:
-   Run 50 simulations of the "TDSystem" to generate training data.
-   Train a "RandomForestClassifier" and save "phoenix_rf_classifier.joblib" to the "results/" directory.
-   Save a feature scaler "phoenix_feature_scaler.joblib" to the "results/" directory.
-   Display a confusion matrix and classification report for the trained model.

### Step 2: Run the Main Intervention Study

Once the ML classifier is trained, you can run the main comparative study from the book chapter. This script will train (or load) a PPO Reinforcement Learning agent and compare its performance against the "No Intervention," "Naive Fcrit Boost," and "TD-Informed" policies.

"""bash
python phoenix_loop_intervention_RL.py
"""

This script will:
-   Check for a pre-trained RL model in "rl_models/". If not found (or if "FORCE_RETRAIN" is "True"), it will train a new PPO agent and save it.
-   Run evaluation simulations for each of the four intervention conditions.
-   Perform a detailed analysis of the results, printing summaries to the console.
-   Generate and save summary files (".txt", ".json") and several plots (".png") to the "results/" directory.

### Step 3: (Optional) Run the Proactive Stabilization Study

This script runs a separate, focused experiment on an RL agent trained with a reward function that heavily incentivizes proactive stability.

"""bash
python proactive_stabilization_study.py --env_level challenging --timesteps 500000
"""

You can customize the environment difficulty ("--env_level") and the number of training timesteps ("--timesteps"). This will train/load a specialized RL model and evaluate its performance, saving relevant plots and summaries.

## Expected Outputs

All outputs are saved to the "results/" directory, organized by the environment level (e.g., "_easy_debug", "_challenging"). Key outputs include:

-   **ML Models:** "phoenix_rf_classifier.joblib", "phoenix_feature_scaler.joblib"
-   **RL Models:** "rl_models/phoenix_loop_rl_ppo_agent_challenging.zip" (example name)
-   **Summary Files:**
    -   "simulation_summary_[level].txt": Detailed text summary of the comparative study.
    -   "summary_intervention_RL_[level].json": A compact JSON summary suitable for programmatic analysis.
-   **Plots (".png"):**
    -   "rl_agent_strategy_example_[level].png": A detailed plot showing the RL agent's moment-to-moment decisions.
    -   "fcrit_dynamics_average_trajectories_incl_rl_[level].png": A plot comparing the average "F_crit" levels for each intervention policy.
    -   "phoenix_loop_navigation_[level].png": A 3D plot of the system's trajectory in the diagnostic space ("Speed", "Couple", "rhoE").
    -   Confusion matrices for the ML classifier.

## Citation

If you use or refer to this code or the concepts from Threshold Dialectics, please cite the accompanying book:

@book{pond2025threshold,
  author    = {Axel Pond},
  title     = {Threshold Dialectics: Understanding Complex Systems and Enabling Active Robustness},
  year      = {2025},
  isbn      = {978-82-693862-2-6},
  publisher = {Amazon Kindle Direct Publishing},
  url       = {https://www.thresholddialectics.com},
  note      = {Code repository: \url{https://github.com/threshold-dialectics}}
}

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.