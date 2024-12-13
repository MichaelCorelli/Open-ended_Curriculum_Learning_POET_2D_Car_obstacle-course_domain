# Open-ended Curriculum Learning with POET in a 2D Car Obstacle-Course Domain

## **Overview**
This project implements a reinforcement learning framework based on the **Paired Open-Ended Trailblazer (POET)** algorithm. The goal is to train a 2D car agent to navigate dynamically evolving obstacle courses while simultaneously evolving the environments for progressive learning.

The environment is built using **Box2D** and **Pygame**, and the agent is trained using **Deep Q-Learning (Double DQN)**. The POET algorithm is adapted to handle 2D car-specific challenges, such as ramps, bumps, and holes.

This project was developed for the **Reinforcement Learning course A.Y. 2024/2025**.

---

## **Key Features**

### 1. **Agent**
- **Vehicle Control**:
  - Actions: 9 discrete combinations of steering and acceleration.
  - Observations: LiDAR-based sensor data and car's position/velocity.

### 2. **Environment**
- **Obstacle types**: ramps, holes, and bumps.
- **Mutations**: environments evolve dynamically with increasing difficulty.
- Rendered using Pygame with customizable textures.

### 3. **POET Algorithm**
- **Co-evolution**: simultaneous evolution of agent policies and environment difficulties.
- **Novelty ranking**: promotes diversity by evaluating the novelty of new environments.
- **Minimal Criterion (MC)**: ensures environments meet specific performance thresholds before being added.
- **Policy Transfer**: Transfers learned policies between environments to tackle harder challenges.

### 4. **Double DQN**
- Neural network-based policy optimization with:
  - Batch normalization and dropout for stable training.
  - Experience replay for sample efficiency.

---

## **File Structure**

```
Open-ended_Curriculum_Learning_POET_2D_Car_obstacle-course_domain/
├── main.py
├── env.py
├── model.py
├── poet.py
├── utils.py
├── textures/
│   ├── background.png
│   └── wheel.png
├── requirements.txt
├── README.md
└── LICENSE
```

### **`env.py`**
- Defines the **`CarEnvironment`** class:
  - Implements the simulation logic using Box2D.
  - Handles obstacles, rendering, and reward calculation.
- Includes car dynamics and LiDAR data collection.

### **`poet.py`**
- Implements the **POET** algorithm:
  - `main_loop`: Main POET training and evolution loop.
  - `mutate_envs`: Mutates environments and evaluates offspring.
  - `evaluate_candidates`: Updates policies using Evolution Strategies (ES).

### **`model.py`**
- Defines the **Double DQN (DDQN)** agent:
  - Neural network for Q-value estimation.
  - Training logic, experience replay buffer, and evaluation methods.

### **`utils.py`**
- Utility functions for:
  - State dict to vector conversion and vice versa.
  - Neural network parameter manipulation.

### **`main.py`**
- Entry point for running the project:
  - Initializes the environment and agent.
  - Executes the POET main loop.
  - Visualizes and evaluates trained agents and environments.

---

## **Setup and Installation**

### **Requirements**
- Python 3.9.6
- Dependencies:
  - `torch`
  - `numpy`
  - `matplotlib`
  - `pygame`
  - `gymnasium`
  - `Box2D`
  - `scipy`
  - `scikit-learn`
  - `tqdm`

### **Installation**
1. Clone this repository:
   ```bash
   git clone https://github.com/MichaelCorelli/Open-ended_Curriculum_Learning_POET_2D_Car_obstacle-course_domain
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Set up a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

---

## **How to Run**

### **1. Training the Agent**
Run the main script to execute the POET algorithm and train the agent:
```bash
python main.py
```

- The program will:
  1. Initialize a simple 2D car environment.
  2. Train the agent using Double DQN.
  3. Mutate and evaluate new environments.

### **2. Visualizing the Trained Agent**
After training, you can visualize the agent's performance in various environments:
- Use the arrow keys to navigate between environments.
- Press **Right Arrow (→)** to view the next environment.
- Press **Left Arrow (←)** to view the previous environment.

### **3. Profiling (Optional)**
To profile the execution:
```bash
python main.py --profile
```

---

## **Project Workflow**

1. **Initialize Environment**: Create a basic 2D car simulation.
2. **Agent Training**: Train the car agent using Double DQN on the initial environment.
3. **Environment Mutation**: Gradually evolve environments by adding/changing obstacles.
4. **Policy Transfer**: Transfer trained policies between environments to tackle harder challenges.
5. **Evaluation**: Assess agent performance and log metrics.

---

## **Key Metrics**
The project evaluates the agent's performance using the following metrics:

- **Mean Reward per Environment**: average cumulative reward the agent receives across episodes.
- **Standard Deviation of Rewards**: measures the variability in rewards, indicating consistency.
- **Minimum and Maximum Rewards**: tracks the best and worst performance across episodes.
- **Mean and Standard Deviation of Steps per Episode**: indicates how efficiently the agent completes tasks.
- **Mean, Minimum, and Maximum Final Positions**: assesses how far the agent travels in the environment.

---

## References

1. **R. Wang, J. Lehman, J. Clune, and K. O. Stanley**  
   *"Paired Open-Ended Trailblazer (POET): Endlessly Generating Increasingly Complex and Diverse Learning Environments and Their Solutions,"*  
   arXiv preprint arXiv:1901.01753v3, 2019.  
   [https://arxiv.org/abs/1901.01753](https://arxiv.org/abs/1901.01753)

2. **Roberto Capobianco**  
   *More off-policy & DQN.pdf,* Lecture Notes, 2024.

3. **KRL Group**  
   GitHub Repository for Reinforcement Learning Course, 2024.  
   [https://github.com/KRLGroup/RL_2024.git](https://github.com/KRLGroup/RL_2024.git)

## **Contacts**
For questions or contributions, please reach out to:
- Name: Gianmarco Donnesi
- Email: donnesi.2152311@studenti.uniroma1.it

- Name: Michael Corelli
- Email: corelli.1938627@studenti.uniroma1.it

## **License**
This project is licensed under the [GPL-3.0 License](LICENSE). See the file for more details.