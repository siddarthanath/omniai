<p align="center">
 <img src="assets/logo.svg" alt="OmniAI Logo" width="400"/>
</p>

> [!IMPORTANT]  
> This project is currently in a very early development and experimental stage.

OmniAI provides implementations of foundational AI algorithms from first principles. Built for learning, understanding, and mastering the core concepts of artificial intelligence and machine learning.

## Architecture Overview 🌟

This repository implements AI algorithms from ground up, focusing on both theoretical foundations and practical implementations. The codebase follows familiar scikit-learn and PyTorch patterns while exposing the underlying mechanics.

## AI Taxonomy 🗺️

```mermaid
graph LR
%% Main AI hierarchy
AI[Artificial Intelligence] --> TRAD[Traditional AI]
AI --> ML[Machine Learning]

%% Traditional AI methods
TRAD --> ES[Expert Systems]
ES --> ES_DESC[["Input: Rules + Facts<br>Output: Decisions<br>Goal: Apply expert knowledge"]]

TRAD --> SEARCH[Search & Planning]
SEARCH --> SEARCH_DESC[["Input: Problem space + Goals<br>Output: Action sequence<br>Goal: Find optimal solution"]]

%% Machine Learning branches
ML --> SL[Supervised Learning]
SL --> SL_DESC[["Input: Labeled data (X,y)<br>Output: Predictions<br>Goal: Learn input→output mapping"]]

ML --> UL[Unsupervised Learning]
UL --> UL_DESC[["Input: Unlabeled data<br>Output: Patterns/Clusters<br>Goal: Find hidden structure"]]

ML --> SSL[Self-Supervised]
SSL --> SSL_DESC[["Input: Unlabeled + auto-labels<br>Output: Representations<br>Goal: Learn without human labels"]]

%% DL and RL relationships
ML <--> DL[Deep Learning]
DL --> DL_DESC[["Input: Raw data<br>Output: Learned features<br>Goal: Learn hierarchical representations"]]

ML <--> RL[Reinforcement Learning]
RL --> RL_DESC[["Input: Environment states<br>Output: Actions/Policy<br>Goal: Learn optimal behavior"]]

DL <--> RL

%% Styling
classDef default fill:#f9f9f9,stroke:#333,stroke-width:2px,color:#000
classDef aiNode fill:#e3f2fd,stroke:#1565c0,stroke-width:4px,color:#000
classDef mlNode fill:#e8f5e9,stroke:#2e7d32,stroke-width:3px,color:#000
classDef descNode fill:#f5f5f5,stroke:#666,stroke-width:1px,color:#000

class AI aiNode
class ML,DL,RL mlNode
class ES_DESC,SEARCH_DESC,SL_DESC,UL_DESC,SSL_DESC,DL_DESC,RL_DESC descNode
```

## Core Objectives 🎯

### Educational Foundation
- Master mathematical foundations through probabilistic and calculus-based approaches
- Understand gradient descent mechanics beyond `.fit()`
- Implement backpropagation from first principles
- Explore probabilistic models (Naive Bayes, GMMs) at a fundamental level

### Technical Mastery
- Debug complex models through deep understanding of internals
- Optimise implementations for specific use cases
- Identify and resolve performance bottlenecks
- Make informed architectural decisions

### Engineering Excellence
- Understand core algorithms at implementation level
- Master mathematical foundations behind each approach
- Learn model limitations and assumptions
- Develop robust debugging strategies

## Implementation Philosophy 🛠️

While powerful libraries like scikit-learn, PyTorch, and TensorFlow exist, implementing from scratch provides:
- Deep understanding of hyperparameter impacts
- Intuition for algorithm selection
- Recognition of model constraints
- Clear architectural decision-making capability

Most of the implementations will take a blend of scikit-learn and PyTorch form.

## Mathematical Approaches 🧮 
We aim to provide both analytical/statistical and optimisation solutions for each model. 

### Discriminative Methods
- Foundation: Modelling $P\left(\underline{\textbf{y}}|\underline{\textbf{x}}\right)$, where $\underline{\textbf{x}}$ follows a specified data distribution.

   Implementations:
    - Linear Regression
    - Gaussian Mixture Models
    - Hidden Markov Models

### Generative Methods
- Foundation: Modelling $P\left(\underline{\textbf{x}}, \underline{\textbf{y}}\right)=P\left(\underline{\textbf{x}}\right)P\left(\underline{\textbf{x}}|	\underline{\textbf{y}}\right)$.

   Implementations:
    - AutoEncoder
    - GANs
    - Transformers

### Optimisation Methods
- Foundation: Modelling minimisation through analytical and heuristic optimisation algorithms.
   
   Implementations:
    - Linear Regression with MSE optimisation (equivalent to NLL optimisation)
    - Neural Networks with backpropagation
    - Support Vector Machines with margin maximisation

## Getting Started 🚀 
Using this library could never be easier:
1. Instantiate a model.
2. Instantiate a trainer, with a compatible integration of optimiser and its' config.
3. Instantiate a suitable loss function.
4. Start training!

```python
# Instantiate model
model = LinearRegression(input_dim=1)
# Instantiate config for optimiser
optimiser_config = GradientDescentConfig(lr=0.1)
# Instantiate optimiser
optimiser = GradientOptimiser(model.parameters(), config=optimiser_config)
# Instantiate config for trainer
trainer_config = GradientTrainerConfig(batch_size=32, epochs=100)
# Instantiate trainer
trainer = GradientTrainer(config=trainer_config, optimiser=optimiser)
# Instantiate loss function
loss_fn = MSELoss()
# Train model!
trainer.train(model=model, X=X, y=y, loss_fn=loss_fn)
```
Note that some models have an analytical solution to find their optimal parameters - in this case, training is not necessary.
## Documentation 📖

Detailed documentation available in the `/docs` directory:
- Library architecture
- Algorithm implementations
- Mathematical derivations
- Usage examples
- Performance benchmarks

## Authors 📚
- Siddartha Nath 
