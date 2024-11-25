<p align="center">
 <img src="assets/logo.svg" alt="OmniAI Logo" width="400"/>
</p>

# IMPORTANT❗
> This project is under maintenance and as such, some files may not run. This will be fixed in due course.

OmniAI provides implementations of foundational AI algorithms from first principles. Built for learning, understanding, and mastering the core concepts of artificial intelligence and machine learning.

## 🌟 Architecture Overview

This repository implements AI algorithms from ground up, focusing on both theoretical foundations and practical implementations. The codebase follows familiar scikit-learn and PyTorch patterns while exposing the underlying mechanics.

## 🗺️ AI Taxonomy

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
classDef default fill:#f9f9f9,stroke:#333,stroke-width:2px
classDef aiNode fill:#e3f2fd,stroke:#1565c0,stroke-width:4px
classDef mlNode fill:#e8f5e9,stroke:#2e7d32,stroke-width:3px
classDef descNode fill:#f5f5f5,stroke:#666,stroke-width:1px

class AI aiNode
class ML,DL,RL mlNode
class ES_DESC,SEARCH_DESC,SL_DESC,UL_DESC,SSL_DESC,DL_DESC,RL_DESC descNode
```

## 🎯 Core Objectives

### Educational Foundation
- Master mathematical foundations through probabilistic and calculus-based approaches
- Understand gradient descent mechanics beyond `.fit()`
- Implement backpropagation from first principles
- Explore probabilistic models (Naive Bayes, GMMs) at a fundamental level

### Technical Mastery
- Debug complex models through deep understanding of internals
- Optimize implementations for specific use cases
- Identify and resolve performance bottlenecks
- Make informed architectural decisions

### Engineering Excellence
- Understand core algorithms at implementation level
- Master mathematical foundations behind each approach
- Learn model limitations and assumptions
- Develop robust debugging strategies

## 🛠️ Implementation Philosophy

While powerful libraries like scikit-learn, PyTorch, and TensorFlow exist, implementing from scratch provides:
- Deep understanding of hyperparameter impacts
- Intuition for algorithm selection
- Recognition of model constraints
- Clear architectural decision-making capability

Most of the implementations will take a blend of scikit-learn and PyTorch form.
## 🧮 Mathematical Approaches

### Probabilistic Methods
- Foundation: Bayes' Theorem and probability theory
- Implementations:
 - Naive Bayes using P(y|X)
 - Gaussian Mixture Models
 - Hidden Markov Models

### Optimisation Methods
- Foundation: Calculus and gradient descent
- Implementations:
 - Linear Regression with MSE optimisation
 - Neural Networks with backpropagation
 - Support Vector Machines with margin maximisation

We will also aim to demonstrate the connection between these two approaches in order to build
a unified interface.
## 🚀 Getting Started

```bash
TBC...
```

## 📖 Documentation

Detailed documentation available in the `/docs` directory:
- Algorithm implementations
- Mathematical derivations
- Usage examples
- Performance benchmarks

## Authors 📚
- Siddartha Nath 
