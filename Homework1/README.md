# Homework 1 | Deep Learning 2025-26

This section of the repository contains both the implementation and the generated results for Homework 1 of the Deep Learning course. 

The homework consists of three questions. Only Questions 1 and 2 involve programming. Question 3 is theoretical and solved separately on paper. All code related to this homework, along with the plots, saved models, and result files produced during experimentation, is included in this directory.

All source code files for Homework 1 are stored in `homework1_code/`.

The organization of the implementation follows the structure of the assignment and is detailed below.

## Question 1 - Linear Models and Basic Neural Networks
### 1. Perceptron Classifier
**File:** `homework1_code/hw1-perceptron.py`  
Implements a multi-class perceptron and evaluates it on the EMNIST Letters dataset.

To carry out Part (a), proceed with the perceptron execution as follows:

```bash
python3 homework1_code/hw1-perceptron.py \
  --epochs 20 \
  --data-path emnist-letters.npz \
  --save-path perceptron-best.pkl \
  --accuracy-plot perceptron-accs.pdf \
  --scores perceptron-scores.json
```

### 2. Logistic Regression
**File:** `homework1_code/hw1-logistic-regression.py`  
Implements a logistic regression classifier, including training with stochastic gradient descent, evaluation, and experiments involving feature engineering (such as HOG features) and hyperparameter tuning.

For Part (a), the logistic regression model can be executed using the following command:

```bash
python3 homework1_code/hw1-logistic-regression.py \
  --mode single \
  --epochs 20 \
  --data-path emnist-letters.npz \
  --save-path Q1-logistic-regression-best.pkl \
  --accuracy-plot Q1-logistic-regression-accs.pdf \
  --scores Q1-logistic-regression-scores.json
```

For Part (c), the grid-search experiment should be executed with:

```bash
python3 homework1_code/hw1-logistic-regression.py \
  --mode grid \
  --epochs 20 \
  --data-path emnist-letters.npz \
  --save-path Q1-logistic-grid-search-best.pkl \
  --scores Q1-logistic-grid-search-results.json
```

### 3. Multi-Layer Perceptron
**File:** `homework1_code/hw1-multilayer-perceptron.py`  
Implements a simple neural network with one hidden layer, trained using manual backpropagation.

For Part (a), the multi-layer perceptron can be executed using the following command:

```bash
python3 homework1_code/hw1-multilayer-perceptron.py \
  --epochs 20 \
  --data-path emnist-letters.npz \
  --save-path mlperceptron-best.pkl \
  --accuracy-plot mlperceptron-accs.pdf \
  --loss-plot mlperceptron-loss.pdf \
  --scores logistic-scores.json
```

## Question 2 - Feedforward Neural Networks (FFNs)
All programming components of Question 2 are implemented in:  
**File:** `homework1_code/hw1-ffn.py`  

TODO: completar com cada alínea

## Question 3
TODO: deixar nota a indicar em que ficheiro estão as demonstrações? ou não