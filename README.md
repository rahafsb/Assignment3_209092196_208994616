# README
# Assignment 3 submitted by Rahaf Sbeh and Saar Buium 

## Extending a Multi-layer Artificial Neural Network (MLP)

This repository contains the modified implementation of a Multi-layer Perceptron (MLP) from Chapter 11 of the book "Machine Learning with PyTorch and Scikit-Learn" by Raschka et al. (2022). The modifications include extending the original MLP code to support **two hidden layers** instead of a single hidden layer.

### Links to Source Code:

1. **Original Code from Chapter 11:**  
   [GitHub Link to Original Code](https://github.com/rasbt/machine-learning-book/blob/main/ch11/ch11.ipynb)

2. **Modified Code with Two Hidden Layers:**  
   [GitHub Link to Modified Code](https://github.com/rahafsb/Assignment3_209092196_208994616/blob/main/MLP.ipynb)

---

### Tasks Implemented:

1. **Extending the MLP Architecture:**  
   The original MLP code was revised to include two hidden layers with configurable neurons and activation functions.

2. **MNIST Classification:**  
   The revised model was applied to the MNIST dataset, using the full architecture described in class ("Solution 1: A plain deep NN"). A **Train(70%)/Test(30%) validation procedure** was employed, and the **macro AUC score** was calculated to evaluate performance.

3. **Performance Comparison:**  
   The revised MLP was compared to:
   - The original single hidden layer implementation  by Raschka.
   - A fully connected ANN implemented using PyTorch.

---

### How to Run the Code:

1. Clone this repository:
   ```bash
   git clone <ADD_YOUR_MODIFIED_REPO_LINK_HERE>
   cd <repository-folder>
   ```

2. Install required libraries:
   ```bash
   pip install numpy matplotlib scikit-learn torch torchvision
   ```

3. Open and run the modified Jupyter notebook:
   ```bash
   jupyter notebook MLP.ipynb
   ```

4. Follow the instructions in the notebook to reproduce the results.

---

### Results Overview:

The results of the predictive performance comparison are detailed in the accompanying report (`report.pdf`). Key metrics like **macro AUC** demonstrate the impact of using two hidden layers compared to one. Additionally, insights into the pyTorch implementation are also provided.

For further details, please refer to the `report.pdf` included in this repository and zip file.

