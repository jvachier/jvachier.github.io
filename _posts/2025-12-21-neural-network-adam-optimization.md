---
layout: default
title: Neural Network Math - Adam Optimization
date: 2025-12-21
excerpt: Comprehensive mathematical breakdown of neural network implementation with Adam optimization for digit classification.
---

# Mathematical Documentation: Neural Network with Adam Optimization

A comprehensive mathematical breakdown of all equations and operations implemented in a neural network for digit classification, featuring the Adam optimization algorithm.

## Overview

This post provides detailed mathematical formulations for:
- Activation functions (ReLU, Sigmoid, Softmax)
- Loss and cost functions
- Forward and backward propagation
- Adam optimization algorithm
- Performance metrics

**Network Architecture:**
- Input: 784 features (28×28 images)
- Hidden Layer 1: 128 neurons (ReLU)
- Hidden Layer 2: 40 neurons (ReLU)
- Output Layer: 10 neurons (Softmax)

---

## 1. Activation Functions

### ReLU (Rectified Linear Unit)

**Function:**
$$\text{ReLU}(x) = \max(0, x)$$

**Derivative:**
$$\frac{d\text{ReLU}(x)}{dx} = \begin{cases} 0 & \text{if } x \leq 0 \\ 1 & \text{if } x > 0 \end{cases}$$

**Purpose:** Introduces non-linearity while avoiding vanishing gradient problem.

### Softmax

**Function:**
$$\text{Softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}}$$

**Properties:**
- Output sums to 1: $\sum \text{Softmax}(x_i) = 1$
- Each output in range (0, 1)
- Converts logits to probability distribution

---

## 2. Loss and Cost Functions

### Mean Squared Error (MSE)

**Per-sample error:**
$$\text{MSE}_{\text{sample}} = (y_{\text{true}} - y_{\text{pred}})^2$$

**Total cost:**
$$J = \frac{1}{N} \sum_{i=1}^{N} (y_{\text{true}}^{(i)} - y_{\text{pred}}^{(i)})^2$$

---

## 3. Forward Propagation

### Layer 1 (Input → Hidden 1)

**Linear transformation:**
$$Z^{[1]} = W^{[1]} X + b^{[1]}$$

**Activation:**
$$A^{[1]} = \text{ReLU}(Z^{[1]})$$

**Dimensions:**
- $X$: (784 × m) where m is batch size
- $W^{[1]}$: (128 × 784)
- $b^{[1]}$: (128 × 1), broadcasted to (128 × m)

### Layer 2 (Hidden 1 → Hidden 2)

$$Z^{[2]} = W^{[2]} A^{[1]} + b^{[2]}$$
$$A^{[2]} = \text{ReLU}(Z^{[2]})$$

### Layer 3 (Hidden 2 → Output)

$$Z^{[3]} = W^{[3]} A^{[2]} + b^{[3]}$$
$$A^{[3]} = \text{Softmax}(Z^{[3]})$$

---

## 4. Backpropagation

### Output Layer Gradient

**Error at output:**
$$\delta^{[3]} = A^{[3]} - Y$$

**Weight gradient:**
$$\frac{\partial J}{\partial W^{[3]}} = \delta^{[3]} \cdot (A^{[2]})^T$$

**Bias gradient:**
$$\frac{\partial J}{\partial b^{[3]}} = \frac{1}{m}\sum_{i=1}^{m} \delta_i^{[3]}$$

### Hidden Layer Gradients

**Error propagation:**
$$\delta^{[2]} = (W^{[3]})^T \delta^{[3]} \odot \text{ReLU}'(Z^{[2]})$$

Where $\odot$ denotes element-wise multiplication (Hadamard product).

---

## 5. Adam Optimization Algorithm

Adam (Adaptive Moment Estimation) combines momentum and RMSprop for efficient optimization.

### Hyperparameters

$$\beta_1 = 0.9 \quad \text{(momentum decay rate)}$$
$$\beta_2 = 0.99 \quad \text{(RMSprop decay rate)}$$
$$\epsilon = 10^{-8} \quad \text{(numerical stability)}$$

### First Moment (Momentum)

$$m_t^{[l]} = \beta_1 \cdot m_{t-1}^{[l]} + (1 - \beta_1) \cdot \frac{\partial J}{\partial W^{[l]}}$$

**Interpretation:** Exponentially weighted average of past gradients (velocity).

### Second Moment (RMSprop)

$$v_t^{[l]} = \beta_2 \cdot v_{t-1}^{[l]} + (1 - \beta_2) \cdot \left(\frac{\partial J}{\partial W^{[l]}}\right)^2$$

**Interpretation:** Exponentially weighted average of squared gradients (acceleration).

### Bias Correction

**Corrected first moment:**
$$\hat{m}_t^{[l]} = \frac{m_t^{[l]}}{1 - \beta_1^t}$$

**Corrected second moment:**
$$\hat{v}_t^{[l]} = \frac{v_t^{[l]}}{1 - \beta_2^t}$$

### Parameter Update Rule

$$W_t^{[l]} = W_{t-1}^{[l]} - \alpha \cdot \frac{\hat{m}_t^{[l]}}{\sqrt{\hat{v}_t^{[l]}} + \epsilon}$$

---

## 6. Performance Metrics

### Categorical Accuracy

$$\text{Accuracy} = \frac{1}{m}\sum_{i=1}^{m} \mathbb{1}[\arg\max(y_{\text{true}}^{(i)}) = \arg\max(y_{\text{pred}}^{(i)})]$$

### R² Score (Coefficient of Determination)

$$R^2 = 1 - \frac{\sum_i (y_{\text{true}}^{(i)} - y_{\text{pred}}^{(i)})^2}{\sum_i (y_{\text{true}}^{(i)} - \bar{y}_{\text{true}})^2}$$

---

## Training Loop Summary

For each iteration t = 1, 2, ..., T:

1. **Forward Pass**: Compute $Z^{[l]}$ and $A^{[l]}$ for l = 1, 2, 3
2. **Compute Loss**: $\text{Loss} = \text{MSE}(A^{[3]}, Y)$
3. **Backward Pass**: Compute gradients $\partial J/\partial W^{[l]}$ and $\partial J/\partial b^{[l]}$
4. **Adam Update**:
   - Update first moments: $m_t = \beta_1 m_{t-1} + (1-\beta_1)\nabla J$
   - Update second moments: $v_t = \beta_2 v_{t-1} + (1-\beta_2)(\nabla J)^2$
   - Bias correction: $\hat{m}_t = m_t/(1-\beta_1^t)$, $\hat{v}_t = v_t/(1-\beta_2^t)$
   - Parameter update: $\theta_t = \theta_{t-1} - \alpha \cdot \hat{m}_t/(\sqrt{\hat{v}_t} + \epsilon)$
5. **Metrics**: Calculate accuracy, RMSE, R², and cost

---

## Mathematical Advantages of Adam

1. **Adaptive Learning Rates**: Each parameter has its own effective learning rate based on gradient history
2. **Momentum**: Helps escape local minima and accelerates convergence
3. **Bias Correction**: Ensures proper updates even in early training stages
4. **Robust to Hyperparameters**: Default values ($\beta_1=0.9$, $\beta_2=0.99$) work well for most problems
5. **Efficient**: Computationally similar to standard SGD with minimal memory overhead

---

## Implementation Notes

**Network Configuration:**

| Layer | Input Size | Output Size | Activation | Parameters |
|-------|------------|-------------|------------|------------|
| Input | 784 | 784 | - | 0 |
| Hidden 1 | 784 | 128 | ReLU | $W_1$(128×784), $b_1$(128×1) |
| Hidden 2 | 128 | 40 | ReLU | $W_2$(40×128), $b_2$(40×1) |
| Output | 40 | 10 | Softmax | $W_3$(10×40), $b_3$(10×1) |

**Total Parameters:** 128×784 + 128 + 40×128 + 40 + 10×40 + 10 = 105,898

---

## Reference

Kingma, D. P., & Ba, J. (2014). Adam: A Method for Stochastic Optimization. *arXiv preprint arXiv:1412.6980*.

---

**Tags:** #NeuralNetworks #DeepLearning #Optimization #Adam #Mathematics #MachineLearning
