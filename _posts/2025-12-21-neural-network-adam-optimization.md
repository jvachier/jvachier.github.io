---
layout: default
title: Building a Neural Network from Scratch - Understanding Adam Optimization
date: 2025-12-21
excerpt: A deep dive into the mathematics behind neural networks and the Adam optimizer, implemented from scratch in NumPy for digit classification.
---

# Building a Neural Network from Scratch: Understanding Adam Optimization

When I set out to build a digit classifier from scratch, I wanted to truly understand what happens under the hood of modern neural networks. This post is a deep dive into the mathematics behind a three-layer neural network using the Adam optimization algorithm, no black boxes, just pure NumPy and math.

**[→ View Full Implementation on GitHub](https://github.com/jvachier/Image_classification_neural_network_numpy-Adam-Optimization)**

## Why Build from Scratch?

Modern deep learning frameworks like TensorFlow and PyTorch are powerful, but they abstract away the underlying mathematics. Building a neural network from first principles helps us understand:

- How gradients actually flow backward through the network
- Why Adam optimizer is so effective compared to vanilla gradient descent
- What happens during each forward and backward pass
- How to debug training issues when they arise

## The Challenge: Digit Classification

**Task:** Classify handwritten digits (0-9) from 28×28 grayscale images

**Network Architecture:**
- **Input Layer:** 784 neurons (flattened 28×28 images)
- **Hidden Layer 1:** 128 neurons with ReLU activation
- **Hidden Layer 2:** 40 neurons with ReLU activation  
- **Output Layer:** 10 neurons with Softmax (one per digit class)

**Total Parameters:** 105,898 trainable weights and biases

---

## Part 1: The Building Blocks

### Activation Functions

Neural networks need non-linearity to learn complex patterns. I used two key activation functions:

**ReLU (Rectified Linear Unit)** for hidden layers:
$\text{ReLU}(x) = \max(0, x)\,.$

Why ReLU? It's simple, fast, and avoids the vanishing gradient problem that plagues sigmoid activations in deep networks. The derivative is equally simple:

$\frac{d\text{ReLU}(x)}{dx} = \begin{cases} 0 & \text{if } x \leq 0 \\ 1 & \text{if } x > 0 \end{cases}\,.$

**Softmax** for the output layer:
$\text{Softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{10} e^{x_j}}\,.$

Softmax converts raw scores into a probability distribution, perfect for multi-class classification. The outputs sum to 1, making them interpretable as class probabilities.

### Loss Function

I used **Mean Squared Error (MSE)** to measure prediction quality:

$J = \frac{1}{N} \sum_{i=1}^{N} (y_{\text{true}}^{(i)} - y_{\text{pred}}^{(i)})^2\,.$

While cross-entropy is more common for classification, MSE works well here and simplifies the math for backpropagation.

---

## Part 2: Forward Propagation

Forward propagation is the process of transforming input images into predictions. Each layer performs two operations: linear transformation and activation.

### Layer 1: Input → Hidden 1

$Z^{[1]} = W^{[1]} X + b^{[1]}\,,$
$A^{[1]} = \text{ReLU}(Z^{[1]})\,,$

where:
- $X$: Input batch of images (784 × m)
- $W^{[1]}$: Weight matrix (128 × 784)
- $b^{[1]}$: Bias vector (128 × 1)
- $A^{[1]}$: Activated output (128 × m)

### Layer 2: Hidden 1 → Hidden 2

$Z^{[2]} = W^{[2]} A^{[1]} + b^{[2]}\,,$
$A^{[2]} = \text{ReLU}(Z^{[2]})\,.$

### Layer 3: Hidden 2 → Output

$Z^{[3]} = W^{[3]} A^{[2]} + b^{[3]}\,,$
$A^{[3]} = \text{Softmax}(Z^{[3]})\,.$

The final output $A^{[3]}$ gives us 10 probabilities, one for each digit class.

---

## Part 3: Backpropagation - Where the Magic Happens

Backpropagation computes how much each weight contributed to the error, allowing us to update them in the right direction. This is where calculus and the chain rule shine.

### Output Layer Gradient

Starting from the output, the error signal is remarkably simple with MSE and Softmax:

$\delta^{[3]} = A^{[3]} - Y\,.$

This is just the difference between predictions and true labels! From here we can compute weight and bias gradients:

$\frac{\partial J}{\partial W^{[3]}} = \delta^{[3]} \cdot (A^{[2]})^T\,,$
$\frac{\partial J}{\partial b^{[3]}} = \frac{1}{m}\sum_{i=1}^{m} \delta_i^{[3]}\,.$

### Hidden Layer Gradients

For hidden layers, we propagate the error backward using the chain rule:

$\delta^{[2]} = (W^{[3]})^T \delta^{[3]} \odot \text{ReLU}'(Z^{[2]})\,.$

The $\odot$ symbol means element-wise multiplication (Hadamard product). The ReLU derivative acts as a gate—it only lets gradients flow through neurons that were active (positive) during forward propagation.

We repeat this process for all layers, computing gradients from output to input.

---

## Part 4: Adam Optimization - The Secret Sauce

Standard gradient descent updates weights by simply subtracting the gradient times a learning rate:

$W = W - \alpha \nabla J\,,$

but this is slow and can get stuck. Enter **Adam (Adaptive Moment Estimation)**, which combines the best ideas from momentum and RMSprop.

### Why Adam Works

Adam maintains two moving averages for each parameter:

1. **First moment (momentum):** Exponentially weighted average of past gradients
2. **Second moment (RMSprop):** Exponentially weighted average of squared past gradients

**Momentum equation:**
$m_t = \beta_1 m_{t-1} + (1 - \beta_1) \nabla J\,.$

Think of momentum as velocity, it helps the optimizer build up speed in directions with consistent gradients and dampens oscillations.

**RMSprop equation:**
$v_t = \beta_2 v_{t-1} + (1 - \beta_2) (\nabla J)^2\,.$

This tracks the "variance" of gradients. Parameters with large, noisy gradients get smaller effective learning rates.

### Bias Correction

Since we initialize $m_0 = 0$ and $v_0 = 0$, early estimates are biased toward zero. Adam corrects this:

$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}\,.$

### The Update Rule

Finally, we update parameters using both moments:

$W_t = W_{t-1} - \alpha \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}\,.$

Each parameter gets its own adaptive learning rate based on its gradient history. This makes Adam robust and fast.

**Hyperparameters I used:**
- $\beta_1 = 0.9$ (momentum decay)
- $\beta_2 = 0.99$ (RMSprop decay)
- $\epsilon = 10^{-8}$ (numerical stability)

---

## Part 5: Training Loop

Putting it all together, each training iteration follows this sequence:

1. **Forward Pass:** Compute activations $A^{[1]}, A^{[2]}, A^{[3]}$
2. **Compute Loss:** $J = \text{MSE}(A^{[3]}, Y)$
3. **Backward Pass:** Compute all gradients $\partial J/\partial W^{[l]}, \partial J/\partial b^{[l]}$
4. **Adam Update:**
   - Update momentum: $m_t = \beta_1 m_{t-1} + (1-\beta_1)\nabla J$
   - Update RMSprop: $v_t = \beta_2 v_{t-1} + (1-\beta_2)(\nabla J)^2$
   - Bias correction: $\hat{m}_t, \hat{v}_t$
   - Update parameters: $W \leftarrow W - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}$
5. **Track Metrics:** Accuracy, R², RMSE

---

## Results and Insights

Building this network from scratch revealed several key insights:

### Why Adam Outperforms SGD

1. **Adaptive learning rates:** Each parameter adjusts at its own pace
2. **Momentum:** Helps escape local minima and accelerates convergence
3. **Robustness:** Default hyperparameters ($\beta_1=0.9, \beta_2=0.99$) work remarkably well
4. **Efficiency:** Minimal computational overhead compared to vanilla SGD

### Network Design Decisions

- **Why 128 → 40 neurons?** This creates a bottleneck that forces the network to learn compressed representations
- **Why ReLU over sigmoid?** Faster computation and no vanishing gradients
- **Why MSE over cross-entropy?** Both work, but MSE simplifies derivative computation

### Performance Metrics

**Accuracy:** Measures the proportion of correct predictions
$\text{Accuracy} = \frac{1}{m}\sum_{i=1}^{m} \mathbb{1}[\arg\max(y_{\text{true}}^{(i)}) = \arg\max(y_{\text{pred}}^{(i)})]\,,$

where $\mathbb{1}[\cdot]$ is the indicator function that returns 1 if the condition is true, 0 otherwise. In other words: for each sample, we check if the predicted class (highest probability) matches the true class (position of 1 in one-hot encoding).

**R² Score:** Measures fit quality (1 = perfect, 0 = poor)
$R^2 = 1 - \frac{\sum_i (y_{\text{true}}^{(i)} - y_{\text{pred}}^{(i)})^2}{\sum_i (y_{\text{true}}^{(i)} - \bar{y}_{\text{true}})^2}\,.$

---

## Key Takeaways

1. **Understanding beats abstraction:** Building from scratch deepened my intuition about neural networks far more than using high-level APIs
2. **Adam is worth it:** The complexity of implementing Adam pays off in faster, more stable training
3. **Mathematics matters:** Every modern deep learning technique is rooted in elegant mathematical principles
4. **NumPy is powerful:** You don't need specialized frameworks for educational implementations

---

## What's Next?

Possible extensions to explore:
- Implement dropout regularization
- Try different network architectures (deeper or wider)
- Experiment with other optimizers (RMSprop, AdaGrad, Nadam)
- Add batch normalization
- Visualize learned features in hidden layers

---

## Implementation Details

**Network Configuration:**

| Layer | Input | Output | Activation | Parameters |
|-------|-------|--------|------------|------------|
| Input | 784 | 784 | - | 0 |
| Hidden 1 | 784 | 128 | ReLU | 100,480 |
| Hidden 2 | 128 | 40 | ReLU | 5,160 |
| Output | 40 | 10 | Softmax | 410 |

**[View the complete implementation on GitHub →](https://github.com/jvachier/Image_classification_neural_network_numpy-Adam-Optimization)**

---

## References

Kingma, D. P., & Ba, J. (2014). *Adam: A Method for Stochastic Optimization.* arXiv:1412.6980.

---

**Tags:** #DeepLearning #NeuralNetworks #Adam #Optimization #NumPy #FromScratch #Mathematics

*Have questions or suggestions? Feel free to reach out or open an issue on the GitHub repository!*

