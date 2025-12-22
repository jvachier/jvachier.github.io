---
layout: default
title: Linear Regression - Analytic and Numerical Methods vs Sklearn
date: 2025-12-22
excerpt: A comprehensive comparison of three approaches to linear regression - implementing the analytic solution using least squares, numerical optimization with gradient descent, and scikit-learn's optimized library.
---

# Linear Regression: Analytic and Numerical Methods vs Sklearn

When I set out to truly understand linear regression, I didn't just want to call `sklearn.fit()` and move on. I wanted to implement it from scratch using both the closed-form mathematical solution and iterative numerical optimization. This post compares three approaches to salary prediction: an analytic solution using the least squares method, a numerical solution using gradient descent, and the scikit-learn implementation.

**[→ View Full Implementation on Kaggle](https://www.kaggle.com/code/jvachier/lr-analytic-numerical-methods-vs-sklearn)**

## The Problem: Predicting Salary from Experience

**Task:** Predict salary as a function of years of experience using univariate linear regression

**Datasets tested:**
1. Real-world salary data
2. Synthetic linear data
3. Synthetic linear data with offset

This project demonstrates the mathematical foundations of linear regression by implementing both analytic and numerical methods from scratch, and validates these implementations against scikit-learn's optimized library.

---

## Part 1: The Analytic Method - Ordinary Least Squares

The analytic solution employs the ordinary least squares (OLS) method to determine model parameters in closed form. Given the linear model:

$$
f(X) = \beta_0 + X\beta_1 \quad \text{or} \quad \hat{\mathbf{y}} = \mathbf{X}_b\boldsymbol{\beta}
$$

where:
- $X$ is the feature (years of experience)
- $\mathbf{X}_b$ is the $N \times 2$ augmented feature matrix with a column of ones for the intercept term
- $\boldsymbol{\beta} = (\beta_0, \beta_1)^T$ are the model parameters ($\beta_0$ is the intercept, $\beta_1$ is the slope)

### Deriving the Closed-Form Solution

The parameters $\boldsymbol{\beta}$ are computed by minimizing the residual sum of squares (RSS):

$$
\text{RSS}(\boldsymbol{\beta}) = \sum_{i=1}^{N}(y_i - f(x_i))^2 = (\mathbf{y} - \mathbf{X}_b\boldsymbol{\beta})^T(\mathbf{y} - \mathbf{X}_b\boldsymbol{\beta})
$$

Expanding this expression:

$$
\text{RSS}(\boldsymbol{\beta}) = \mathbf{y}^T\mathbf{y} - 2\boldsymbol{\beta}^T\mathbf{X}_b^T\mathbf{y} + \boldsymbol{\beta}^T\mathbf{X}_b^T\mathbf{X}_b\boldsymbol{\beta}
$$

Taking the derivative with respect to $\boldsymbol{\beta}$ and setting it to zero:

$$
\frac{\partial \text{RSS}}{\partial \boldsymbol{\beta}} = -2\mathbf{X}_b^T\mathbf{y} + 2\mathbf{X}_b^T\mathbf{X}_b\boldsymbol{\beta} = 0
$$

Solving for $\boldsymbol{\beta}$ yields the **Normal Equation**:

$$
\boldsymbol{\beta} = (\mathbf{X}_b^T\mathbf{X}_b)^{-1}\mathbf{X}_b^T\mathbf{y}
$$

This closed-form solution is exact (up to numerical precision) and requires no hyperparameter tuning.

### Advantages and Limitations

**Advantages:**
- Exact solution in one step
- No hyperparameters to tune
- Mathematically elegant

**Limitations:**
- Requires matrix inversion: $O(n^3)$ complexity
- Computationally expensive for large datasets
- Numerical instability if $\mathbf{X}_b^T\mathbf{X}_b$ is singular or near-singular

---

## Part 2: The Numerical Method - Gradient Descent

The numerical approach implements gradient descent optimization to iteratively find the optimal parameters. Instead of computing the solution directly, we start with random parameters and gradually improve them.

### Cost Function

We use the **Mean Squared Error (MSE)** as our cost function:

$$
J(w, b) = \frac{1}{N}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2
$$

where $\hat{y}_i = wx_i + b$ is the prediction, $w$ is the weight (slope), and $b$ is the bias (intercept).

### Computing Gradients

The parameters are updated iteratively using the gradient of the cost function:

$$
w = w - \alpha \frac{\partial J}{\partial w}, \quad b = b - \alpha \frac{\partial J}{\partial b}
$$

where $\alpha$ is the learning rate, and the gradients are:

$$
\frac{\partial J}{\partial w} = -\frac{2}{N}\sum_{i=1}^{N} x_i(y_i - \hat{y}_i)
$$

$$
\frac{\partial J}{\partial b} = -\frac{2}{N}\sum_{i=1}^{N} (y_i - \hat{y}_i)
$$

### The Algorithm

1. **Initialize** parameters: $w = 0, b = 0$ (or random values)
2. **For each iteration:**
   - Compute predictions: $\hat{y}_i = wx_i + b$
   - Calculate cost: $J(w, b)$
   - Compute gradients: $\frac{\partial J}{\partial w}, \frac{\partial J}{\partial b}$
   - Update parameters: $w \leftarrow w - \alpha \frac{\partial J}{\partial w}, b \leftarrow b - \alpha \frac{\partial J}{\partial b}$
3. **Repeat** until convergence or max iterations reached

### Advantages and Limitations

**Advantages:**
- Scales well to large datasets
- Memory efficient (no matrix inversion)
- Can be extended to more complex models easily

**Limitations:**
- Requires hyperparameter tuning (learning rate, iterations)
- Convergence not guaranteed with poor hyperparameters
- May get stuck in local minima for non-convex problems (not an issue for linear regression)

---

## Part 3: Scikit-learn Implementation

Scikit-learn's `LinearRegression` provides a highly optimized implementation. Under the hood, it uses efficient linear algebra routines (LAPACK) to solve the normal equation or, for large datasets, uses singular value decomposition (SVD) for numerical stability.

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X, y)
predictions = model.predict(X)
```

This serves as our benchmark for validating the custom implementations.

---

## Part 4: Comparing the Methods

### Implementation Structure

The code is organized as a Python class containing three main methods:

- `univariate_linear_regression()`: Analytic solution using matrix operations
- `Gradient_descent()`: Numerical solution with iterative optimization
- `skleanrn_LR()`: Wrapper for scikit-learn's implementation

### Visualizations

The implementation includes comprehensive visualizations:

1. **Regression plots:** Comparing all three methods on the same data
2. **Cost convergence:** Tracking how gradient descent cost decreases over iterations
3. **Parameter space:** 3D surface plot showing the cost landscape and the path gradient descent takes

### Performance Metrics

Both custom implementations are validated using:

- **Root Mean Squared Error (RMSE):** Measures prediction error magnitude

$$
\text{RMSE} = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2}
$$

- **R² Score (Coefficient of Determination):** Measures goodness of fit

$$
R^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}
$$

where $\bar{y}$ is the mean of the true values. $R^2 = 1$ indicates perfect fit, $R^2 = 0$ indicates the model performs no better than predicting the mean.

---

## Results and Insights

### Key Findings

1. **Identical Results:** All three methods produce virtually identical parameters and predictions (within numerical precision)
2. **Convergence Analysis:** Gradient descent converges smoothly, with cost decreasing exponentially in early iterations
3. **Computational Trade-offs:**
   - Analytic: Instant but memory-intensive
   - Gradient descent: Slower but scalable
   - Sklearn: Best of both worlds with automatic method selection

### When to Use Each Method

**Analytic (Normal Equation):**
- Small to medium datasets (< 10,000 samples)
- When you need the exact solution
- Educational purposes to understand the math

**Gradient Descent:**
- Large datasets where matrix inversion is prohibitive
- When building intuition for optimization
- As a foundation for more complex models (regularization, neural networks)

**Scikit-learn:**
- Production code
- When you need reliability and performance
- When you don't need to understand the internals

---

## Mathematical Connections

### Why Do They Converge?

Linear regression with MSE is a **convex optimization problem**. The cost function forms a bowl-shaped surface (quadratic) with a single global minimum. This guarantees that:

1. The analytic solution finds the exact minimum
2. Gradient descent will converge to the same minimum (with appropriate learning rate)
3. Both match scikit-learn's implementation

### The Gradient Descent Path

Visualizing the cost surface reveals how gradient descent navigates the parameter space. It takes steps proportional to the negative gradient, always moving downhill, eventually settling at the minimum where $\nabla J = 0$.

---

## Implementation Highlights

### Dataset Examples

**1. Real-world salary data:**
- Demonstrates practical application
- Shows how both methods handle real noise

**2. Synthetic linear data:**
- Perfect for validation (known ground truth)
- Confirms mathematical correctness

**3. Synthetic data with offset:**
- Tests intercept term handling
- Validates bias parameter learning

Each example includes:
- Regression line visualization
- Cost convergence plot (for gradient descent)
- Performance metrics comparison

---

## Key Takeaways

1. **Understanding beats black boxes:** Implementing from scratch reveals the elegant mathematics underlying linear regression
2. **Multiple paths, same destination:** Analytic and numerical methods arrive at identical solutions through different means
3. **Trade-offs matter:** Choose the method based on dataset size, computational resources, and learning objectives
4. **Convexity is powerful:** The convex nature of linear regression makes it uniquely solvable both analytically and numerically
5. **NumPy is sufficient:** You don't need deep learning frameworks for foundational algorithms

---

## What's Next?

Possible extensions to explore:
- **Regularization:** Ridge (L2) and Lasso (L1) regression
- **Multivariate regression:** Multiple features
- **Polynomial regression:** Non-linear relationships with linear methods
- **Stochastic gradient descent:** Mini-batch optimization for massive datasets
- **Closed-form vs iterative for Ridge:** Comparing methods with regularization

---

## Implementation Details

**Libraries Used:**
- **NumPy:** Matrix operations and numerical computing
- **Pandas:** Data manipulation
- **Matplotlib:** Visualization
- **scikit-learn:** Benchmark implementation

**Code Organization:**
```python
class LinearRegressionComparison:
    def univariate_linear_regression(X, y)
    def Gradient_descent(X, y, learning_rate, iterations)
    def skleanrn_LR(X, y)
```

**[View the complete implementation on Kaggle →](https://www.kaggle.com/code/jvachier/lr-analytic-numerical-methods-vs-sklearn)**

---

## References

Hastie, T., Tibshirani, R., & Friedman, J. (2017). *The Elements of Statistical Learning: Data Mining, Inference, and Prediction* (2nd ed.). Springer.

---

## Additional Resources

- [Kaggle Notebook (Original)](https://www.kaggle.com/code/jvachier/linear-regression)
- [Kaggle Notebook (Full Implementation)](https://www.kaggle.com/code/jvachier/lr-analytic-numerical-methods-vs-sklearn)

---

**Tags:** #MachineLearning #LinearRegression #GradientDescent #Mathematics #NumPy #FromScratch #Optimization

*Created: May 2022 | Published: December 2025*
