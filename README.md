# Micro Bundle Adjustment 🌱
Getting into bundle adjustment solvers can be hard.
To make it easier, I have made a very basic pure Pytorch implementation, useful for teaching/learning.
Despite being less than 100 lines of code, it can handle 10^6 3D points with ease, due to utilizing sparsity and GPU.

# Features

- [x] Basic Implementation of two-view bundle adjustment for any type of camera

# TODO List
- [x] Example with residual function
- [x] Generalize implementation to K-views
- [1/2] Clean up and document code
- []  

# Usage
See [demos](demos).

Basically you define a (non-batched) function for your residuals, which takes in your observations, camera params, and 3D points and outputs the residual.
Sending this together with an initial guess is all you need to do!
Gradients are computed automatically, and you can use arbitrary camera models.
