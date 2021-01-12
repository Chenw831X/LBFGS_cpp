# LBFGS_cpp
LBFGS implementation using Eigen in C++

LBFGS is a kind of quasi-Newton method, which is used to solve the minimization problem without constraints. By storing the vector sequence s, y to approximate the inverse of the Hessian matrix, so as to avoid the time and space cost caused by assembling the Hessian matrix, and also avoid the cost of solving the linear equation (H*p = -g). For a detailed introduction of LBFGS, please refer to: [https://aria42.com/blog/2014/12/understanding-lbfgs](https://aria42.com/blog/2014/12/understanding-lbfgs)

This code can be compiled and run:
```
$ g++ -I/path/to/eigen -o main.cpp -o main
$ ./main
23 iterations
x =
1 1 1 1 1 1 1 1 1 1
f(x) = 1.87948e-19
total time: 0.000476837
```
