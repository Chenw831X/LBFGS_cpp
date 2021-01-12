#include <Eigen/Core>
#include <Eigen/Eigen>

#include <stdexcept>
#include <vector>
#include <iostream>

#include <time.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#endif

double cpuSecond(){
        struct timeval tp;
        gettimeofday(&tp, NULL);
        return ((double)tp.tv_sec + (double)tp.tv_usec * 1e-6);
}

// the size of vector x
const int n = 10;
// define the Rosenbrock function int the following
// we use LBFGS to minimize this function
// input: vector x
// output: objective function value fx and
//         gradient vector at x
double computeFG(const Eigen::VectorXd& x, Eigen::VectorXd& grad)
{
    double fx = 0.0;
    for(int i = 0; i < n; i += 2)
    {
        double t1 = 1.0 - x[i];
        double t2 = 10 * (x[i + 1] - x[i] * x[i]);
        grad[i + 1] = 20 * t2;
        grad[i]     = -2.0 * (x[i] * grad[i + 1] + t1);
        fx += t1 * t1 + t2 * t2;
    }
    return fx;
}

// set up parameters
int lbfgs_m = 6;   // the number of corrections to approximate the inverse Hessian matrix
double lbfgs_epsilon = (double)(1e-6);  // absolute tolerance for convergence test
double lbfgs_epsilon_rel = (double)(1e-5);  // relative tolerance for convergence test
int lbfgs_max_iterations = 100;  // the maximum number of LBFGS iterations
int lbfgs_max_linesearch = 20;   // the maximum number of trials for the line search
double lbfgs_min_step = (double)(1e-20); // the min step length allowed in the line search
double lbfgs_max_step = (double)(1e+20); // the max step length allowed in the line search
double lbfgs_ftol = (double)(1e-4); // c1 in Armijo condition
double lbfgs_wolfe = 0.9;  // c2 in Curvature condition

double lbfgs_theta = 1.0;  // theta * I is the initial approximation to the Hessian matrix
Eigen::MatrixXd lbfgs_s(n, lbfgs_m); // history of s vectors
Eigen::MatrixXd lbfgs_y(n, lbfgs_m); // history of y vectors
Eigen::VectorXd lbfgs_ys(lbfgs_m);   // history of y's values
Eigen::VectorXd lbfgs_alpha(lbfgs_m);// temporary values used in two-loop algorithm
int lbfgs_ncorr = 0; // number of correction vectors in the history
int lbfgs_ptr = lbfgs_m; // a pointer to locate the most recent history
                                                 // s, y, ys are stored in cyclic order

Eigen::VectorXd lbfgs_xp(n);  // the last vector x
Eigen::VectorXd lbfgs_grad(n);// the current gradient
Eigen::VectorXd lbfgs_gradp(n);// the last gradient
Eigen::VectorXd lbfgs_drt(n); // search direction
// Initial guess
Eigen::VectorXd lbfgs_x = Eigen::VectorXd::Zero(n); // the current vector x
double lbfgs_fx;  // the current objective function value

// add correction vectors to BFGS matrix
void LBFGS_add_correction(const Eigen::VectorXd& s, const Eigen::VectorXd& y){
        const int loc = lbfgs_ptr % lbfgs_m;
        lbfgs_s.col(loc) = s;
        lbfgs_y.col(loc) = y;

        const double ys = lbfgs_s.col(loc).dot(lbfgs_y.col(loc));
        lbfgs_ys[loc] = ys;

        lbfgs_theta = lbfgs_y.col(loc).squaredNorm() / ys;

        if(lbfgs_ncorr < lbfgs_m)
                ++lbfgs_ncorr;
        lbfgs_ptr = loc + 1;
}

// two-loop algorithm to compute - H^(-1) * g
void LBFGS_apply_Hv(const Eigen::VectorXd& v, const double& a, Eigen::VectorXd& res){
        res.resize(v.size());

        // forwaed loop
        res = a * v;
        int j = lbfgs_ptr % lbfgs_m;
        for(int i = 0; i < lbfgs_ncorr; ++i){
                j = (j + lbfgs_m - 1) % lbfgs_m;
                lbfgs_alpha[j] = lbfgs_s.col(j).dot(res) / lbfgs_ys[j];
                res -= lbfgs_alpha[j] * lbfgs_y.col(j);
        }

        // apply initial H0
        res /= lbfgs_theta;

        // backward loop
        for(int i = 0; i < lbfgs_ncorr; ++i){
                const double beta = lbfgs_y.col(j).dot(res) / lbfgs_ys[j];
                res += (lbfgs_alpha[j] - beta) * lbfgs_s.col(j);
                j = (j + 1) % lbfgs_m;
        }
}

// backtracking line search based on Regular Wolfe Condition (not Strong Wolfe Condition)
void LBFGS_linesearch(double& fx, Eigen::VectorXd& x, Eigen::VectorXd& grad, double& step, const Eigen::VectorXd& drt, const Eigen::VectorXd& xp){
        //decreasing and increasing factors
        const double dec = 0.5;
        const double inc = 2.1;

        // check the value of step
        if(step <= 0.0)
                throw std::invalid_argument("'step' must be positive!");

        // save the function value at the current x
        const double fx_init = fx;
        // projection of gradient on the search direction
        const double dg_init = grad.dot(drt);
        // make sure drt points to a descent direction
        if(dg_init > 0.0)
                throw std::logic_error("the moving direction increase the objective function value!");
        const double test_decr = lbfgs_ftol * dg_init;
        double width;

        int iter;
        for(iter = 0; iter < lbfgs_max_linesearch; ++iter){
                x = xp + step * drt;
                fx = computeFG(x, grad);

                // Armijo Condition
                if(fx > fx_init + step * test_decr){
                        width = dec;
                }
                else{
                        const double dg = grad.dot(drt);
                        // Curvature Condition
                        if(dg < lbfgs_wolfe * dg_init){
                                width = inc;
                        }
                        else{
                                break;
                        }
                }

                if(step < lbfgs_min_step)
                        throw std::runtime_error("the line search step became smaller than the minimum value allowed!");
                if(step > lbfgs_max_step)
                        throw std::runtime_error("the line search step became larger than the maximum value allowed!");

                step *= width;
        }

        if(iter >= lbfgs_max_linesearch)
                throw std::runtime_error("the line search routine reached the maximum number of iterations!");
}

// use LBFGS to minimize objective function value
int LBFGS(){
        lbfgs_fx = computeFG(lbfgs_x, lbfgs_grad);
        double gnorm = lbfgs_grad.norm();
        // exit if the initial x is already a minimizer
        if(gnorm <= lbfgs_epsilon || gnorm <= lbfgs_epsilon_rel * lbfgs_x.norm()){
                return 0;
        }
        // Initial direction
        lbfgs_drt = -lbfgs_grad;
        // Initial step size
        double step = 1.0 / lbfgs_drt.norm();

        int k = 1;
        for( ; ; ){
                lbfgs_xp = lbfgs_x;
                lbfgs_gradp = lbfgs_grad;

                // linesearch to update fx, x, grad
                LBFGS_linesearch(lbfgs_fx, lbfgs_x, lbfgs_grad, step, lbfgs_drt, lbfgs_xp);

                gnorm = lbfgs_grad.norm();
                // Convergence test
                if(gnorm <= lbfgs_epsilon || gnorm <= lbfgs_epsilon_rel * lbfgs_x.norm()){
                        return k;
                }
                if(lbfgs_max_iterations != 0 && k >= lbfgs_max_iterations){
                        return k;
                }

                // update s and y
                LBFGS_add_correction(lbfgs_x - lbfgs_xp, lbfgs_grad - lbfgs_gradp);
                // compute drt = - H^(-1) * g
                LBFGS_apply_Hv(lbfgs_grad, -1.0, lbfgs_drt);
                // reset step = 1.0 as initial guess for the next line search
                step = 1.0;
                ++k;
        }
        return k;
}
