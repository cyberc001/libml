#ifndef DEFS_H
#define DEFS_H

#include "math/vec.h"

/* Generic definitions, used throughout the program */
typedef double (*activation_func)(double x);
typedef double (*loss_func)(vec y, vec _y, ...); // y - accurate value, _y - predicted value
typedef double (*loss_d_func)(vec y, vec _y, size_t i, ...); // i - vector component index to take derivative in regard to

/* Common activation functions */
double identity(double x);
double sigmoid(double x);
double relu(double x); // Rectified Linear Unit, ReLU
double hard_tanh(double x);

/* Common activation functions' derivatives */
double d_identity(double x);
double d_sigmoid(double x);
double d_tanh(double x);
double d_relu(double x);
double d_hard_tanh(double x);

/* Common loss functions */
double squared_loss(vec y, vec _y, ...);
struct neg_log_likelihood_ns_loss_args {
	vec pre_y; // pre-activation values for output layer
	size_t neg_ln;
	size_t* neg_idx; // indicies of samples; first 1 positive, then negative
};
/* Requires that sigmoid activation function is used for last layer.
 * Use struct neg_log_likelihood_ns_loss_args* as first va_args argument! */
double neg_log_likelihood_ns_loss(vec y, vec _y, ...);

/* Common loss functions' partial derivatives */
double d_squared_loss(vec y, vec _y, size_t i, ...);
/* Requires that sigmoid activation function is used for last layer.
 * Use struct neg_log_likelihood_ns_loss_args* as first va_args argument! */
double d_neg_log_likelihood_ns_loss(vec y, vec _y, size_t i, ...);

/* Other stuff */
#define _PRINT_SEP "--------------------------------------------------\n"

#endif
