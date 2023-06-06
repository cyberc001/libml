#include "defs.h"
#include <math.h>
#include <stdarg.h>

/* Common activation functions */
double identity(double x) { return x; }
double sigmoid(double x) { return 1 / (1 + pow(M_E, -x)); }
double tanh(double x) { return (pow(M_E, 2*x) - 1) / (pow(M_E, 2*x) + 1); }
double relu(double x) { return fmax(x, 0); }
double hard_tanh(double x) { return fmax(fmin(x, 1), -1); }

/* Common activation functions' derivatives */
double d_identity(double x) { return 1; }
double d_sigmoid(double x) { return pow(M_E, -x) / pow(1 + pow(M_E, -x), 2); }
double d_tanh(double x) { return 4*pow(M_E, 2*x) / pow(pow(M_E, 2*x) + 1, 2); }
double d_relu(double x) { return x >= 0 ? 1 : 0; }
double d_hard_tanh(double x) { return x >= -1 && x <= 1 ? 1 : 0; }

/* Common loss functions */
double squared_loss(vec y, vec _y, ...)
{
	double l = 0;
	for(size_t i = 0; i < y.n; ++i)
		l += pow(y.data[i] - _y.data[i], 2);
	return l / y.n;
}
double neg_log_likelihood_ns_loss(vec y, vec _y, ...)
{
	va_list vargs; va_start(vargs, _y);
	struct neg_log_likelihood_ns_loss_args* args = va_arg(vargs, struct neg_log_likelihood_ns_loss_args*);
	va_end(vargs);

	double l = -log(_y.data[args->neg_idx[0]]);
	for(size_t i = 1; i < args->neg_ln + 1; ++i)
		l -= log(sigmoid(-args->pre_y.data[args->neg_idx[i]]));
	return l;
}

/* Common loss functions' partial derivatives */
double d_squared_loss(vec y, vec _y, size_t i, ...)
{
	return 2*(_y.data[i] - y.data[i]) / y.n;
}
double d_neg_log_likelihood_ns_loss(vec y, vec _y, size_t i, ...)
{
	va_list vargs; va_start(vargs, i);
	struct neg_log_likelihood_ns_loss_args* args = va_arg(vargs, struct neg_log_likelihood_ns_loss_args*);
	va_end(vargs);

	if(args->neg_idx[0] == i)
		return _y.data[i] - 1;
	else{
		for(size_t j = 1; j < args->neg_ln + 1; ++j)
			if(args->neg_idx[j] == i)
				return _y.data[i];
		return 0;
	}
}
