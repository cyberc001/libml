#include "layer.h"
#include <time.h>
#include <stdio.h>

/***** Layer manipulation functions *****/

nn_layer nn_layer_create(size_t sz, size_t next_sz, int flags, activation_func actfunc, activation_func d_actfunc)
{
	nn_layer out = (nn_layer){weights: mat_create(sz + !!(flags & NNFLAG_LAYER_HAS_BIAS), next_sz),
								flags: flags,
								actfunc: actfunc, d_actfunc: d_actfunc,
								prevalues: vec_create(sz), values: vec_create(sz)};
	return out;
}

void nn_layer_free(nn_layer* lr)
{
	mat_free(lr->weights);
	vec_free(lr->prevalues);
	vec_free(lr->values);
}

static int __rand_init = 0;
void nn_layer_randomize_weights(nn_layer* lr)
{
	if(!__rand_init){
		srand(time(NULL));
		__rand_init = 1;
	}
	size_t sz = lr->weights.n * lr->weights.m;
	for(size_t i = 0; i < sz; ++i)
		lr->weights.data[i] = rand() / (double)RAND_MAX;
}

/***** Display (debug) functions *****/

void nn_layer_print_weights(nn_layer* lr)
{
	mat_print(lr->weights);
}
