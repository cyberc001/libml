#ifndef NETWORK_H
#define NETWORK_H

#include "layer.h"
#include "training_set.h"
#include "math/noise_distr.h"

#define NNFLAG_DEBUG_LOSS 1
#define NNFLAG_DEBUG_VALUES 2

#define NNFLAG_DONT_RANDOMIZE_WEIGHTS 64

#define NN_ERROR_LAYERS_CNT_NOT_MATCH 1
#define NN_ERROR_WEIGHTS_DIM_NOT_MATCH 2

typedef struct {
	nn_layer* layers;
	size_t layers_cnt;

	/* parameters */
	loss_func lossfunc;
	loss_d_func d_lossfunc;
	double learn_rate;
} nn_network;

#define NN_TRAIN_FLAGS_DEBUG 1	// enables debug output, usually for loss functions values

typedef struct nn_train_params nn_train_params;
struct nn_train_params {
	nn_network* nw;
	nn_training_set* set;

	double target_loss; // < 0  => stop only when epoch limit is reached
	size_t epoch_limit; // == 0 => stop only when target loss is reached
	
	int flags;
};
#define NN_TRAINING_PARAMS_DEFAULT(_type, ...) (_type){.target_loss = -1, .epoch_limit = 0, .flags = 0, ##__VA_ARGS__}

/***** Network manipulation functions *****/

nn_network nn_network_create(size_t layers_cnt,
								loss_func lossfunc, loss_d_func d_lossfunc,
								double learn_rate);

void nn_network_free(nn_network* nw);

void nn_network_randomize_weights(nn_network* nw);

/***** Math functions *****/

vec nn_network_loss_grad(loss_d_func d_lossfunc, vec expected, vec predicted);

/***** Saving functions *****/

void nn_network_save_weights(FILE* fd, nn_network* nw);
int nn_network_load_weights(FILE* fd, nn_network* nw);

/***** Display (debug) functions *****/

void nn_network_print_weights(nn_network* nw);

#endif
