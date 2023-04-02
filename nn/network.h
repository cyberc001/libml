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

/***** Network manipulation functions *****/

nn_network nn_network_create(size_t layers_cnt,
								loss_func lossfunc, loss_d_func d_lossfunc,
								double learn_rate);

void nn_network_free(nn_network* nw);

void nn_network_randomize_weights(nn_network* nw);

/***** Conventional training functions *****/

/* Note: input vector should have the same size as input layer. */
vec nn_network_feedforward(nn_network* nw, vec input);
/* Should be called after feedforward(). */
void nn_network_backpropagate(nn_network* nw, vec expected, vec predicted);

void nn_network_train(nn_network* nw, nn_training_set* set,
						size_t epochs, double loss_target,
						int flags);

/***** Negative sampling skip-gram training functions *****/
/* Network should consist of 3 layers (1 input / 1 hidden / 1 output).
 * Input and hidden layers should use identity activation,
 * Output should use sigmoid activation. */

void nn_network_sgns_backpropagate(nn_network* nw, vec expected, vec predicted,
								vec input, size_t input_idx,
								struct neg_log_likelihood_ns_loss_args* ns_dat);

/* Training set should be one-hot encoded. */
void nn_network_sgns_train(nn_network* nw, nn_training_set* set,
						size_t epochs, double loss_target,
						int flags,
						size_t neg_sample_amt, noise_distr* distr);

/***** Saving functions *****/

void nn_network_save_weights(FILE* fd, nn_network* nw);
int nn_network_load_weights(FILE* fd, nn_network* nw);

/***** Display (debug) functions *****/

void nn_network_print_weights(nn_network* nw);

#endif
