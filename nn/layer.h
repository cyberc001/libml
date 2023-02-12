#ifndef LAYER_H
#define LAYER_H

#include "math/mat.h"
#include "defs.h"

typedef struct {
	mat weights;

	/* intermediary calculated values, for convenience */
	vec prevalues; // values calculated before activation function was applied
	vec values;

	/* parameters */
	int flags;
	activation_func actfunc; // activation function
	activation_func d_actfunc; // activation function derivative
} nn_layer;

/***** Layer manipulation functions *****/

/* Note:
 * sz / next_sz include bias neuron (if it's present).
 */
nn_layer nn_layer_create(size_t sz, size_t next_sz, int flags,
							activation_func actfunc, activation_func d_actfunc);

void nn_layer_free(nn_layer* lr);

/* Note:
 * rand() function is seeded automatically.
 */
void nn_layer_randomize_weights(nn_layer* lr);

/***** Display (debug) functions *****/

void nn_layer_print_weights(nn_layer* lr);

#endif
