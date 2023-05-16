#ifndef TRAINING_SET_H
#define TRAINING_SET_H

#include "math/vec.h"
#include <stdio.h>

#define NN_TRAINING_SET_TYPE_DOUBLE 1
#define NN_TRAINING_SET_TYPE_ONEHOT 2
#define NN_TRAINING_SET_TYPE_SEQUENCE_DOUBLE 3 /* Get/set functions require one additional vaarg argument of type (size_t) that indicates timestamp number. Probe from 0 until it returns -inf as the first element in the output vector. */
#define NN_TRAINING_SET_TYPE_SEQUENCE_ONEHOT 4 /* same as for sequence of doubles. */

typedef struct nn_training_set nn_training_set;
struct nn_training_set {
	int data_type;
	size_t size;
	size_t in_size, out_size;
	void *data_in, *data_out;
	/* vector argument should already be created by vec_create(); call */
	void (*get_input)(nn_training_set*, vec, size_t, ...);
	void (*get_output)(nn_training_set*, vec, size_t, ...);
	void (*set_input)(nn_training_set*, vec, size_t, ...);
	void (*set_output)(nn_training_set*, vec, size_t, ...);
	void (*shuffle)(nn_training_set*); // doesn't have to seed RNG
};

nn_training_set nn_training_set_create(int data_type, size_t size, size_t in_size, size_t out_size);
void nn_training_set_free(nn_training_set* set);

void nn_training_set_expand(nn_training_set* set, size_t how_much);
void nn_training_set_add(nn_training_set* set, vec v_in, vec v_out);

void nn_training_set_save(FILE* fd, nn_training_set* set);
nn_training_set nn_training_set_load(FILE* fd);

#endif
