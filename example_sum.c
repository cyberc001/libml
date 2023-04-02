#include <stdio.h>
#include "network.h"

int main()
{
	nn_network nw = nn_network_create(2, squared_loss, d_squared_loss, 0.1);
	nw.layers[0] = nn_layer_create(2, 1, NNFLAG_LAYER_HAS_BIAS, relu, d_relu);
	nw.layers[1] = nn_layer_create(1, 0, 0, relu, d_relu);

	nn_training_set set = nn_training_set_create(NN_TRAINING_SET_TYPE_DOUBLE, 200, 2, 1);
	vec in = vec_create(2);
	vec out = vec_create(1);
	for(size_t i = 0; i < set.size; ++i){
		double in1 = (rand() % 100) / 1000.,
			   in2 = (rand() % 100) / 1000.;
		in.data[0] = in1;
		in.data[1] = in2;
		out.data[0] = in1 + in2 + 0.5;
		set.set_input(&set, in, i);
		set.set_output(&set, out, i);
		printf("%g + %g + 0.5 = %g\n", in1, in2, in1 + in2 + 0.5);
	}

	nn_network_train(&nw, &set, 100, 0, NNFLAG_DEBUG_LOSS);
	//nn_network_train(&nw, &set, 0, 1e-30, 0);
	nn_network_print_weights(&nw);
	
	for(size_t i = 0; i < 5; ++i){
		vec in = vec_create(2);
		double in1 = (rand() % 100) / 1000.,
			   in2 = (rand() % 100) / 1000.;
		in.data[0] = in1; in.data[1] = in2;
		vec out = nn_network_feedforward(&nw, in);
		printf("%g + %g + 0.5 = %g\n", in1, in2, out.data[0]);
		vec_free(in);
		vec_free(out);
	}

	// memory cleanup
	nn_training_set_free(&set);
	nn_network_free(&nw);
}
