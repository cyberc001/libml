#include <stdio.h>
#include "network.h"

int main()
{
	const size_t layer_cnt = 4;
	const size_t d = 2, p = 2;
	nn_network nw = nn_network_create(layer_cnt, squared_loss, d_squared_loss, 0.1);
	nw.layers[0] = nn_layer_create(d + p, 4*p, 0, relu, d_relu);
	nw.layers[1] = nn_layer_create(2*p, 4*p, 0, relu, d_relu);
	nw.layers[2] = nn_layer_create(2*p, 4*p, 0, relu, d_relu);
	nw.layers[3] = nn_layer_create(p, 0, 0, relu, d_relu);
	nn_network_randomize_weights(&nw);
	nn_network_lstm_init(&nw);

	nn_training_set set = nn_training_set_create(NN_TRAINING_SET_TYPE_SEQUENCE_DOUBLE, 500, d, p);
	vec in = vec_create(d);
	vec out = vec_create(p);
	for(size_t i = 0; i < set.size; ++i){
		double seq0 = (rand() % 100) / 1000.;
		double seq1 = (rand() % 100) / 1000.;
		for(size_t j = 0; j < 5; ++j){
			in.data[0] = seq0;
			in.data[1] = seq1;
			seq0 = out.data[0] = in.data[0] + 0.1;
			seq1 = out.data[1] = in.data[1] + 0.1;
			/*if(i <= 5){
				in.n = d; printf("input: "); vec_print(in);
				out.n = p; printf("output: "); vec_print(out);
			}*/
			in.n = out.n = j;
			set.set_input(&set, in, i);
			set.set_output(&set, out, i);
		}
	}

	printf("BEFORE:\n"); nn_network_print_weights(&nw);
	nn_network_lstm_train(&nw, &set, 100, 0, NNFLAG_DEBUG_LOSS);
	printf("AFTER:\n"); nn_network_print_weights(&nw);

	for(size_t i = 0; i < 5; ++i){
		vec input = vec_create(d);
		mat h_in = mat_create(layer_cnt - 1, p);
		mat_zero(h_in);
		mat h_out = mat_create(layer_cnt - 1, p);

		double seq0 = (rand() % 100) / 1000.;
		double seq1 = (rand() % 100) / 1000.;
		input.data[0] = seq0;
		input.data[1] = seq1;
		nn_network_lstm_reset_state(&nw);
		vec output = nn_network_lstm_feedforward(&nw, input, h_in, h_out);

		printf("input: "); vec_print(input);
		printf("output: "); vec_print(output);

		vec_free(input);
		vec_free(output);
		mat_free(h_in);
		mat_free(h_out);
	}

	// memory cleanup
	vec_free(in);
	vec_free(out);
	nn_training_set_free(&set);
	nn_network_free(&nw);
}
