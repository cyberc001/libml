#include <stdio.h>
#include "network_lstm.h"
#include <math.h>

double toy_squared_loss(vec y, vec _y, ...)
{
	return pow(_y.data[0] - y.data[0], 2);
}
double toy_squared_loss_d(vec y, vec _y, size_t i, ...)
{
	return i == 0 ? 2 * (_y.data[0] - y.data[0]) : 0;
}

int main()
{
	const size_t layer_cnt = 3;
	const size_t d = 2, p = 4;
	nn_network nw = nn_network_create(layer_cnt, toy_squared_loss, toy_squared_loss_d, 0.1);
	nw.layers[0] = nn_layer_create(d + p, 4*p, 0, relu, d_relu);
	nw.layers[1] = nn_layer_create(2*p, 4*p, 0, relu, d_relu);
	nw.layers[2] = nn_layer_create(p, 0, 0, relu, d_relu);
	nn_network_lstm_init(&nw);
	nn_network_randomize_weights(&nw);

	nn_training_set set = nn_training_set_create(NN_TRAINING_SET_TYPE_SEQUENCE_DOUBLE, 1, d, p);
	vec input = vec_create(d), output = vec_create_zero(p);

	input.data[0] = input.data[1] = 0.3;
	output.data[0] = -0.5;
	set.set_input(&set, input, 0, 0);
	set.set_output(&set, output, 0, 0);
	input.data[0] = input.data[1] = 0.1;
	output.data[0] = 0.2;
	set.set_input(&set, input, 0, 1);
	set.set_output(&set, output, 0, 1);
	input.data[0] = input.data[1] = 0.05;
	output.data[0] = 0.1;
	set.set_input(&set, input, 0, 2);
	set.set_output(&set, output, 0, 2);
	input.data[0] = input.data[1] = 0.3;
	output.data[0] = -0.5;
	set.set_input(&set, input, 0, 3);
	set.set_output(&set, output, 0, 3);

	vec_free(input); vec_free(output);

	nn_network_lstm_train(.nw = &nw, .set = &set, .flags = NN_TRAIN_FLAGS_DEBUG);

	nn_network_free(&nw);
	nn_training_set_free(&set);
}
