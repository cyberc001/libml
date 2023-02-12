#include "network.h"
#include <stdio.h>
#include <math.h>

/***** Network manipulation functions *****/

nn_network nn_network_create(size_t layers_cnt,
								loss_func lossfunc, loss_d_func d_lossfunc,
								double learn_rate)
{
	nn_network nw = {layers: malloc(sizeof(nn_layer) * layers_cnt), layers_cnt: layers_cnt,
						lossfunc: lossfunc, d_lossfunc: d_lossfunc,
						learn_rate: learn_rate};
	return nw;
}

void nn_network_free(nn_network* nw)
{
	for(size_t i = 0; i < nw->layers_cnt; ++i)
		nn_layer_free(&nw->layers[i]);
	free(nw->layers);
}

void nn_network_randomize_weights(nn_network* nw)
{
	for(size_t i = 0; i < nw->layers_cnt; ++i)
		nn_layer_randomize_weights(&nw->layers[i]);
}

/***** Training functions *****/
vec nn_network_feedforward(nn_network* nw, vec input)
{
	// set input layer's calculated values as input values (thanks cap)
	vec_free(nw->layers[0].values);
	nw->layers[0].values = vec_dup(input);
	mat data = mat_from_vec_row(input);
	for(size_t i = 1; i < nw->layers_cnt; ++i){
		nn_layer* lr = &nw->layers[i];
		nn_layer* lr_prev = &nw->layers[i-1];
		vec_free(lr->prevalues);
		vec_free(lr->values);
		// multiply current row vector by weights matrix, effectively calculating values in each neuron in the current layer
		mat data2 = mat_mul(data, lr_prev->weights);
		mat_free(data);
		data = data2;
		lr->prevalues = vec_from_mat_row(data);
		// apply activation function to each element (column) in current row vector
		for(size_t j = 0; j < data.n; ++j)
			data.data[j] = lr->actfunc(data.data[j]);
		lr->values = vec_from_mat_row(data);
	}
	vec res = vec_from_mat_row(data);
	mat_free(data);
	return res;
}

static vec loss_gradient(loss_d_func d_lossfunc, vec expected, vec predicted)
{
	vec grad = vec_create(predicted.n);
	for(size_t i = 0; i < predicted.n; ++i)
		grad.data[i] = d_lossfunc(expected, predicted, i, NULL);
	return grad;
}
static vec d_act(nn_layer* lr)
{ // returns a vector of activation function derivatives calculated for the layer's pre-activation output values
	vec out = vec_create(lr->prevalues.n);
	for(size_t i = 0; i < out.n; ++i)
		out.data[i] = lr->d_actfunc(lr->prevalues.data[i]);
	return out;
}
static void update_weights(nn_network* nw,
							nn_layer* lr, nn_layer* prev_lr,
							vec delta, vec input)
{
	mat delta_m = mat_from_vec_column(delta);
	mat input_m = mat_from_vec_row(prev_lr ? prev_lr->values : input); // x_i-1 is either output from previous layer or neural network input
	mat deriv = mat_mul(delta_m, input_m);
	mat_free(delta_m); // TODO optimise this with in-place multiplication (also pass in arguments as matrices or make functions that multiply vectors and matrices)
	mat_free(input_m);

	mat w = mat_smul(lr->weights, nw->learn_rate);
	mat correction = mat_emul(w, deriv);
	mat_free(w); // TODO optimise this crap aswell
	mat_free(deriv);

	mat_psub(lr->weights, correction);
	mat_free(correction);
}

/* Credits to https://sudeepraja.github.io/Neural/ */
void nn_network_backpropagate(nn_network* nw,
								vec expected, vec predicted,
								vec input)
{
	vec lossg = loss_gradient(nw->d_lossfunc, expected, predicted);
	vec dact = d_act(&nw->layers[nw->layers_cnt - 1]);
	vec_pemul(lossg, dact); // lossg = delta_L
	vec_free(dact);

	update_weights(nw, &nw->layers[nw->layers_cnt - 1],
					nw->layers_cnt >= 2 ? &nw->layers[nw->layers_cnt - 2] : NULL,
					lossg, input);

	// calculate delta_i, from L - 1 to 1
	vec next_delta = lossg;
	for(size_t i = nw->layers_cnt - 2;; --i){
		nn_layer* lr = &nw->layers[i];
		vec dact = d_act(lr);

		// multiply i+1th weight matrix by i+1th delta
		mat next_delta_m = mat_from_vec_column(next_delta);
		mat weight_by_delta = mat_mul(lr->weights, next_delta_m);
		mat_free(next_delta_m);
		vec weight_by_delta_v = vec_from_mat_column(weight_by_delta);
		mat_free(weight_by_delta);
		vec_pemul(weight_by_delta_v, dact);
		vec_free(dact);

		update_weights(nw, lr, i == 0 ? NULL : &nw->layers[i - 1],
				weight_by_delta_v, input);

		vec_free(next_delta);
		next_delta = weight_by_delta_v;
		if(i == 0) break;
	}
	vec_free(next_delta);
}

void nn_network_train(nn_network* nw, nn_training_set* set,
						size_t epochs, double loss_target,
						int debug_flags)
{
	nn_network_randomize_weights(nw);

	int seek_loss = (epochs == 0);
	double total_loss = INFINITY;
	vec input = vec_create(nw->layers[0].values.n);
	vec output = vec_create(nw->layers[nw->layers_cnt - 1].values.n);

	for(; epochs > 0 || seek_loss; --epochs){
		set->shuffle(set);
		if(debug_flags & NNFLAG_DEBUG_LOSS)
			printf("loss: %g ", total_loss);
		total_loss = 0;

		for(size_t i = 0; i < set->size; ++i){
			set->get_input(set, input, i);
			set->get_output(set, output, i);
			vec predicted = nn_network_feedforward(nw, input);
			double loss = nw->lossfunc(output, predicted, NULL);
			total_loss += loss;

			nn_network_backpropagate(nw, output, predicted,
										input);

			vec_free(predicted);
		}
		total_loss /= set->size;
		if(debug_flags & NNFLAG_DEBUG_LOSS)
			printf("--> %g\n", total_loss);
		if(seek_loss && total_loss <= loss_target)
			break;
	}

	vec_free(input);
	vec_free(output);
}

/***** Negative sampling skip-gram training functions *****/

static vec sgns_loss_gradient(loss_d_func d_lossfunc, vec expected, vec predicted, struct neg_log_likelihood_ns_loss_args* ns_dat)
{
	vec grad = vec_create(1 + ns_dat->neg_ln);
	for(size_t i = 0; i < ns_dat->neg_ln + 1; ++i)
		grad.data[i] = d_lossfunc(expected, predicted, ns_dat->neg_idx[i], ns_dat);
	return grad;
}

void nn_network_sgns_backpropagate(nn_network* nw, vec expected, vec predicted,
									vec input, size_t input_idx,
									struct neg_log_likelihood_ns_loss_args* ns_dat)
{
	vec grad = sgns_loss_gradient(nw->d_lossfunc, expected, predicted, ns_dat);
	// compute updates for output matrix
	mat grad_m = mat_from_vec_column(grad);
	mat h_m = mat_from_vec_row(nw->layers[1].values);
	mat delta_w_out = mat_mul(grad_m, h_m);
	mat_free(h_m);
	mat_free(grad_m);
	mat_psmul(delta_w_out, nw->learn_rate);
	// compute updates for input matrix
	grad_m = mat_from_vec_row(grad);
	mat w_out = spcmat_tran(nw->layers[1].weights, ns_dat->neg_ln + 1, ns_dat->neg_idx);
	mat delta_w_in = mat_mul(grad_m, w_out);
	mat_free(w_out);
	mat_free(grad_m);
	mat_psmul(delta_w_in, nw->learn_rate);
	// apply updates
	mat delta_w_out_t = mat_tran(delta_w_out);
	mat_free(delta_w_out);

	sprmat_psub1(nw->layers[0].weights, delta_w_in, 1, &input_idx);
	size_t in_n = nw->layers[0].values.n;
	size_t pos_n = nw->layers[2].values.n / in_n;
	size_t* neg_idx = malloc(sizeof(size_t) * (ns_dat->neg_ln + 1));
	memcpy(neg_idx, ns_dat->neg_idx, sizeof(size_t) * (ns_dat->neg_ln + 1));
	for(size_t i = 0; i < pos_n; ++i){
		spcmat_psub1(nw->layers[1].weights, delta_w_out_t, ns_dat->neg_ln + 1, neg_idx);
		for(size_t j = 0; j < ns_dat->neg_ln + 1; ++j)
			neg_idx[j] += in_n;
	}
	vec_free(grad);
	mat_free(delta_w_out_t);
	mat_free(delta_w_in);
	free(neg_idx);
}

void nn_network_sgns_train(nn_network* nw, nn_training_set* set,
						size_t epochs, double loss_target,
						int debug_flags,
						size_t neg_sample_amt, noise_distr* distr)
{
	nn_network_randomize_weights(nw);

	int seek_loss = (epochs == 0);
	double total_loss = INFINITY;
	double min_loss = INFINITY;
	vec input = vec_create(nw->layers[0].values.n);
	vec output = vec_create(nw->layers[nw->layers_cnt - 1].values.n);

	size_t pos_sample_amt = set->out_size;
	size_t* neg_samples = malloc(sizeof(size_t) * (neg_sample_amt + 1));
	struct neg_log_likelihood_ns_loss_args ns_dat;
	ns_dat.neg_ln = neg_sample_amt;
	ns_dat.neg_idx = neg_samples;

	for(; epochs > 0 || seek_loss; --epochs){
		set->shuffle(set);
		if(debug_flags && NNFLAG_DEBUG_LOSS)
			printf("loss: %-10g min loss: %-10g\n", total_loss, min_loss);
		total_loss = 0;

		for(size_t i = 0; i < set->size; ++i){
			set->get_input(set, input, i);
			set->get_output(set, output, i);

			double loss = 0;

			vec predicted;
			for(size_t j = 0; j < pos_sample_amt; ++j){
				neg_samples[0] = ((size_t*)set->data_out + set->out_size * i)[j];
				for(size_t k = 1; k < neg_sample_amt + 1; ++k)
					neg_samples[k] = noise_distr_pick(distr, set->out_size, (size_t*)set->data_out + set->out_size * i);

				predicted = nn_network_feedforward(nw, input);
				ns_dat.pre_y = nw->layers[nw->layers_cnt - 1].prevalues;
				loss += nw->lossfunc(output, predicted, &ns_dat);

				nn_network_sgns_backpropagate(nw, output, predicted, input, ((size_t*)set->data_in + set->in_size * i)[0], &ns_dat);
				if(j != pos_sample_amt - 1) 
					vec_free(predicted);
			}
			total_loss += loss / pos_sample_amt;

			if(debug_flags & NNFLAG_DEBUG_VALUES){
				printf("expected values\t");
				for(size_t j = 0; j < set->out_size; ++j)
					printf("%lu ", ((size_t*)set->data_out + i * set->out_size)[j]);
				puts("");
				printf("predicted probabilities\t");

				for(size_t j = 0; j < set->out_size; ++j)
					printf("%lg ", predicted.data[((size_t*)set->data_out + i * set->out_size)[j]]);
				puts("");
			}
			vec_free(predicted);
		}
		total_loss /= set->size;
		if(seek_loss && total_loss <= loss_target)
			break;
		if(total_loss < min_loss)
			min_loss = total_loss;
	}

	vec_free(input);
	vec_free(output);
}

/***** Display (debug) functions *****/

void nn_network_print_weights(nn_network* nw)
{
	printf("Input layer:\n" _PRINT_SEP);
	nn_layer_print_weights(&nw->layers[0]);
	printf(_PRINT_SEP);
	for(size_t i = 1; i < nw->layers_cnt - 1; ++i){
		printf("Hidden layer #%lu:\n" _PRINT_SEP, i);
		nn_layer_print_weights(&nw->layers[i]);
		printf(_PRINT_SEP);
	}
	printf("Output layer:\n" _PRINT_SEP);
	nn_layer_print_weights(&nw->layers[nw->layers_cnt - 1]);
	printf(_PRINT_SEP);
}
