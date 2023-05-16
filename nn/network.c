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

/***** Conventional training functions *****/

vec nn_network_feedforward(nn_network* nw, vec input)
{
	vec_free(nw->layers[0].values);
	if(nw->layers[0].flags & NNFLAG_LAYER_HAS_BIAS){
		nw->layers[0].values = vec_dup_resize(input, input.n + 1);
		nw->layers[0].values.data[input.n] = 1;
	}
	else
		nw->layers[0].values = vec_dup(input);
	mat data = mat_from_vec_row(nw->layers[0].values);

	for(size_t i = 1; i < nw->layers_cnt; ++i){
		nn_layer* lr = &nw->layers[i];
		nn_layer* lr_prev = &nw->layers[i-1];
		vec_free(lr->prevalues);
		vec_free(lr->values);
		// multiply current row vector by weights matrix, effectively calculating values in each neuron in the current layer
		mat data2 = mat_mul(data, lr_prev->weights);
		mat_free(data);
		data = data2;
		if(lr->flags & NNFLAG_LAYER_HAS_BIAS){
			data.data = realloc(data.data, sizeof(double) * (data.n + 1));
			data.data[data.n] = 1;
			++data.n;
		}
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
static void update_weights(nn_network* nw, nn_layer* lr, vec delta)
{
	mat delta_m = mat_from_vec_column(delta);
	mat input_m = mat_from_vec_row(lr->values);
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
								vec expected, vec predicted)
{
	vec lossg = loss_gradient(nw->d_lossfunc, expected, predicted);
	vec dact = d_act(&nw->layers[nw->layers_cnt - 1]);
	vec_pemul(lossg, dact); // lossg = delta_L
	vec_free(dact);

	update_weights(nw, &nw->layers[nw->layers_cnt - 2], lossg);

	// calculate delta_i, from L - 1 to 1
	vec next_delta = lossg;
	if(nw->layers_cnt > 2){
		for(size_t i = nw->layers_cnt - 3;; --i){
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

			update_weights(nw, lr, weight_by_delta_v);

			vec_free(next_delta);
			next_delta = weight_by_delta_v;
			if(i == 0) break;
		}
	}
	vec_free(next_delta);
}

void nn_network_train(nn_network* nw, nn_training_set* set,
						size_t epochs, double loss_target,
						int flags)
{
	if(!(flags & NNFLAG_DONT_RANDOMIZE_WEIGHTS))
		nn_network_randomize_weights(nw);

	int seek_loss = (epochs == 0);
	double total_loss = INFINITY;
	vec input = vec_create(nw->layers[0].values.n);
	vec output = vec_create(nw->layers[nw->layers_cnt - 1].values.n);

	for(; epochs > 0 || seek_loss; --epochs){
		set->shuffle(set);
		if(flags & NNFLAG_DEBUG_LOSS)
			printf("loss: %g ", total_loss);
		total_loss = 0;

		for(size_t i = 0; i < set->size; ++i){
			set->get_input(set, input, i);
			set->get_output(set, output, i);
			vec predicted = nn_network_feedforward(nw, input);
			double loss = nw->lossfunc(output, predicted, NULL);
			total_loss += loss;

			nn_network_backpropagate(nw, output, predicted);

			vec_free(predicted);
		}
		total_loss /= set->size;
		if(flags & NNFLAG_DEBUG_LOSS)
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
	mat grad_m = mat_from_vec_column_nocopy(grad);
	mat h_m = mat_from_vec_row(nw->layers[1].values);
	mat delta_w_out = mat_mul(grad_m, h_m);
	mat_free(h_m);
	mat_psmul(delta_w_out, nw->learn_rate);
	// compute updates for input matrix
	grad_m = mat_from_vec_row_nocopy(grad);
	mat w_out = spcmat_tran(nw->layers[1].weights, ns_dat->neg_ln + 1, ns_dat->neg_idx);
	mat delta_w_in = mat_mul(grad_m, w_out);
	mat_free(w_out);
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
						int flags,
						size_t neg_sample_amt, noise_distr* distr)
{
	if(!(flags & NNFLAG_DONT_RANDOMIZE_WEIGHTS))
		nn_network_randomize_weights(nw);

	int seek_loss = (epochs == 0);
	double total_loss = INFINITY;
	vec input = vec_create(nw->layers[0].values.n);
	vec output = vec_create(nw->layers[nw->layers_cnt - 1].values.n);

	size_t pos_sample_amt = set->out_size;
	size_t* neg_samples = malloc(sizeof(size_t) * (neg_sample_amt + 1));
	struct neg_log_likelihood_ns_loss_args ns_dat;
	ns_dat.neg_ln = neg_sample_amt;
	ns_dat.neg_idx = neg_samples;

	double min_loss = INFINITY;
	mat* best_weights = malloc(sizeof(mat) * nw->layers_cnt);
	for(size_t i = 0; i < nw->layers_cnt; ++i)
		best_weights[i] = mat_create(nw->layers[i].weights.m, nw->layers[i].weights.n);

	for(; epochs > 0 || seek_loss; --epochs){
		set->shuffle(set);
		if(flags && NNFLAG_DEBUG_LOSS)
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

			if(flags & NNFLAG_DEBUG_VALUES){
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
		if(total_loss < min_loss){
			min_loss = total_loss;
			for(size_t i = 0; i < nw->layers_cnt; ++i)
				mat_copy_over(best_weights[i], nw->layers[i].weights);
		}
		if(seek_loss && total_loss <= loss_target)
			break;
	}

	if(!seek_loss){
		for(size_t i = 0; i < nw->layers_cnt; ++i)
			mat_copy_over(nw->layers[i].weights, best_weights[i]);
	}

	for(size_t i = 0; i < nw->layers_cnt; ++i)
		mat_free(best_weights[i]);
	free(best_weights);
	free(neg_samples);
	vec_free(input);
	vec_free(output);
}

/***** Long-short term memory training functions *****/

typedef struct {
	vec bias_input;
	vec bias_forget;
	vec bias_output;
	vec bias_cstate;

	// current state
	vec cell_state;
	
	// state history
	mat prev_gates;
	mat prev_hidden_states;
	mat prev_inputs;
	mat prev_cell_states;
} lstm_ldata;

#define LSTM_LAYER_BIAS(lr, _type) (((lstm_ldata*)(lr).data)->bias_ ## _type)
#define LSTM_LAYER_CELL_STATE(lr) (((lstm_ldata*)(lr).data)->cell_state)

#define LSTM_LAYER_PREV_GATES(lr) (((lstm_ldata*)(lr).data)->prev_gates)
#define LSTM_LAYER_PREV_HIDDEN_STATES(lr) (((lstm_ldata*)(lr).data)->prev_hidden_states)
#define LSTM_LAYER_PREV_INPUTS(lr) (((lstm_ldata*)(lr).data)->prev_inputs)
#define LSTM_LAYER_PREV_CELL_STATES(lr) (((lstm_ldata*)(lr).data)->prev_cell_states)

static void nn_network_lstm_free_data(void* _data)
{
	lstm_ldata* data = (lstm_ldata*)_data;
	vec_free(data->bias_input);
	vec_free(data->bias_forget);
	vec_free(data->bias_output);
	vec_free(data->bias_cstate);

	vec_free(data->cell_state);

	mat_free(data->prev_gates);
	mat_free(data->prev_hidden_states);
	mat_free(data->prev_inputs);
	mat_free(data->prev_cell_states);
}

void nn_network_lstm_init(nn_network* nw)
{
	size_t p = nw->layers[0].weights.n / 4;
	for(size_t k = 0; k < nw->layers_cnt - 1; ++k){
		nw->layers[k].data = malloc(sizeof(lstm_ldata));
		nw->layers[k].data_free = nn_network_lstm_free_data;
		LSTM_LAYER_BIAS(nw->layers[k], input) = vec_create_random(p);
		LSTM_LAYER_BIAS(nw->layers[k], forget) = vec_create_random(p);
		LSTM_LAYER_BIAS(nw->layers[k], output) = vec_create_random(p);
		LSTM_LAYER_BIAS(nw->layers[k], cstate) = vec_create_random(p);
		LSTM_LAYER_CELL_STATE(nw->layers[k]) = vec_create(p);
		vec_zero(LSTM_LAYER_CELL_STATE(nw->layers[k]));

		LSTM_LAYER_PREV_GATES(nw->layers[k]) = mat_create(0, 4 * p);
		LSTM_LAYER_PREV_HIDDEN_STATES(nw->layers[k]) = mat_create(0, 2 * p);
		LSTM_LAYER_PREV_INPUTS(nw->layers[k]) = mat_create(0, 2 * p);
		LSTM_LAYER_PREV_CELL_STATES(nw->layers[k]) = mat_create(0, p);
	}
}
void nn_network_lstm_reset_state(nn_network* nw)
{
	size_t p = nw->layers[0].weights.n / 4;
	for(size_t l = 0; l < nw->layers_cnt - 1; ++l){
		vec_zero(LSTM_LAYER_CELL_STATE(nw->layers[l]));
		mat_free(LSTM_LAYER_PREV_GATES(nw->layers[l]));
		mat_free(LSTM_LAYER_PREV_HIDDEN_STATES(nw->layers[l]));
		mat_free(LSTM_LAYER_PREV_INPUTS(nw->layers[l]));
		mat_free(LSTM_LAYER_PREV_CELL_STATES(nw->layers[l]));
		LSTM_LAYER_PREV_GATES(nw->layers[l]) = mat_create(0, 4 * p);
		LSTM_LAYER_PREV_HIDDEN_STATES(nw->layers[l]) = mat_create(0, 2 * p);
		LSTM_LAYER_PREV_INPUTS(nw->layers[l]) = mat_create(0, 2 * p);
		LSTM_LAYER_PREV_CELL_STATES(nw->layers[l]) = mat_create(0, p);
	}
}

static void lstm_add_bias(mat gates, nn_layer* lr, size_t p)
{
	rncmat_padd1(gates, mat_from_vec_row_nocopy(LSTM_LAYER_BIAS(*lr, input)), 0, p);		
	rncmat_padd1(gates, mat_from_vec_row_nocopy(LSTM_LAYER_BIAS(*lr, forget)), p, 2 * p);
	rncmat_padd1(gates, mat_from_vec_row_nocopy(LSTM_LAYER_BIAS(*lr, output)), 2 * p, 3 * p);
	rncmat_padd1(gates, mat_from_vec_row_nocopy(LSTM_LAYER_BIAS(*lr, cstate)), 3 * p, 4 * p);
}

static void lstm_update_cell_state(vec cell_state, mat gates, mat data, size_t p)
{
	// intermediary values are kept in gates matrix for efficiency
	rncmat_pemul1(gates, mat_from_vec_row_nocopy(cell_state), p, 2 * p); // forget gate * previous cell-state
	rncmat_pemul12(gates, gates, 0, p, 3 * p); // input gate * new cell-state
	// copy all calculations above into cell-state
	mat_copy_to_vec_row(cell_state, gates, p); 
	rncmat_padd2(mat_from_vec_row_nocopy(cell_state), gates, 0, p);
	// leak cell-state into hidden state
	for(size_t i = 0; i < p; ++i)
		data.data[i] = gates.data[2 * p + i] * tanh(cell_state.data[i]);
}

vec nn_network_lstm_feedforward(nn_network* nw, vec input, mat h_in, mat h_out)
{
	size_t p = nw->layers[0].weights.n / 4;
	size_t d = nw->layers[0].weights.m - p;

	vec_free(nw->layers[0].values);
	nw->layers[0].values = vec_dup(input);

	// feedforward for (input, hidden state from previous timestamp) pair
	mat data = mat_create(1, d + p); // hidden state vector
	vec_row_copy_to_mat(data, input,); // first d elements are from input data
	mat_copy_to_mat(data, h_in, p, d,); // second p elements are from hidden state vector from previous timestamp
	mat gates = mat_mul(data, nw->layers[0].weights);
	mat_free(data);
	data = mat_create(1, 2 * p); // change hidden state vector length for hidden-to-hidden updates
	mat_zero(data);
	mat_copy_to_mat(data, h_in, p, p,); // get hidden state from previous timestamp
	mat_append_row_vec(LSTM_LAYER_PREV_INPUTS(nw->layers[0]), vec_from_mat_nocopy(data));

	lstm_add_bias(gates, &nw->layers[0], p);
	mat_apply_activation_func_range(gates, sigmoid, 0, 3 * p);
	mat_apply_activation_func_range(gates, tanh, 3 * p,);
	mat_append_row_vec(LSTM_LAYER_PREV_GATES(nw->layers[0]), vec_from_mat_nocopy(gates));
	lstm_update_cell_state(LSTM_LAYER_CELL_STATE(nw->layers[0]), gates, data, p);
	mat_append_row_vec(LSTM_LAYER_PREV_CELL_STATES(nw->layers[0]), LSTM_LAYER_CELL_STATE(nw->layers[0]));
	mat_append_row_vec(LSTM_LAYER_PREV_HIDDEN_STATES(nw->layers[0]), vec_from_mat_nocopy(data));

	mat_copy_to_mat(h_out, data, p,,);

	mat_free(gates);
	for(size_t k = 1; k < nw->layers_cnt - 1; ++k){
		nn_layer* lr = &nw->layers[k];
		mat_append_row_vec(LSTM_LAYER_PREV_INPUTS(*lr), vec_from_mat_nocopy(data));

		mat_copy_to_mat(data, h_in, p, p, k * p); // get hidden state from previous timestamp
		gates = mat_mul(data, lr->weights);
		lstm_add_bias(gates, &nw->layers[0], p);
		mat_apply_activation_func_range(gates, sigmoid, 0, 3 * p);
		mat_apply_activation_func_range(gates, tanh, 3 * p,);
		mat_append_row_vec(LSTM_LAYER_PREV_GATES(nw->layers[k]), vec_from_mat_nocopy(gates));

		lstm_update_cell_state(LSTM_LAYER_CELL_STATE(*lr), gates, data, p);
		mat_append_row_vec(LSTM_LAYER_PREV_CELL_STATES(*lr), LSTM_LAYER_CELL_STATE(*lr));
		mat_append_row_vec(LSTM_LAYER_PREV_HIDDEN_STATES(*lr), vec_from_mat_nocopy(data));

		mat_copy_to_mat(h_out, data, p, p * k,);

		mat_free(gates);
	}

	vec output = vec_from_mat_nocopy(data); vec_resize(output, p);
	return output;
}

typedef struct {
	mat gates;
	vec bias_input;
	vec bias_forget;
	vec bias_output;
	vec bias_cstate;
} lstm_diff;

static void lstm_diff_free(lstm_diff* diff)
{
	mat_free(diff->gates);
	vec_free(diff->bias_input);		
	vec_free(diff->bias_forget);
	vec_free(diff->bias_output);
	vec_free(diff->bias_cstate);
}
static void lstm_diff_apply(lstm_diff* diff, nn_network* nw, nn_layer* lr)
{
	mat_psmul(diff->gates, nw->learn_rate);
	vec_psmul(diff->bias_input, nw->learn_rate);
	vec_psmul(diff->bias_forget, nw->learn_rate);
	vec_psmul(diff->bias_output, nw->learn_rate);
	vec_psmul(diff->bias_cstate, nw->learn_rate);
	mat_psub(lr->weights, diff->gates);
	vec_psub(LSTM_LAYER_BIAS(*lr, input), diff->bias_input);
	vec_psub(LSTM_LAYER_BIAS(*lr, forget), diff->bias_forget);
	vec_psub(LSTM_LAYER_BIAS(*lr, output), diff->bias_output);
	vec_psub(LSTM_LAYER_BIAS(*lr, cstate), diff->bias_cstate);
}

static lstm_diff lstm_get_top_diff(size_t p, nn_layer* lr, size_t t,
								vec top_diff_h, vec top_diff_s, /*in*/
								vec* bottom_diff_h, vec* bottom_diff_s /*out*/)
{
	vec gates = (vec){data: LSTM_LAYER_PREV_GATES(*lr).data + 4*t*p,
						n: 4*p};
	vec cell_state = (vec){data: LSTM_LAYER_PREV_CELL_STATES(*lr).data + t*p,
						n: p};
	vec h = (vec){data: LSTM_LAYER_PREV_HIDDEN_STATES(*lr).data + 2*t*p,
						n: 2*p};
	vec xc = (vec){data: LSTM_LAYER_PREV_INPUTS(*lr).data + 2*t*p,
						n: 2*p};

	vec ds = vec_create(p); vec_copy(ds, gates,, 2*p,); // ds = gate_output
	vec_pemul(ds, top_diff_h);
	vec_padd(ds, top_diff_s); // ds = gate_output * top_diff_h + top_diff_s
	vec _do = vec_dup(cell_state);
	vec_pemul(_do, top_diff_h);// do = cell_state * top_diff_h
	vec di = vec_create(p); vec_copy(di, gates,, 3*p,); vec_pemul(di, ds); // di = new_cstate * ds
	vec dg = vec_create(p); vec_copy(dg, gates,,,); vec_pemul(dg, ds); // dg = gate_input * ds
	vec df;
	if(t == 0) { df = vec_create(p); vec_zero(df); }
	else{ 
		vec prev_cell_state = (vec){data: LSTM_LAYER_PREV_CELL_STATES(*lr).data + (t-1)*p, n: p};
		df = vec_dup(prev_cell_state);
	}
	vec_pemul(df, ds); // df = prev_cell_state * ds
	
	// derivatives with input as denominator
	vec gate_input = (vec){data: gates.data, n: p};
	vec_pemul(di, gate_input);
	vec_assign_expr(di, el * (1 - gate_input.data[i]));

	vec gate_forget = (vec){data: gates.data + p, n: p};
	vec_pemul(df, gate_forget);
	vec_assign_expr(df, el * (1 - gate_forget.data[i]));

	vec gate_output = (vec){data: gates.data + 2*p, n: p};
	vec_pemul(_do, gate_output);
	vec_assign_expr(_do, el * (1 - gate_output.data[i]));

	vec gate_cstate = (vec){data: gates.data + 3*p, n: p};
	vec_assign_expr(dg, el * (1 - gate_cstate.data[i]*gate_cstate.data[i]));

	printf("prev gates %lu: ", t); vec_print(gates);
	printf("prev cell state: "); vec_print(cell_state);
	printf("prev hidden state: "); vec_print(h);
	printf("di: "); vec_print(di);
	printf("df: "); vec_print(df);
	printf("do: "); vec_print(_do);
	printf("dg: "); vec_print(dg);

	lstm_diff diff;
	vec dgates = vec_create(4*p);
	vec_copy(dgates, di,,, p);
	vec_copy(dgates, df, p,, p);
	vec_copy(dgates, _do, 2*p,, p);
	vec_copy(dgates, dg, 3*p,, p);
	diff.gates = vec_outer_product(xc, dgates);
	diff.bias_input = di;
	diff.bias_forget = df;
	diff.bias_output = _do;
	diff.bias_cstate = dg;

	vec dxc = vec_create(2*p); vec_zero(dxc);
	vec dxc_inc = vec_create(2*p);
	rncmat_vec_pdot(lr->weights, di, dxc_inc, 0, p); vec_padd(dxc, dxc_inc);
	rncmat_vec_pdot(lr->weights, df, dxc_inc, p, 2*p); vec_padd(dxc, dxc_inc);
	rncmat_vec_pdot(lr->weights, _do, dxc_inc, 2*p, 3*p); vec_padd(dxc, dxc_inc);
	rncmat_vec_pdot(lr->weights, dg, dxc_inc, 3*p, 4*p); vec_padd(dxc, dxc_inc);

	if(bottom_diff_s->data)
		vec_free(*bottom_diff_s);
	if(bottom_diff_h->data)
		vec_free(*bottom_diff_h);

	vec_pemul(ds, gate_forget);
	*bottom_diff_s = ds;
	vec_resize(dxc, p); *bottom_diff_h = dxc;

	vec_free(dgates);
	vec_free(dxc_inc);
	return diff;
}

void nn_network_lstm_backpropagate(nn_network* nw, vec* expected_arr, size_t expected_cnt)
{
	size_t p = nw->layers[0].weights.n / 4;

	vec top_diff_s = vec_create(p); vec_zero(top_diff_s);
	vec bottom_diff_h = {data: NULL}, bottom_diff_s = {data: NULL};

	for(size_t t = 0; t < expected_cnt; ++t){
		printf("TIMESTAMP: %lu\n", t);
		printf("expected: "); vec_print(expected_arr[t]);
		printf("got: "); vec_print((vec){data: LSTM_LAYER_PREV_HIDDEN_STATES(nw->layers[nw->layers_cnt - 2]).data + t*2*p, n: p});
		vec top_diff_h = loss_gradient(nw->d_lossfunc, expected_arr[t], (vec){data: LSTM_LAYER_PREV_HIDDEN_STATES(nw->layers[nw->layers_cnt - 2]).data + t*2*p, n: p});
		printf("loss gradient (top_diff_h): "); vec_print(top_diff_h);
		printf("LAYER: %lu\n", nw->layers_cnt - 2);
		lstm_diff diff = lstm_get_top_diff(p, &nw->layers[nw->layers_cnt - 2], t, top_diff_h, top_diff_s, &bottom_diff_h, &bottom_diff_s);
		lstm_diff_apply(&diff, nw, &nw->layers[nw->layers_cnt - 2]);
		lstm_diff_free(&diff);
		for(size_t l = nw->layers_cnt - 3; nw->layers_cnt >= 3 /* l still gets checked for 0, at the end of the loop, this condition just skips the loop entirely if network doesn't have enough layers */; --l){
			printf("LAYER: %lu\n", l);
			vec lr_top_diff_h = vec_dup(top_diff_h);
			vec_padd(lr_top_diff_h, bottom_diff_h);

			diff = lstm_get_top_diff(p, &nw->layers[l], t, lr_top_diff_h, bottom_diff_s, &bottom_diff_h, &bottom_diff_s);

			lstm_diff_apply(&diff, nw, &nw->layers[l]);
			vec_free(lr_top_diff_h);
			lstm_diff_free(&diff);
	
			if(l == 0) break;
		}
		vec_free(top_diff_h);
		puts("\n");
	}

	vec_free(top_diff_s);
	vec_free(bottom_diff_h); vec_free(bottom_diff_s);
}

void nn_network_lstm_train(nn_network* nw, nn_training_set* set,
						size_t epochs, double loss_target,
						int flags)
{
	if(!(flags & NNFLAG_DONT_RANDOMIZE_WEIGHTS))
		nn_network_randomize_weights(nw);

	size_t p = nw->layers[0].weights.n / 4;
	size_t d = nw->layers[0].values.n - p;
	size_t dim_out = nw->layers[nw->layers_cnt - 1].values.n;

	int seek_loss = (epochs == 0);
	double total_loss = INFINITY;

	size_t io_size = 1;
	vec* input = malloc(sizeof(vec));
	input[0] = vec_create(d);
	vec* output = malloc(sizeof(vec));
	output[0] = vec_create(dim_out);
	mat h_in = mat_create(nw->layers_cnt - 1, p);
	mat h_out = mat_create(nw->layers_cnt - 1, p);

	for(; epochs > 0 || seek_loss; --epochs){
		set->shuffle(set);
		if(flags & NNFLAG_DEBUG_LOSS)
			printf("loss: %g ", total_loss);
		total_loss = 0;

		for(size_t i = 0; i < set->size; ++i){
			input[0].data[0] = 0;
			size_t data_idx = 0;
			while(1){
				if(data_idx >= io_size){
					size_t diff = data_idx - io_size;
					input = realloc(input, sizeof(vec) * (data_idx + 1));
					output = realloc(output, sizeof(vec) * (data_idx + 1));
					for(size_t j = 0; j <= diff; ++j){
						input[io_size + j] = vec_create(d);
						output[io_size + j] = vec_create(dim_out);
					}
					io_size = data_idx + 1;
				}

				set->get_input(set, input[data_idx], i, data_idx);
				if(isinf(input[data_idx].data[0])) // end of sequence
					break; // now data_idx == sequence length
				set->get_output(set, output[data_idx], i, data_idx);
				
				input[data_idx].n = d;
				output[data_idx].n = dim_out;

				++data_idx;
			}

			mat_zero(h_in);
			nn_network_lstm_reset_state(nw);

			for(size_t j = 0; j < data_idx; ++j){
				vec predicted = nn_network_lstm_feedforward(nw, input[j], h_in, h_out);
				double loss = nw->lossfunc(output[j], predicted, NULL);
				total_loss += loss / data_idx;

				mat tmp = h_in;
				h_in = h_out;
				h_out = tmp;
				vec_free(predicted);
			}
			nn_network_lstm_backpropagate(nw, output, data_idx);
		}
		total_loss /= set->size;
		if(flags & NNFLAG_DEBUG_LOSS)
			printf("--> %g\n", total_loss);
		if(seek_loss && total_loss <= loss_target)
			break;
	}

	mat_free(h_in);
	mat_free(h_out);
	for(size_t i = 0; i < io_size; ++i){
		vec_free(input[i]);
		vec_free(output[i]);
	}
	free(input);
	free(output);
}

/***** Saving functions *****/

void nn_network_save_weights(FILE* fd, nn_network* nw)
{
	fwrite(&nw->layers_cnt, sizeof(nw->layers_cnt), 1, fd);
	for(size_t i = 0; i < nw->layers_cnt - 1; ++i){
		nn_layer* lr = &nw->layers[i];
		fwrite(&lr->weights.m, sizeof(lr->weights.m), 1, fd);
		fwrite(&lr->weights.n, sizeof(lr->weights.n), 1, fd);
		fwrite(lr->weights.data, sizeof(double), lr->weights.m * lr->weights.n, fd);
	}
}

int nn_network_load_weights(FILE* fd, nn_network* nw)
{
	size_t layers_cnt;
	fread(&layers_cnt, sizeof(layers_cnt), 1, fd);
	if(layers_cnt != nw->layers_cnt)
		return NN_ERROR_LAYERS_CNT_NOT_MATCH;
	for(size_t i = 0; i < layers_cnt - 1; ++i){
		nn_layer* lr = &nw->layers[i];
		size_t m, n;
		fread(&m, sizeof(m), 1, fd);
		fread(&n, sizeof(n), 1, fd);
		if(m != lr->weights.m || n != lr->weights.n)
			return NN_ERROR_WEIGHTS_DIM_NOT_MATCH;
		fread(lr->weights.data, sizeof(double), m * n, fd);
	}
	return 0;
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
