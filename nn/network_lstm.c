#include "network_lstm.h"
#include <math.h>

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

	lstm_diff diff;
} lstm_ldata;

#define LSTM_LAYER_BIAS(lr, _type) (((lstm_ldata*)(lr).data)->bias_ ## _type)
#define LSTM_LAYER_CELL_STATE(lr) (((lstm_ldata*)(lr).data)->cell_state)

#define LSTM_LAYER_PREV_GATES(lr) (((lstm_ldata*)(lr).data)->prev_gates)
#define LSTM_LAYER_PREV_HIDDEN_STATES(lr) (((lstm_ldata*)(lr).data)->prev_hidden_states)
#define LSTM_LAYER_PREV_INPUTS(lr) (((lstm_ldata*)(lr).data)->prev_inputs)
#define LSTM_LAYER_PREV_CELL_STATES(lr) (((lstm_ldata*)(lr).data)->prev_cell_states)

#define LSTM_LAYER_DIFF(lr) (((lstm_ldata*)(lr).data)->diff)

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

	lstm_diff_free(&data->diff);
}

static void lstm_add_bias(mat gates, nn_layer* lr, size_t p)
{
	rnrmat_padd1(gates, mat_from_vec_column_nocopy(LSTM_LAYER_BIAS(*lr, input)), 0, p);
	rnrmat_padd1(gates, mat_from_vec_column_nocopy(LSTM_LAYER_BIAS(*lr, forget)), p, 2 * p);
	rnrmat_padd1(gates, mat_from_vec_column_nocopy(LSTM_LAYER_BIAS(*lr, output)), 2 * p, 3 * p);
	rnrmat_padd1(gates, mat_from_vec_column_nocopy(LSTM_LAYER_BIAS(*lr, cstate)), 3 * p, 4 * p);
}
static void lstm_update_cell_state(vec cell_state, mat gates, mat data, size_t p)
{
	// intermediary values are kept in gates matrix for efficiency
	rnrmat_pemul1(gates, mat_from_vec_column_nocopy(cell_state), p, 2 * p); // forget gate * previous cell-state
	rnrmat_pemul12(gates, gates, 0, p, 3 * p); // input gate * new cell-state
	// copy all calculations above into cell-state
	mat_copy_to_vec_row(cell_state, gates, p);
	rnrmat_padd2(mat_from_vec_column_nocopy(cell_state), gates, 0, p);
	// leak cell-state into hidden state
	for(size_t i = 0; i < p; ++i)
		data.data[i] = gates.data[2 * p + i] * /*tanh(*/cell_state.data[i];
}

void nn_network_lstm_init(nn_network* nw)
{
	size_t p = nw->layers[0].weights.m / 4;
	size_t d = nw->layers[0].weights.n - p;
	for(size_t k = 0; k < nw->layers_cnt - 1; ++k){
		nw->layers[k].data = malloc(sizeof(lstm_ldata));
		nw->layers[k].data_free = nn_network_lstm_free_data;
		LSTM_LAYER_BIAS(nw->layers[k], input) = vec_create_random(p);
		LSTM_LAYER_BIAS(nw->layers[k], forget) = vec_create_random(p);
		LSTM_LAYER_BIAS(nw->layers[k], output) = vec_create_random(p);
		LSTM_LAYER_BIAS(nw->layers[k], cstate) = vec_create_random(p);
		LSTM_LAYER_CELL_STATE(nw->layers[k]) = vec_create_zero(p);

		LSTM_LAYER_PREV_GATES(nw->layers[k]) = mat_create(0, 4 * p);
		LSTM_LAYER_PREV_HIDDEN_STATES(nw->layers[k]) = mat_create(0, 2 * p);
		LSTM_LAYER_PREV_INPUTS(nw->layers[k]) = mat_create(0, k == 0 ? d+p : 2*p);
		LSTM_LAYER_PREV_CELL_STATES(nw->layers[k]) = mat_create(0, p);

		LSTM_LAYER_DIFF(nw->layers[k]).gates = mat_create_zero(4*p, k == 0 ? d+p : 2*p);
		LSTM_LAYER_DIFF(nw->layers[k]).bias_input = vec_create_zero(p);
		LSTM_LAYER_DIFF(nw->layers[k]).bias_forget = vec_create_zero(p);
		LSTM_LAYER_DIFF(nw->layers[k]).bias_output = vec_create_zero(p);
		LSTM_LAYER_DIFF(nw->layers[k]).bias_cstate = vec_create_zero(p);
	}
}
void nn_network_lstm_reset_state(nn_network* nw)
{
	size_t p = nw->layers[0].weights.m / 4;
	size_t d = nw->layers[0].weights.n - p;
	for(size_t l = 0; l < nw->layers_cnt - 1; ++l){
		vec_zero(LSTM_LAYER_CELL_STATE(nw->layers[l]));
		mat_free(LSTM_LAYER_PREV_GATES(nw->layers[l]));
		mat_free(LSTM_LAYER_PREV_HIDDEN_STATES(nw->layers[l]));
		mat_free(LSTM_LAYER_PREV_INPUTS(nw->layers[l]));
		mat_free(LSTM_LAYER_PREV_CELL_STATES(nw->layers[l]));

		LSTM_LAYER_PREV_GATES(nw->layers[l]) = mat_create(0, 4 * p);
		LSTM_LAYER_PREV_HIDDEN_STATES(nw->layers[l]) = mat_create(0, 2 * p);
		LSTM_LAYER_PREV_INPUTS(nw->layers[l]) = mat_create(0, l == 0 ? d+p : 2*p);
		LSTM_LAYER_PREV_CELL_STATES(nw->layers[l]) = mat_create(0, p);

		mat_zero(LSTM_LAYER_DIFF(nw->layers[l]).gates);
		vec_zero(LSTM_LAYER_DIFF(nw->layers[l]).bias_input);
		vec_zero(LSTM_LAYER_DIFF(nw->layers[l]).bias_forget);
		vec_zero(LSTM_LAYER_DIFF(nw->layers[l]).bias_output);
		vec_zero(LSTM_LAYER_DIFF(nw->layers[l]).bias_cstate);
	}
}

vec nn_network_lstm_feedforward(nn_network* nw, vec input, mat h_in, mat h_out)
{
	size_t p = nw->layers[0].weights.m / 4;
	size_t d = nw->layers[0].weights.n - p;

	// feedforward for (input, hidden state from previous timestamp) pair
	mat data = mat_create(d + p, 1); // hidden state vector
	vec_row_copy_to_mat(data, input,); // first d elements are from input data
	mat_copy_to_mat(data, h_in, p, d,); // second p elements are from hidden state vector from previous timestamp
	mat_append_row_vec(LSTM_LAYER_PREV_INPUTS(nw->layers[0]), vec_from_mat_nocopy(data));
	mat gates = mat_mul(nw->layers[0].weights, data);
	mat_free(data);

	lstm_add_bias(gates, &nw->layers[0], p);
	mat_apply_activation_func_range(gates, sigmoid, 0, 3 * p);
	mat_apply_activation_func_range(gates, tanh, 3 * p,);
	
	data = mat_create_zero(2 * p, 1); // change hidden state vector length for hidden-to-hidden updates
	mat_copy_to_mat(data, h_in, p, p,); // get hidden state from previous timestamp

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
		gates = mat_mul(lr->weights, data);
		lstm_add_bias(gates, &nw->layers[k], p);
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

static void lstm_get_top_diff(size_t d, size_t p, nn_layer* lr, int is_input_lr, size_t t, mat t_weights,
								vec top_diff_h, vec top_diff_s, /*in*/
								vec* bottom_diff_h, vec* bottom_diff_s /*out*/)
{
	vec gates = (vec){data: LSTM_LAYER_PREV_GATES(*lr).data + 4*t*p,
						n: 4*p};
	vec cell_state = (vec){data: LSTM_LAYER_PREV_CELL_STATES(*lr).data + t*p,
						n: p};
	vec xc = (vec){data: LSTM_LAYER_PREV_INPUTS(*lr).data + (is_input_lr ? t*(d+p) : 2*t*p),
						n: (is_input_lr ? d + p : 2*p)};

	vec ds = vec_create(p); vec_copy(ds, gates,, 2*p,); // ds = gate_output
	vec_pemul(ds, top_diff_h);
	vec_padd(ds, top_diff_s); // ds = gate_output * top_diff_h + top_diff_s	
	vec _do = vec_dup(cell_state);
	vec_pemul(_do, top_diff_h);// do = cell_state * top_diff_h
	vec di = vec_create(p); vec_copy(di, gates,, 3*p,); vec_pemul(di, ds); // di = new_cstate * ds
	vec dg = vec_create(p); vec_copy(dg, gates,,,); vec_pemul(dg, ds); // dg = gate_input * ds
	vec df;
	if(t == 0) df = vec_create_zero(p); 
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

	lstm_diff diff = LSTM_LAYER_DIFF(*lr);
	vec dgates = vec_create(4*p);
	vec_copy(dgates, di,,, p);
	vec_copy(dgates, df, p,, p);
	vec_copy(dgates, _do, 2*p,, p);
	vec_copy(dgates, dg, 3*p,, p);
	mat gates_diff = vec_outer_product(dgates, xc);
	mat_padd(diff.gates, gates_diff);
	mat_free(gates_diff);
	vec_padd(diff.bias_input, di);
	vec_padd(diff.bias_forget, df);
	vec_padd(diff.bias_output, _do);
	vec_padd(diff.bias_cstate, dg);

	vec dxc = vec_create_zero(is_input_lr ? d + p : 2*p);
	vec dxc_inc = vec_create(is_input_lr ? d + p : 2*p);
	rncmat_vec_pdot(t_weights, di, dxc_inc, 0, p); vec_padd(dxc, dxc_inc);
	rncmat_vec_pdot(t_weights, df, dxc_inc, p, 2*p); vec_padd(dxc, dxc_inc);
	rncmat_vec_pdot(t_weights, _do, dxc_inc, 2*p, 3*p); vec_padd(dxc, dxc_inc);
	rncmat_vec_pdot(t_weights, dg, dxc_inc, 3*p, 4*p); vec_padd(dxc, dxc_inc);

	if(bottom_diff_s->data)
		vec_free(*bottom_diff_s);
	if(bottom_diff_h->data)
		vec_free(*bottom_diff_h);

	vec_pemul(ds, gate_forget);
	*bottom_diff_s = ds;
	vec_shift_resize(dxc, d); *bottom_diff_h = dxc;

	vec_free(dgates);
	vec_free(dxc_inc);
	vec_free(di);
	vec_free(df);
	vec_free(_do);
	vec_free(dg);
}

void nn_network_lstm_backpropagate(nn_network* nw, vec* expected_arr, size_t expected_cnt)
{
	size_t p = nw->layers[0].weights.m / 4;
	size_t d = nw->layers[0].weights.n - p;

	vec top_diff_s = vec_create_zero(p);
	int first_top_diff_s = 1; // logic for freeing memory taken by the first top_diff_s vector which is just zeros
	vec bottom_diff_h = {data: NULL}, bottom_diff_s = {data: NULL};

	mat* t_weights = malloc(sizeof(mat) * nw->layers_cnt - 1);
	for(size_t i = 0; i < nw->layers_cnt - 1; ++i)
		t_weights[i] = mat_tran(nw->layers[i].weights);

	for(size_t t = expected_cnt - 1;; --t){
		vec top_diff_h = nn_network_loss_grad(nw->d_lossfunc, expected_arr[t], (vec){data: LSTM_LAYER_PREV_HIDDEN_STATES(nw->layers[nw->layers_cnt - 2]).data + t*2*p, n: p});
		if(bottom_diff_h.data)
			vec_padd(top_diff_h, bottom_diff_h);
		lstm_get_top_diff(d, p, &nw->layers[nw->layers_cnt - 2], nw->layers_cnt == 2, t, t_weights[nw->layers_cnt - 2], top_diff_h, top_diff_s, &bottom_diff_h, &bottom_diff_s);
		if(first_top_diff_s){
			first_top_diff_s = 0;
			vec_free(top_diff_s);
		}
		top_diff_s = bottom_diff_s;
		for(size_t l = nw->layers_cnt - 3; nw->layers_cnt >= 3 /* l still gets checked for 0, at the end of the loop, this condition just skips the loop entirely if network doesn't have enough layers */; --l){
			vec lr_top_diff_h = vec_dup(top_diff_h);
			vec_padd(lr_top_diff_h, bottom_diff_h);

			lstm_get_top_diff(d, p, &nw->layers[l], l == 0, t, t_weights[l], lr_top_diff_h, bottom_diff_s, &bottom_diff_h, &bottom_diff_s);
			top_diff_s = bottom_diff_s;

			vec_free(lr_top_diff_h);
	
			if(l == 0) break;
		}
		vec_free(top_diff_h);
		if(t == 0) break;
	}

	for(size_t i = 0; i < nw->layers_cnt - 1; ++i)
		mat_free(t_weights[i]);
	free(t_weights);

	lstm_diff_apply(&LSTM_LAYER_DIFF(nw->layers[nw->layers_cnt - 2]), nw, &nw->layers[nw->layers_cnt - 2]);
	for(size_t l = nw->layers_cnt - 3; nw->layers_cnt >= 3; --l){
		lstm_diff_apply(&LSTM_LAYER_DIFF(nw->layers[l]), nw, &nw->layers[l]);
		if(l == 0) break;
	}

	vec_free(top_diff_s);
	vec_free(bottom_diff_h);
}

void __nn_network_lstm_train(nn_train_params params)
{
	nn_network* nw = params.nw;
	nn_training_set* set = params.set;
	int flags = params.flags;

	size_t p = nw->layers[0].weights.m / 4;
	size_t d = nw->layers[0].weights.n - p;

	size_t io_size = 1;
	vec* input = malloc(sizeof(vec));
	input[0] = vec_create(d);
	vec* output = malloc(sizeof(vec));
	output[0] = vec_create(p);
	mat h_in = mat_create(nw->layers_cnt - 1, p);
	mat h_out = mat_create(nw->layers_cnt - 1, p);

	double total_loss = INFINITY;
	for(size_t epoch_count = 0; !(params.epoch_limit > 0 && epoch_count >= params.epoch_limit) && !(params.target_loss >= 0 && total_loss < params.target_loss); ++epoch_count){
		set->shuffle(set);
		total_loss = 0;

		for(size_t i = 0; i < set->size; ++i){
			size_t data_idx = 0;
			while(1){
				if(data_idx >= io_size){
					size_t diff = data_idx - io_size;
					input = realloc(input, sizeof(vec) * (data_idx + 1));
					output = realloc(output, sizeof(vec) * (data_idx + 1));
					for(size_t j = 0; j <= diff; ++j){
						input[io_size + j] = vec_create(d);
						output[io_size + j] = vec_create(p);
					}
					io_size = data_idx + 1;
				}

				set->get_input(set, input[data_idx], i, data_idx);
				if(isinf(input[data_idx].data[0])) // end of sequence
					break; // now data_idx == sequence length
				set->get_output(set, output[data_idx], i, data_idx);

				++data_idx;
			}

			mat_zero(h_in);
			nn_network_lstm_reset_state(nw);

			for(size_t j = 0; j < data_idx; ++j){
				vec predicted = nn_network_lstm_feedforward(nw, input[j], h_in, h_out);
				double loss = nw->lossfunc(output[j], predicted, NULL);
				total_loss += loss / data_idx;
				/*printf("predicted: "); vec_print(predicted);
				printf("expected: "); vec_print(output[j]);
				printf("loss: %lg\n", loss);*/

				mat tmp = h_in;
				h_in = h_out;
				h_out = tmp;
				vec_free(predicted);
			}
			nn_network_lstm_backpropagate(nw, output, data_idx);
		}

		total_loss /= set->size;
		if(flags & NN_TRAIN_FLAGS_DEBUG)
			fprintf(stderr, "epoch: %lu loss: %lg\n", epoch_count, total_loss);
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
