#include "training_set.h"

/* Real value vectors */

static void __nn_get_input_double(nn_training_set* set, vec v, size_t i)
{
	memcpy(v.data, ((double*)set->data_in) + i * set->in_size, sizeof(double) * set->in_size);
}
static void __nn_get_output_double(nn_training_set* set, vec v, size_t i)
{
	memcpy(v.data, ((double*)set->data_out) + i * set->out_size, sizeof(double) * set->out_size);
}
static void __nn_set_input_double(nn_training_set* set, vec v, size_t i)
{
	memcpy(((double*)set->data_in) + i * set->in_size, v.data, sizeof(double) * set->in_size);
}
static void __nn_set_output_double(nn_training_set* set, vec v, size_t i)
{
	memcpy(((double*)set->data_out) + i * set->out_size, v.data, sizeof(double) * set->out_size);
}

static void __nn_shuffle_double(nn_training_set* set)
{
	double* tmp = malloc(sizeof(double) * (set->in_size > set->out_size ? set->in_size : set->out_size));
	for(size_t i = 0; i < set->size / 2; ++i){
		size_t a = rand() % set->size,
			   b = rand() % set->size;

		memcpy(tmp, (double*)set->data_in + a * set->in_size, sizeof(double) * set->in_size);
		memcpy((double*)set->data_in + a * set->in_size, (double*)set->data_in + b * set->in_size, sizeof(double) * set->in_size);
		memcpy((double*)set->data_in + b * set->in_size, tmp, sizeof(double) * set->in_size);
		memcpy(tmp, (double*)set->data_out + a * set->out_size, sizeof(double) * set->out_size);
		memcpy((double*)set->data_out + a * set->out_size, (double*)set->data_out + b * set->out_size, sizeof(double) * set->out_size);
		memcpy((double*)set->data_out + b * set->out_size, tmp, sizeof(double) * set->out_size);
	}
	free(tmp);
}

/* One-hot encoded vectors */

static void __nn_get_input_onehot(nn_training_set* set, vec v, size_t i)
{
	memset(v.data, 0, v.n * sizeof(double));
	for(size_t j = 0; j < set->in_size; ++j)
		v.data[((size_t*)set->data_in)[i * set->in_size + j]] = 1;
}
static void __nn_get_output_onehot(nn_training_set* set, vec v, size_t i)
{
	memset(v.data, 0, v.n * sizeof(double));
	for(size_t j = 0; j < set->out_size; ++j)
		v.data[((size_t*)set->data_out)[i * set->out_size + j]] = 1;
}
static void __nn_set_input_onehot(nn_training_set* set, vec v, size_t i)
{
	size_t onehot_gap = v.n / set->in_size;
	for(size_t j = 0; j < set->in_size; ++j)
		for(size_t k = onehot_gap * j; k < onehot_gap * (j + 1); ++k)
			if(v.data[k] >= 0.5){
				((size_t*)set->data_in)[i * set->in_size + j] = k;
				break;
			}
}
static void __nn_set_output_onehot(nn_training_set* set, vec v, size_t i)
{
	size_t onehot_gap = v.n / set->out_size;
	for(size_t j = 0; j < set->out_size; ++j)
		for(size_t k = onehot_gap * j; k < onehot_gap * (j + 1); ++k)
			if(v.data[k] >= 0.5){
				((size_t*)set->data_out)[i * set->out_size + j] = k;
				break;
			}
}

static void __nn_shuffle_onehot(nn_training_set* set)
{
	size_t* tmp = malloc(sizeof(size_t) * (set->in_size > set->out_size ? set->in_size : set->out_size));
	for(size_t i = 0; i < set->size / 2; ++i){
		size_t a = rand() % set->size,
			   b = rand() % set->size;

		memcpy(tmp, (size_t*)set->data_in + a * set->in_size, sizeof(size_t) * set->in_size);
		memcpy((size_t*)set->data_in + a * set->in_size, (size_t*)set->data_in + b * set->in_size, sizeof(size_t) * set->in_size);
		memcpy((size_t*)set->data_in + b * set->in_size, tmp, sizeof(size_t) * set->in_size);
		memcpy(tmp, (size_t*)set->data_out + a * set->out_size, sizeof(size_t) * set->out_size);
		memcpy((size_t*)set->data_out + a * set->out_size, (size_t*)set->data_out + b * set->out_size, sizeof(size_t) * set->out_size);
		memcpy((size_t*)set->data_out + b * set->out_size, tmp, sizeof(size_t) * set->out_size);
	}
	free(tmp);
}


nn_training_set nn_training_set_create(int data_type, size_t size, size_t in_size, size_t out_size)
{
	nn_training_set set;
	set.data_type = data_type;
	set.size = size;
	set.in_size = in_size; set.out_size = out_size;

	switch(data_type){
		case NN_TRAINING_SET_TYPE_DOUBLE:
			set.data_in = malloc(sizeof(double) * size * in_size);
			set.data_out = malloc(sizeof(double) * size * out_size);
			set.get_input = __nn_get_input_double;
			set.get_output = __nn_get_output_double;
			set.set_input = __nn_set_input_double;
			set.set_output = __nn_set_output_double;
			set.shuffle = __nn_shuffle_double;
			break;
		case NN_TRAINING_SET_TYPE_ONEHOT:
			set.data_in = malloc(sizeof(size_t) * size * in_size);
			set.data_out = malloc(sizeof(size_t) * size * out_size);
			set.get_input = __nn_get_input_onehot;
			set.get_output = __nn_get_output_onehot;
			set.set_input = __nn_set_input_onehot;
			set.set_output = __nn_set_output_onehot;
			set.shuffle = __nn_shuffle_onehot;
			break;
	}
	return set;
}
void nn_training_set_free(nn_training_set* set)
{
	free(set->data_in);
	free(set->data_out);
}

void nn_training_set_expand(nn_training_set* set, size_t how_much)
{
	set->size += how_much;
	switch(set->data_type){
		case NN_TRAINING_SET_TYPE_DOUBLE:
			set->data_in = realloc(set->data_in, sizeof(double) * set->size * set->in_size);
			set->data_out = realloc(set->data_out, sizeof(double) * set->size * set->out_size);
			break;
		case NN_TRAINING_SET_TYPE_ONEHOT:
			set->data_in = realloc(set->data_in, sizeof(size_t) * set->size * set->in_size);
			set->data_out = realloc(set->data_out, sizeof(size_t) * set->size * set->out_size);
			break;
	}
}
void nn_training_set_add(nn_training_set* set, vec v_in, vec v_out)
{
	nn_training_set_expand(set, 1);
	set->set_input(set, v_in, set->size - 1);
	set->set_output(set, v_out, set->size - 1);
}

void nn_training_set_save(FILE* fd, nn_training_set* set)
{
	fwrite(&set->data_type, sizeof(set->data_type), 1, fd);
	fwrite(&set->size, sizeof(set->size), 1, fd);
	fwrite(&set->in_size, sizeof(set->in_size), 1, fd);
	fwrite(&set->out_size, sizeof(set->out_size), 1, fd);
	switch(set->data_type){
		case NN_TRAINING_SET_TYPE_DOUBLE:
			fwrite(set->data_in, sizeof(double), set->size * set->in_size, fd);
			fwrite(set->data_out, sizeof(double), set->size * set->out_size, fd);
			break;
		case NN_TRAINING_SET_TYPE_ONEHOT:
			fwrite(set->data_in, sizeof(size_t), set->size * set->in_size, fd);
			fwrite(set->data_out, sizeof(size_t), set->size * set->out_size, fd);
			break;
	}
}
nn_training_set nn_training_set_load(FILE* fd)
{
	int data_type;
	size_t size, in_size, out_size;
	fread(&data_type, sizeof(data_type), 1, fd);
	fread(&size, sizeof(size), 1, fd);
	fread(&in_size, sizeof(in_size), 1, fd);
	fread(&out_size, sizeof(out_size), 1, fd);
	nn_training_set set = nn_training_set_create(data_type, size, in_size, out_size);
	switch(data_type){
		case NN_TRAINING_SET_TYPE_DOUBLE:
			fread(set.data_in, sizeof(double), size * in_size, fd);
			fread(set.data_out, sizeof(double), size * out_size, fd);
			break;
		case NN_TRAINING_SET_TYPE_ONEHOT:
			fread(set.data_in, sizeof(size_t), size * in_size, fd);
			fread(set.data_out, sizeof(size_t), size * out_size, fd);
			break;
	}
	return set;
}
