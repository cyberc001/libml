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

/***** Math functions *****/

vec nn_network_loss_grad(loss_d_func d_lossfunc, vec expected, vec predicted)
{
	vec grad = vec_create(predicted.n);
	for(size_t i = 0; i < predicted.n; ++i)
		grad.data[i] = d_lossfunc(expected, predicted, i, NULL);
	return grad;
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
