#ifndef NETWORK_LSTM_H
#define NETWORK_LSTM_H

#include "network.h"

void nn_network_lstm_init(nn_network* nw);
void nn_network_lstm_reset_state(nn_network* nw);

vec nn_network_lstm_feedforward(nn_network* nw, vec input, mat h_in, mat h_out);
void nn_network_lstm_backpropagate(nn_network* nw, vec* expected_arr, size_t expected_cnt);

#define nn_network_lstm_train(...) (__nn_network_lstm_train(NN_TRAINING_PARAMS_DEFAULT(nn_train_params, ##__VA_ARGS__)))
void __nn_network_lstm_train(nn_train_params params);

#endif
