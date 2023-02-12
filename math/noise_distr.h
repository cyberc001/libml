#ifndef NOISE_DISTR_H
#define NOISE_DISTR_H

#include <stddef.h>

typedef struct {
	size_t ln;
	double* chance;
	double* chance_sum;
} noise_distr;

noise_distr noise_distr_create(size_t freq_ln, size_t* freq, double distr_pow);
size_t noise_distr_pick(noise_distr* distr, size_t avoid_ln, size_t* avoid);

#endif
