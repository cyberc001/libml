#include "noise_distr.h"
#include <stdlib.h>
#include <assert.h>
#include <math.h>

noise_distr noise_distr_create(size_t freq_ln, size_t* freq, double distr_pow)
{
	noise_distr distr = (noise_distr){
		freq_ln, malloc(sizeof(double) * freq_ln),
				 malloc(sizeof(double) * freq_ln)
	};
	size_t sum = 0;
	for(size_t i = 0; i < freq_ln; ++i)
		sum += freq[i];
	double running_chance = 0;
	for(size_t i = 0; i < freq_ln; ++i){
		distr.chance[i] = pow(freq[i], distr_pow) / sum;
		running_chance += distr.chance[i];
		distr.chance_sum[i] = running_chance;
	}
	return distr;
}

void noise_distr_free(noise_distr* distr)
{
	free(distr->chance);
	free(distr->chance_sum);
}

size_t noise_distr_pick(noise_distr* distr, size_t avoid_ln, size_t* avoid)
{
	assert(avoid_ln < distr->ln);
	loop:
		double val = rand() / (double)RAND_MAX;

		size_t l = 0, r = distr->ln;
		while(l < r){
			size_t m = (l + r) / 2;
			if(val > distr->chance_sum[m]) l = m;
			else if(val < distr->chance_sum[m]) r = m;
			else break;
			if(l + 1 >= r) break;
		}
		size_t res = (l + r) / 2;
		for(size_t i = 0; i < avoid_ln; ++i)
			if(avoid[i] == res)
				goto loop;
	return res;
}
