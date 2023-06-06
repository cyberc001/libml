#include "vec.h"
#include <stdio.h>

vec vec_create_random(size_t n)
{
	vec _out = vec_create(n);
	for(size_t i = 0; i < n; ++i)
		_out.data[i] = (double)rand() / RAND_MAX;
	return _out;
}
vec vec_create_zero(size_t n)
{
	vec v = vec_create(n);
	vec_zero(v);
	return v;
}

void vec_print(vec v)
{
	size_t sz = v.n;
	for(size_t i = 0; i < sz; ++i)
		printf("%g ", v.data[i]);
	puts("");
}
