#include "vec.h"
#include <stdio.h>

void vec_print(vec v)
{
	size_t sz = v.n;
	for(size_t i = 0; i < sz; ++i)
		printf("%g ", v.data[i]);
	puts("");
}
