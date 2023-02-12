#ifndef VEC_H
#define VEC_H

#include <stdlib.h>
#include <string.h>

typedef struct {
	size_t n;
	double* data;
} vec;

#define vec_create(n) ((vec){(n), malloc(sizeof(double) * (n))})
#define vec_free(v) (free((v).data))

#define vec_from_mat_row(_m) ((vec){(_m).n, memcpy(malloc((_m).n * sizeof(double)), (_m).data, (_m).n * sizeof(double))})
#define vec_from_mat_column(_m) ((vec){(_m).m, memcpy(malloc((_m).m * sizeof(double)), (_m).data, (_m).m * sizeof(double))})

/* Note: vectors should have the same size. */
#define vec_copy(to, from) {memcpy((to).data, (from).data, (to).n * sizeof(double));}
#define vec_dup(v) ((vec){(v).n, memcpy(malloc((v).n * sizeof(double)), (v).data, (v).n * sizeof(double))})

/* s - Scalar
 * p - in Place (v1 = v1 ...)
 * e - Element-wise
 */

#define vec_psmul(v, s) {for(size_t i = 0; i < (v).n; ++i) (v).data[i] *= (s);}
#define vec_pemul(v1, v2) {for(size_t i = 0; i < (v1).n; ++i) (v1).data[i] *= (v2).data[i]; }

void vec_print(vec v);

#endif
