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
#define vec_resize(v, nn) {(v).n = (nn); (v).data = realloc((v).data, sizeof(double) * (nn));}

vec vec_create_random(size_t n);

// TODO unite these 2 macros
#define vec_from_mat_row(_m) ((vec){(_m).n, memcpy(malloc((_m).n * sizeof(double)), (_m).data, (_m).n * sizeof(double))})
#define vec_from_mat_column(_m) ((vec){(_m).m, memcpy(malloc((_m).m * sizeof(double)), (_m).data, (_m).m * sizeof(double))})
#define vec_from_mat_nocopy(_m) ((vec){(_m).n * (_m).m, (_m).data})

#define vec_copy(to, from, begin_to, begin_from, amt) {memcpy((to).data + ((#begin_to)[0] == '\0' ? 0 : begin_to +0), (from).data + ((#begin_from)[0] == '\0' ? 0 : begin_from +0), ((#amt)[0] == '\0' ? (to).n : amt +0) * sizeof(double));}
#define vec_dup(v) ((vec){(v).n, memcpy(malloc((v).n * sizeof(double)), (v).data, (v).n * sizeof(double))})
#define vec_dup_resize(v, nn) ((vec){(nn), memcpy(malloc((nn) * sizeof(double)), (v).data, (v).n * sizeof(double))})

#define vec_zero(v) (memset((v).data, 0, sizeof(double) * (v).n))

#define vec_assign_expr(v, expr) {for(size_t i = 0; i < (v).n; ++i){ double el = (v).data[i]; (v).data[i] = (expr); }}

/* s - Scalar
 * p - in Place (v1 = v1 ...)
 * e - Element-wise
 */

#define vec_padd(v1, v2) {for(size_t i = 0; i < (v1).n; ++i) (v1).data[i] += v2.data[i];}
#define vec_psub(v1, v2) {for(size_t i = 0; i < (v1).n; ++i) (v1).data[i] -= v2.data[i];}

#define vec_psmul(v, s) {for(size_t i = 0; i < (v).n; ++i) (v).data[i] *= (s);}
#define vec_pemul(v1, v2) {for(size_t i = 0; i < (v1).n; ++i) (v1).data[i] *= (v2).data[i]; }

void vec_print(vec v);

#endif
