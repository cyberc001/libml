#ifndef VEC_H
#define VEC_H

#include <stdlib.h>
#include <string.h>

typedef struct {
	size_t n;
	double* data;
} vec;

#define vec_create(n) ((vec){(n), malloc(sizeof(double) * (n))})
vec vec_create_random(size_t n);
vec vec_create_zero(size_t n);

#define vec_free(v) (free((v).data))
#define vec_resize(v, nn) {(v).n = (nn); (v).data = realloc((v).data, sizeof(double) * (nn));}
#define vec_shift_resize(v, off) { memmove((v).data, (v).data + (off), sizeof(double) * ((v).n - (off))); (v).n -= (off); (v).data = realloc((v).data, sizeof(double) * (v).n); }
#define vec_zero(v) (memset((v).data, 0, sizeof(double) * (v).n))

#define vec_from_mat(_m) ((vec){(_m).n * (_m).m, memcpy(malloc((_m).n * (_m).m * sizeof(double)), (_m).data, (_m).n * (_m).m * sizeof(double))})
#define vec_from_mat_nocopy(_m) ((vec){(_m).n * (_m).m, (_m).data})

#define vec_copy(to, from, begin_to, begin_from, amt) {memcpy((to).data + ((#begin_to)[0] == '\0' ? 0 : begin_to +0), (from).data + ((#begin_from)[0] == '\0' ? 0 : begin_from +0), ((#amt)[0] == '\0' ? (to).n : amt +0) * sizeof(double));}
#define vec_dup(v) ((vec){(v).n, memcpy(malloc((v).n * sizeof(double)), (v).data, (v).n * sizeof(double))})
#define vec_dup_resize(v, nn) ((vec){(nn), memcpy(malloc((nn) * sizeof(double)), (v).data, (v).n * sizeof(double))})

#define vec_assign_expr(v, expr) {for(size_t i = 0; i < (v).n; ++i){ double el = (v).data[i]; (v).data[i] = (expr); }}

/* s - Scalar
 * p - in Place (v1 = v1 ...)
 * e - Element-wise
 */

void vec_padd(vec v1, vec v2);
void vec_psub(vec v1, vec v2);

void vec_psmul(vec v, double s);
void vec_pemul(vec v1, vec v2);

void vec_print(vec v);

#endif
