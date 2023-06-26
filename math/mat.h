#ifndef MAT_H
#define MAT_H

#include <stddef.h>
#include "vec.h"

typedef struct {
	size_t m, n;
	double* data;
} mat;

#define mat_create(m, n) ((mat){(m), (n), malloc(sizeof(double) * (m) * (n))})
mat mat_create_zero(size_t m, size_t n);
#define mat_free(m) (free((m).data))

#define mat_resize(_m, _mm, _mn) ((mat){(_mm), (_mn), realloc((_m).data, sizeof(double) * (_mm) * (_mn))})
#define mat_append_row_vec(_m, v) {(_m) = mat_resize((_m), (_m).m + 1, (_m).n); vec_row_copy_to_mat((_m), v, ((_m).m - 1) * (_m).n); }

#define mat_from_vec_row(v) ((mat){1, (v).n, memcpy(malloc((v).n * sizeof(double)), (v).data, (v).n * sizeof(double))})
#define mat_from_vec_column(v) ((mat){(v).n, 1, memcpy(malloc((v).n * sizeof(double)), (v).data, (v).n * sizeof(double))})

#define mat_from_vec_row_nocopy(v) ((mat){1, (v).n, (v).data})
#define mat_from_vec_column_nocopy(v) ((mat){(v).n, 1, (v).data})

#define mat_get(_mat, i, j) ((_mat).data[(i) * (_mat).n + (j)])
#define mat_copy_over(m1, m2) (memcpy((m1).data, (m2).data, sizeof(double) * (m1).m * (m1).n))

#define vec_row_copy_to_mat(_mat, v, m_start_i) (memcpy((_mat).data + ((#m_start_i)[0] == '\0' ? 0 : m_start_i +0), (v).data, sizeof(double) * (v).n))
#define mat_copy_to_vec_row(v, _mat, mat_start_i) (memcpy((v).data, (_mat).data + ((#mat_start_i)[0] == '\0' ? 0 : mat_start_i +0), sizeof(double) * (v).n))
#define mat_copy_to_mat(m2, m1, amt, start_m2, start_m1) (memcpy((m2).data + ((#start_m2)[0] == '\0' ? 0 : start_m2 +0), (m1).data + ((#start_m1)[0] == '\0' ? 0 : start_m1 +0), sizeof(double) * amt))

#define mat_zero(_mat) (memset((_mat).data, 0, sizeof(double) * (_mat).m * (_mat).n))
#define mat_apply_activation_func(_mat, func) {size_t __sz = (_mat).m * (_mat).n; for(size_t i = 0; i < __sz; ++i) {(_mat).data[i] = func((_mat).data[i]);}}
#define mat_apply_activation_func_range(_mat, func, beg, end) {size_t __end = ((#end)[0] == '\0' ? (_mat).m * (_mat).n : end +0); for(size_t i = beg; i < __end; ++i) {(_mat).data[i] = func((_mat).data[i]);}}

void mat_padd(mat m1, mat m2);
void mat_psub(mat m1, mat m2);

mat mat_mul(mat m1, mat m2);
mat mat_emul(mat m1, mat m2); // element-wise multiplication
mat mat_smul(mat m, double s);
void mat_psmul(mat m, double s);

mat mat_tran(mat m);

mat vec_outer_product(vec v1, vec v2);

void mat_print(mat m);

/* Sparse matrix functions (affect only specified rows / columns) */

// sp = sparse, r = rows, c = columns
// 1/2 - which matrix is sparse (the other one isn't)
// elements not in specified rows/columns are 0

mat sprmat_mul1(mat m1, mat m2, size_t row_cnt, size_t* rows);
mat sprmat_mul2(mat m1, mat m2, size_t row_cnt, size_t* rows);
mat spcmat_mul1(mat m1, mat m2, size_t column_cnt, size_t* columns);

mat sprmat_emul1(mat m1, mat m2, size_t row_cnt, size_t* rows);
mat spcmat_emul1(mat m1, mat m2, size_t column_cnt, size_t* columns);

void sprmat_psub1(mat m1, mat m2, size_t row_cnt, size_t* rows);
void spcmat_psub1(mat m1, mat m2, size_t column_cnt, size_t* columns);

mat spcmat_tran(mat m, size_t column_cnt, size_t* columns);

/* Range matrix functions (affect only specified range of rows / columns) */

// rn = range, r = rows, c = columns
// 1/2 - which matrix is ranged (the other one isn't), 12 - both are ranged with different range start points

void rncmat_padd1(mat m1, mat m2, size_t column_beg, size_t column_end);
void rncmat_padd2(mat m1, mat m2, size_t column_beg, size_t column_end);
void rnrmat_padd1(mat m1, mat m2, size_t row_beg, size_t row_end);
void rnrmat_padd2(mat m1, mat m2, size_t row_beg, size_t row_end);

void rncmat_pemul1(mat m1, mat m2, size_t column_beg, size_t column_end);
void rncmat_pemul12(mat m1, mat m2, size_t column_beg1, size_t column_end1,
									size_t column_beg2);
void rnrmat_pemul1(mat m1, mat m2, size_t row_beg, size_t row_end);
void rnrmat_pemul12(mat m1, mat m2, size_t row_beg1, size_t row_end1,
									size_t row_beg2);

void rncmat_vec_pdot(mat m, vec v, vec _out,
						size_t column_beg, size_t column_end); // _out should have the same size as m.m
void rnrmat_vec_pdot(mat m, vec v, vec _out,
						size_t row_beg, size_t row_end);

#endif
