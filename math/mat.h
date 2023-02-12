#ifndef MAT_H
#define MAT_H

#include <stddef.h>
#include "vec.h"

typedef struct {
	size_t m, n;
	double* data;
} mat;

#define mat_create(m, n) ((mat){(m), (n), malloc(sizeof(double) * (m) * (n))})
#define mat_free(m) (free((m).data))

#define mat_from_vec_row(v) ((mat){1, (v).n, memcpy(malloc((v).n * sizeof(double)), (v).data, (v).n * sizeof(double))})
#define mat_from_vec_column(v) ((mat){(v).n, 1, memcpy(malloc((v).n * sizeof(double)), (v).data, (v).n * sizeof(double))})

#define mat_from_vec_row_nocopy(v) ((mat){1, (v).n, (v).data})
#define mat_from_vec_column_nocopy(v) ((mat){(v).n, 1, (v).data})

#define mat_get(_mat, i, j) ((_mat).data[(i) * (_mat).n + (j)])
#define mat_copy_over(m1, m2) (memcpy(m1.data, m2.data, sizeof(double) * m1.m * m1.n))

void mat_psub(mat m1, mat m2);
void mat_padd(mat m1, mat m2);

mat mat_mul(mat m1, mat m2);
mat mat_emul(mat m1, mat m2); // element-wise multiplication
mat mat_smul(mat m, double s);
void mat_psmul(mat m, double s);

mat mat_tran(mat m);

void mat_print(mat m);

/* Sparse matrix functions (affect only specified rows) */

// sp = sparse, r = rows, c = columns
// elements not in specified rows/columns are 0

mat sprmat_mul1(mat m1, mat m2, size_t row_cnt, size_t* rows);
mat sprmat_mul2(mat m1, mat m2, size_t row_cnt, size_t* rows);
mat spcmat_mul1(mat m1, mat m2, size_t column_cnt, size_t* columns);

mat sprmat_emul1(mat m1, mat m2, size_t row_cnt, size_t* rows);
mat spcmat_emul1(mat m1, mat m2, size_t column_cnt, size_t* columns);

void sprmat_psub1(mat m1, mat m2, size_t row_cnt, size_t* rows);
void spcmat_psub1(mat m1, mat m2, size_t column_cnt, size_t* columns);

mat spcmat_tran(mat m, size_t column_cnt, size_t* columns);

#endif
