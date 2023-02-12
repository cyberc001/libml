#include "mat.h"
#include <stdio.h>

void mat_psub(mat m1, mat m2)
{
	size_t sz = m1.m * m1.n;
	for(size_t i = 0; i < sz; ++i)
		m1.data[i] -= m2.data[i];
}
void mat_padd(mat m1, mat m2)
{
	size_t sz = m1.m * m1.n;
	for(size_t i = 0; i < sz; ++i)
		m1.data[i] += m2.data[i];
}

mat mat_mul(mat m1, mat m2)
{
	mat m = mat_create(m1.m, m2.n);
	for(size_t i = 0; i < m1.m; ++i) // i -> row in 1st matrix (row in result)
		for(size_t j = 0; j < m2.n; ++j){ // j -> column in 2nd matrix
			mat_get(m, i, j) = 0;
			for(size_t k = 0; k < m1.n; ++k) // k -> column in 1st matrix (row in 2nd)
				mat_get(m, i, j) += mat_get(m1, i, k) * mat_get(m2, k, j);
		}
	return m;
}
mat mat_emul(mat m1, mat m2)
{
	mat m = mat_create(m1.m, m1.n);
	size_t sz = m1.m * m1.n;
	for(size_t i = 0; i < sz; ++i)
		m.data[i] = m1.data[i] * m2.data[i];
	return m;
}
mat mat_smul(mat m, double s)
{
	mat _m = mat_create(m.m, m.n);
	size_t sz = m.m * m.n;
	for(size_t i = 0; i < sz; ++i)
		_m.data[i] = m.data[i] * s;
	return _m;
}
void mat_psmul(mat m, double s)
{
	size_t sz = m.m * m.n;
	for(size_t i = 0; i < sz; ++i)
		m.data[i] *= s;
}

mat mat_tran(mat m)
{
	mat o = mat_create(m.n, m.m);
	for(size_t i = 0; i < o.m; ++i)
		for(size_t j = 0; j < o.n; ++j)
			mat_get(o, i, j) = mat_get(m, j, i);
	return o;
}

void mat_print(mat m)
{
	size_t sz = m.m * m.n;
	for(size_t i = 0; i < sz; ++i){
		if(i != 0 && i % m.n == 0)
			puts("");
		printf("%g ", m.data[i]);
	}
	puts("");
}

/* Sparse matrix functions */

mat sprmat_mul1(mat m1, mat m2, size_t row_cnt, size_t* rows)
{
	mat m = mat_create(row_cnt, m2.n);
	for(size_t i = 0; i < row_cnt; ++i) // i -> row in 1st matrix (row in result)
		for(size_t j = 0; j < m2.n; ++j){ // j -> column in 2nd matrix
			size_t r = rows[i];
			mat_get(m, i, j) = 0;
			for(size_t k = 0; k < m1.n; ++k) // k -> column in 1st matrix (row in 2nd)
				mat_get(m, i, j) += mat_get(m1, r, k) * mat_get(m2, k, j);
		}
	return m;
}
mat sprmat_mul2(mat m1, mat m2, size_t row_cnt, size_t* rows)
{
	mat m = mat_create(m1.m, m2.n);
	for(size_t i = 0; i < m1.m; ++i) // i -> row in 1st matrix (row in result)
		for(size_t j = 0; j < m2.n; ++j){ // j -> column in 2nd matrix
			mat_get(m, i, j) = 0;
			for(size_t k = 0; k < row_cnt; ++k) // k -> column in 1st matrix (row in 2nd)
				mat_get(m, i, j) += mat_get(m1, i, k) * mat_get(m2, rows[k], j);
		}
	return m;
}

mat spcmat_mul1(mat m1, mat m2, size_t column_cnt, size_t* columns)
{
	mat m = mat_create(m1.m, m2.n);
	for(size_t i = 0; i < m1.m; ++i) // i -> row in 1st matrix (row in result)
		for(size_t j = 0; j < m2.n; ++j){ // j -> column in 2nd matrix
			mat_get(m, i, j) = 0;
			for(size_t k = 0; k < column_cnt; ++k) // k -> column in 1st matrix (row in 2nd)
				mat_get(m, i, j) += mat_get(m1, i, columns[k]) * mat_get(m2, k, j);
		}
	return m;
}

mat sprmat_emul1(mat m1, mat m2, size_t row_cnt, size_t* rows)
{
	mat m = mat_create(m1.m, m1.n);
	memset(m.data, 0, sizeof(double) * m.m * m.n);
	for(size_t i = 0; i < row_cnt; ++i){
		for(size_t j = 0; j < m.n; ++j){
			size_t r = rows[i];
			mat_get(m, r, j) = mat_get(m1, r, j) * mat_get(m2, i, j);
		}
	}
	return m;
}
mat spcmat_emul1(mat m1, mat m2, size_t column_cnt, size_t* columns)
{
	mat m = mat_create(m1.m, m1.n);
	memset(m.data, 0, sizeof(double) * m.m * m.n);
	for(size_t i = 0; i < m.m; ++i){
		for(size_t j = 0; j < column_cnt; ++j){
			size_t c = columns[i];
			mat_get(m, i, c) = mat_get(m1, i, c) * mat_get(m2, i, j);
		}
	}
	return m;
}

void sprmat_psub1(mat m1, mat m2, size_t row_cnt, size_t* rows)
{
	for(size_t i = 0; i < row_cnt; ++i){
		size_t r = rows[i];
		for(size_t j = 0; j < m1.n; ++j)
			mat_get(m1, r, j) -= mat_get(m2, i, j);
	}
}
void spcmat_psub1(mat m1, mat m2, size_t column_cnt, size_t* columns)
{
	for(size_t j = 0; j < column_cnt; ++j){
		size_t c = columns[j];
		for(size_t i = 0; i < m1.m; ++i)
			mat_get(m1, i, c) -= mat_get(m2, i, j);
	}
}

mat spcmat_tran(mat m, size_t column_cnt, size_t* columns)
{
	mat o = mat_create(column_cnt, m.m);
	for(size_t j = 0; j < column_cnt; ++j){
		size_t c = columns[j];
		for(size_t i = 0; i < m.m; ++i)
			mat_get(o, j, i) = mat_get(m, i, c);
	}
	return o;
}
