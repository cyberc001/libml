#include "mat.h"
#include <stdio.h>
#include "common.h"
#include "accel.h"

mat mat_create_zero(size_t m, size_t n)
{
	mat _m = mat_create(m, n);
	mat_zero(_m);
	return _m;
}

void mat_padd(mat m1, mat m2)
{
#if ML_CPU == 1
	size_t sz = m1.m * m1.n;
	for(size_t i = 0; i < sz; ++i)
		m1.data[i] += m2.data[i];
#else
	ACCEL_FUNC_KERNEL("mat_padd");

	ACCEL_FUNC_ARG(0, int, &m1.m);
	ACCEL_FUNC_ARG(1, int, &m1.n);

	ACCEL_FUNC_ARG_BUFF(2, m1_buf, CL_MEM_READ_WRITE, m1.m * m1.n * sizeof(double), m1.data);
	ACCEL_FUNC_ARG_BUFF(3, m2_buf, CL_MEM_READ_ONLY, m2.m * m2.n * sizeof(double), m2.data);

	ACCEL_FUNC_ENQUEUE(m1.m, m1.n, 32,,);
	//ACCEL_FUNC_PROFILE_ENQUEUE("mat_padd");

	clEnqueueReadBuffer(accel_queue, m1_buf, CL_TRUE, 0, m1.m * m1.n * sizeof(double), m1.data, 0, NULL, NULL);
	clReleaseMemObject(m1_buf);
	clReleaseMemObject(m2_buf);
#endif
}
void mat_psub(mat m1, mat m2)
{
	size_t sz = m1.m * m1.n;
	for(size_t i = 0; i < sz; ++i)
		m1.data[i] -= m2.data[i];
}

mat mat_mul(mat m1, mat m2)
{
#if ML_CPU == 1
	mat m = mat_create(m1.m, m2.n);
	for(size_t i = 0; i < m1.m; ++i) // i -> row in 1st matrix (row in result)
		for(size_t j = 0; j < m2.n; ++j){ // j -> column in 2nd matrix
			mat_get(m, i, j) = 0;
			for(size_t k = 0; k < m1.n; ++k) // k -> column in 1st matrix (row in 2nd)
				mat_get(m, i, j) += mat_get(m1, i, k) * mat_get(m2, k, j);
		}
	return m;
#else
	ACCEL_FUNC_KERNEL("mat_mul"); 
	mat m = mat_create_zero(m1.m, m2.n);

	ACCEL_FUNC_ARG(0, int, &m1.m);
	ACCEL_FUNC_ARG(1, int, &m1.n);
	ACCEL_FUNC_ARG(2, int, &m2.n);

	ACCEL_FUNC_ARG_BUFF(5, m_buf, CL_MEM_READ_WRITE, m.m * m.n * sizeof(double), m.data);
	ACCEL_FUNC_ARG_BUFF(3, m1_buf, CL_MEM_READ_ONLY, m1.m * m1.n * sizeof(double), m1.data);
	ACCEL_FUNC_ARG_BUFF(4, m2_buf, CL_MEM_READ_ONLY, m2.m * m2.n * sizeof(double), m2.data);

	ACCEL_FUNC_ENQUEUE(m1.m, m2.n, 32,,);
	//ACCEL_FUNC_PROFILE_ENQUEUE("mat_mul");

	clEnqueueReadBuffer(accel_queue, m_buf, CL_TRUE, 0, m.m * m.n * sizeof(double), m.data, 0, NULL, NULL);
	clReleaseMemObject(m1_buf);
	clReleaseMemObject(m2_buf);
	clReleaseMemObject(m_buf);

	return m;
#endif
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
#if ML_CPU == 1
	for(size_t i = 0; i < o.m; ++i)
		for(size_t j = 0; j < o.n; ++j)
			mat_get(o, i, j) = mat_get(m, j, i)i;
	return o;
#else
	ACCEL_FUNC_KERNEL("mat_tran");

	ACCEL_FUNC_ARG(0, int, &m.m);
	ACCEL_FUNC_ARG(1, int, &m.n);

	ACCEL_FUNC_ARG_BUFF(2, m_buf, CL_MEM_READ_ONLY, m.m * m.n * sizeof(double), m.data);
	ACCEL_FUNC_ARG_BUFF(3, o_buf, CL_MEM_READ_WRITE, o.m * o.n * sizeof(double), o.data);

	ACCEL_FUNC_ENQUEUE(m.m, m.n, 32,,);

	clEnqueueReadBuffer(accel_queue, o_buf, CL_TRUE, 0, o.m * o.n * sizeof(double), o.data, 0, NULL, NULL);
	clReleaseMemObject(m_buf);
	clReleaseMemObject(o_buf);

	return o;
#endif
}


mat vec_outer_product(vec v1, vec v2)
{
	mat _out = mat_create(v1.n, v2.n);
#if ML_CPU == 1
	for(size_t i = 0; i < v1.n; ++i)
		for(size_t j = 0; j < v2.n; ++j)
			_out.data[i * v2.n + j] = v1.data[i] * v2.data[j];
	return _out;
#else
	ACCEL_FUNC_KERNEL("vec_outer_product");

	ACCEL_FUNC_ARG(0, int, &v1.n);
	ACCEL_FUNC_ARG(1, int, &v2.n);

	ACCEL_FUNC_ARG_BUFF(2, v1_buf, CL_MEM_READ_ONLY, v1.n * sizeof(double), v1.data);
	ACCEL_FUNC_ARG_BUFF(3, v2_buf, CL_MEM_READ_ONLY, v2.n * sizeof(double), v2.data);
	ACCEL_FUNC_ARG_BUFF(4, m_buf, CL_MEM_READ_WRITE, _out.m * _out.n * sizeof(double), _out.data);

	ACCEL_FUNC_ENQUEUE(_out.m, _out.n, 32,,);
	//ACCEL_FUNC_PROFILE_ENQUEUE("vec_outer_product");

	clEnqueueReadBuffer(accel_queue, m_buf, CL_TRUE, 0, _out.m * _out.n * sizeof(double), _out.data, 0, NULL, NULL);
	clReleaseMemObject(v1_buf);
	clReleaseMemObject(v2_buf);
	clReleaseMemObject(m_buf);

	return _out;
#endif
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

/* Range matrix functions */

void rncmat_padd1(mat m1, mat m2, size_t column_beg, size_t column_end)
{
	for(size_t i = 0; i < m1.m; ++i)
		for(size_t j = column_beg; j < column_end; ++j)
			mat_get(m1, i, j) += mat_get(m2, i, j - column_beg);
}
void rncmat_padd2(mat m1, mat m2, size_t column_beg, size_t column_end)
{
	for(size_t i = 0; i < m1.m; ++i)
		for(size_t j = column_beg; j < column_end; ++j)
			mat_get(m1, i, j - column_beg) += mat_get(m2, i, j);
}
void rnrmat_padd1(mat m1, mat m2, size_t row_beg, size_t row_end)
{
	for(size_t i = row_beg; i < row_end; ++i)
		for(size_t j = 0; j < m1.n; ++j)
			mat_get(m1, i, j) += mat_get(m2, i - row_beg, j);
}
void rnrmat_padd2(mat m1, mat m2, size_t row_beg, size_t row_end)
{
	for(size_t i = row_beg; i < row_end; ++i)
		for(size_t j = 0; j < m1.n; ++j)
			mat_get(m1, i - row_beg, j) += mat_get(m2, i, j);
}

void rncmat_pemul1(mat m1, mat m2, size_t column_beg, size_t column_end)
{
	for(size_t i = 0; i < m1.m; ++i)
		for(size_t j = column_beg; j < column_end; ++j)
			mat_get(m1, i, j) *= mat_get(m2, i, j - column_beg);
}
void rncmat_pemul12(mat m1, mat m2, size_t column_beg1, size_t column_end1,
									size_t column_beg2)
{
	if(column_beg1 > column_beg2){
		size_t diff = column_beg1 - column_beg2;
		for(size_t i = 0; i < m1.m; ++i)
			for(size_t j = column_beg1; j < column_end1; ++j)
				mat_get(m1, i, j) *= mat_get(m2, i, j - diff);
	}
	else{
		size_t diff = column_beg2 - column_beg1;
		for(size_t i = 0; i < m1.m; ++i)
			for(size_t j = column_beg1; j < column_end1; ++j)
				mat_get(m1, i, j) *= mat_get(m2, i, j + diff);
	}
}
void rnrmat_pemul1(mat m1, mat m2, size_t row_beg, size_t row_end)
{
	for(size_t i = row_beg; i < row_end; ++i)
		for(size_t j = 0; j < m1.n; ++j)
			mat_get(m1, i, j) *= mat_get(m2, i - row_beg, j);
}
void rnrmat_pemul12(mat m1, mat m2, size_t row_beg1, size_t row_end1,
									size_t row_beg2)
{
	if(row_beg1 > row_beg2){
		size_t diff = row_beg1 - row_beg2;
		for(size_t i = row_beg1; i < row_end1; ++i)
			for(size_t j = 0; j < m1.n; ++j)
				mat_get(m1, i, j) *= mat_get(m2, i - diff, j);
	}
	else{
		size_t diff = row_beg2 - row_beg1;
		for(size_t i = row_beg1; i < row_end1; ++i)
			for(size_t j = 0; j < m1.n; ++j)
				mat_get(m1, i, j) *= mat_get(m2, i + diff, j);
	}
}

void rncmat_vec_pdot(mat m, vec v, vec _out,
					size_t column_beg, size_t column_end)
{
	vec_zero(_out);
#if ML_CPU == 1
	for(size_t i = 0; i < m.m; ++i)
		for(size_t j = column_beg; j < column_end; ++j)
			_out.data[i] += mat_get(m, i, j) * v.data[j - column_beg];
#else
	ACCEL_FUNC_KERNEL("rncmat_vec_pdot");
	
	ACCEL_FUNC_ARG(0, int, &m.m);
	ACCEL_FUNC_ARG(1, int, &m.n);
	ACCEL_FUNC_ARG(2, int, &column_beg);
	ACCEL_FUNC_ARG(3, int, &column_end);

	ACCEL_FUNC_ARG_BUFF(6, w_buf, CL_MEM_READ_WRITE, _out.n * sizeof(double), _out.data);
	ACCEL_FUNC_ARG_BUFF(4, m_buf, CL_MEM_READ_ONLY, m.m * m.n * sizeof(double), m.data);
	ACCEL_FUNC_ARG_BUFF(5, v_buf, CL_MEM_READ_ONLY, v.n * sizeof(double), v.data);

	ACCEL_FUNC_ENQUEUE1D(m.m, 32,,);
	//ACCEL_FUNC_PROFILE_ENQUEUE("rncmat_vec_pdot");

	clEnqueueReadBuffer(accel_queue, w_buf, CL_TRUE, 0, _out.n * sizeof(double), _out.data, 0, NULL, NULL);
	clReleaseMemObject(m_buf);
	clReleaseMemObject(v_buf);
	clReleaseMemObject(w_buf);
#endif

}
void rnrmat_vec_pdot(mat m, vec v, vec _out,
						size_t row_beg, size_t row_end)
{
	vec_zero(_out);
	for(size_t i = row_beg; i < row_end; ++i)
		for(size_t j = 0; j < m.n; ++j)
			_out.data[i] += mat_get(m, i, j) * v.data[i - row_beg];
}
