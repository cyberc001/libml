/* General assumptions:
 * > matrices are stored in row-major order.
 */

// Adds matrices A and B (MxN) together, result is stored in matrix A.
COMPILE_CL_PROGRAM(mat_padd,
"__kernel void mat_padd(const int M, const int N,\n"
"			__global double* A, const __global double* B)\n"
"{\n"
"	const int gr = get_global_id(0);\n"
"	const int gc = get_global_id(1);\n"
"	A[gr*N + gc] = B[gr*N + gc];\n"
"}\n");


// source: https://cnugteren.github.io/tutorial/pages/page3.html
// Multiplies A (MxK) and B (KxN) matrices and puts the result in matrix C.
COMPILE_CL_PROGRAM(mat_mul, 
"__kernel void mat_mul(const int M, const int K, const int N,\n"
"			    const __global double* A, const __global double* B,\n"
"			    __global double* C)\n"
"{\n"
"	//global thread IDs: row and column number of matrix C\n"
"	const int gr = get_global_id(0);\n"
"	const int gc = get_global_id(1);\n"
"	double sum = 0;\n"
"	for(int k = 0; k < K; ++k)\n"
"		sum += A[gr*K + k] * B[k*N + gc];\n"
"	C[gr*N + gc] = sum;\n"
"}\n");

// Calculates a dot product between matrix A (MxN) and vector V (N) for a range of columns [cbeg; cend) and puts it in vector W (M).
COMPILE_CL_PROGRAM(rncmat_vec_pdot,
"__kernel void rncmat_vec_pdot(const int M, const int N, const int cbeg, const int cend,\n"
"				const __global double* A, const __global double* V, __global double* W)\n"
"{\n"
"	const int gr = get_global_id(0);\n"
"	for(int gc = cbeg; gc < cend; ++gc)\n"
"		W[gr] += A[gr*N + gc] * V[gc];\n"
"}\n");

// Calculates an outer product between vectors V (M) and W (N).
// Basically, elements of resulting matrix A (MxN) A[i,j] = V[i] * W[j]
COMPILE_CL_PROGRAM(vec_outer_product,
"__kernel void vec_outer_product(const int M, const int N,\n"
"					const __global double* V, const __global double* W, __global double* A)\n"
"{\n"
"	const int gr = get_global_id(0);\n"
"	const int gc = get_global_id(1);\n"
"	A[gr*N + gc] = V[gr] * W[gc];\n"
"}\n");
