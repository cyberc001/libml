// source: https://cnugteren.github.io/tutorial/pages/page3.html

__kernel void mat_mul(const int M, const int K, const int N,
			    const __global double* A, const __global double* B,
			    __global double* C)
{
	//global thread IDs: row and column number of matrix C
	const int gr = get_global_id(0);
	const int gc = get_global_id(1);
	float sum = 0;
	for(int k = 0; k < K; ++k)
		sum += A[gr*K + k] * B[k*N + gc];
	C[gr*N + gc] = sum;
}
