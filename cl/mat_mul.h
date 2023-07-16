// source: https://cnugteren.github.io/tutorial/pages/page3.html
const char* mat_mul = 
"__kernel void mat_mul(const int M, const int K, const int N,\n"
"			    const __global double* A, const __global double* B,\n"
"			    __global double* C)\n"
"{\n"
"	//global thread IDs: row and column number of matrix C\n"
"	const int gr = get_global_id(0);\n"
"	const int gc = get_global_id(1);\n"
"	float sum = 0;\n"
"	for(int k = 0; k < K; ++k)\n"
"		sum += A[gr*K + k] * B[k*N + gc];\n"
"	C[gr*N + gc] = sum;\n"
"}\n";
