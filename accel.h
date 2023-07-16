#ifndef ACCEL_H
#define ACCEL_H

#include <CL/cl.h>
#include "htable_oa.h"

#define ACCEL_ERROR_OPENCL		-1
#define ACCEL_ERROR_NO_PLATFORMS	-2
#define ACCEL_ERROR_NO_DEVICES		-3
#define ACCEL_ERROR_CANT_OPEN_DIR	-4

#define TELL_OPENCL_ERR()\
{\
	fprintf(stderr, "OpenCL error: %s (%d)\n", get_opencl_error(err), err);\
	return ACCEL_ERROR_OPENCL;\
}


DEF_HTABLE_OA(cl_program_dict, const char*, cl_program)

extern cl_context accel_ctx;
extern cl_command_queue accel_queue;

extern cl_program_dict accel_programs;

int accel_init();
void accel_release();

const char* get_opencl_error(int error);

#define ACCEL_FUNC_KERNEL_TEMPLATE(prog_name, kernel_name)\
	static cl_kernel kernel = NULL;\
	static cl_event event;\
	if(!kernel){\
		const cl_program* prog = cl_program_dict_find(&accel_programs, prog_name);\
		if(!prog){\
			fprintf(stderr, "Couldn't find program \"%s\" in the global dictionary\n", prog_name);\
			exit(-1);\
		}\
		int err;\
		kernel = clCreateKernel(*prog, kernel_name, &err);\
		if(err != CL_SUCCESS){\
			fprintf(stderr, "While trying to create kernel \"%s\" for program \"%s\":\n", kernel_name, prog_name);\
			fprintf(stderr, "OpenCL error: %s (%d)\n", get_opencl_error(err), err);\
			exit(-1);\
		}\
		event = clCreateUserEvent(accel_ctx, NULL);\
	}

#define ACCEL_FUNC_ENQUEUE_TEMPLATE(_m, _n, _TS)\
	const size_t TS = (_TS);\
	const size_t offset[2] = {0, 0};\
	const size_t global[2] = {(_m), (_n)};\
	const size_t local[2] = {MIN(TS, (_m)), MIN(TS, (_n))};\
	clEnqueueNDRangeKernel(accel_queue, kernel, 2, offset,\
				global, local, 0, NULL, &event);\
	clWaitForEvents(1, &event); 

#endif
