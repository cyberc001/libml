#include "accel.h"
#include <stdio.h>
#include <dirent.h>

cl_context accel_ctx;
cl_command_queue accel_queue;

cl_program_dict accel_programs;

void accel_callback(const char* errinfo, const void* private_info, size_t cb, void* user_data)
{
	fprintf(stderr, "OpenCL runtime error:\n%s\n", errinfo);
}

static size_t dict_hash(size_t table_sz, const char** key)
{
	size_t h = 0; const char* k = *key;
	for(; *k; ++k)
		h = (h + *k) % table_sz;
	return h;
}
static int dict_cmp(const char** key1, const char** key2)
{
	return strcmp(*key1, *key2);
}

int accel_init()
{
	cl_int err;

	// Figure out what platform and device to use, the dumb way
	cl_uint platform_ln = 0;
	cl_platform_id platform;
	err = clGetPlatformIDs(1, &platform, &platform_ln);
	if(err != CL_SUCCESS)
		TELL_OPENCL_ERR();
	if(!platform_ln){
		fprintf(stderr, "OpenCL error: no platforms available\n");
		return ACCEL_ERROR_NO_PLATFORMS;
	}

	char platform_name[64];
	clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof(platform_name), platform_name, NULL);
	fprintf(stderr, "Hardware acceleration for platform \"%s\"\n", platform_name);

	cl_uint dev_ln = 0;
	cl_device_id dev;
	clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &dev, &dev_ln);
	if(err != CL_SUCCESS)
		TELL_OPENCL_ERR();
	if(!dev_ln){
		fprintf(stderr, "OpenCL error: no devices available\n");
		return ACCEL_ERROR_NO_DEVICES;
	}

	char dev_name[64];
	clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(dev_name), dev_name, NULL);
	fprintf(stderr, "Hardware acceleration for device \"%s\"\n", dev_name);

	accel_ctx = clCreateContext(NULL, 1, &dev, accel_callback, NULL, &err);
	if(err != CL_SUCCESS)
		TELL_OPENCL_ERR();

	accel_queue = clCreateCommandQueueWithProperties(accel_ctx, dev, NULL, &err);
	if(err != CL_SUCCESS)
		TELL_OPENCL_ERR();

	// Find and compile all OpenCL programs, put them in program dictionary
	cl_program_dict_create(&accel_programs, 64, dict_hash, dict_cmp);

	const char* prog_dir = "./cl";
	DIR* dir = opendir(prog_dir);
	if(!dir){
		fprintf(stderr, "Couldn't open \"%s\" for execution\n", prog_dir);
		return ACCEL_ERROR_CANT_OPEN_DIR;
	}
	struct dirent* ent;
	while(ent = readdir(dir)){
		if(!strcmp(ent->d_name, ".") || !strcmp(ent->d_name, ".."))
			continue;
		char* ext = ent->d_name + strlen(ent->d_name) - 1;
		for(; ext >= ent->d_name && *ext != '.'; --ext) // looking for last dot in the file name
			;
		if(strcmp(ext, ".cl"))
			continue;

		char* full_path = malloc(strlen(prog_dir) + strlen(ent->d_name) + 2);
		strcpy(full_path, prog_dir);
		strcat(full_path, "/");
		strcat(full_path, ent->d_name);
		fprintf(stderr, "Compiling \"%s\"...\n", full_path);
		FILE* fd = fopen(full_path, "r");
		if(!fd){
			fprintf(stderr, "Can't compile \"%s\":\nCan't open for reading\n", full_path);
			free(full_path);
			continue;
		}

		fseek(fd, 0, SEEK_END);
		long fsz = ftell(fd);
		fseek(fd, 0, SEEK_SET);
		char* src_str = malloc(fsz + 1);
		fread(src_str, 1, fsz, fd);
		fclose(fd);
		src_str[fsz] = '\0';

		cl_program prog = clCreateProgramWithSource(accel_ctx, 1, (const char**)&src_str, NULL, &err);
		free(src_str);
		if(err != CL_SUCCESS){
			fprintf(stderr, "OpenCL error: %s (%d)\n", get_opencl_error(err), err);
			goto compile_cleanup;
		}
		err = clBuildProgram(prog, 1, &dev, NULL, NULL, NULL);
		size_t build_log_sz;
		clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &build_log_sz);
		char* build_log = malloc(build_log_sz);
		clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, build_log_sz, build_log, NULL);
		if(err != CL_SUCCESS){
			fprintf(stderr, "OpenCL error: %s (%d)\n", get_opencl_error(err), err);
			fprintf(stderr, "%s", build_log);
			free(build_log);
			goto compile_cleanup;
		}
		else{
			fprintf(stderr, "%s", build_log);
			free(build_log);
		}

		size_t prog_name_ln = strlen(ent->d_name) - strlen(ext);
		char* prog_name = malloc(prog_name_ln + 1);
		memcpy(prog_name, ent->d_name, prog_name_ln);
		prog_name[prog_name_ln] = '\0';
		cl_program_dict_insert(&accel_programs, prog_name, prog);
		fprintf(stderr, "OK.\n");

		compile_cleanup:
		free(full_path);
	}

	return 0;
}

void accel_release()
{
	clReleaseContext(accel_ctx);
	clReleaseCommandQueue(accel_queue);

	for(size_t i = 0; i < accel_programs.size; ++i)
		if(cl_program_dict_is_allocated(&accel_programs, i))
			clReleaseProgram(accel_programs.data[i]);
	cl_program_dict_destroy(&accel_programs);
}

const char* get_opencl_error(int error)
{
	// source: https://stackoverflow.com/questions/24326432/convenient-way-to-show-opencl-error-codes
	switch(error){
		// run-time and JIT compiler errors
		case 0: return "CL_SUCCESS";
		case -1: return "CL_DEVICE_NOT_FOUND";
		case -2: return "CL_DEVICE_NOT_AVAILABLE";
		case -3: return "CL_COMPILER_NOT_AVAILABLE";
		case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
		case -5: return "CL_OUT_OF_RESOURCES";
		case -6: return "CL_OUT_OF_HOST_MEMORY";
		case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
		case -8: return "CL_MEM_COPY_OVERLAP";
		case -9: return "CL_IMAGE_FORMAT_MISMATCH";
		case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
		case -11: return "CL_BUILD_PROGRAM_FAILURE";
		case -12: return "CL_MAP_FAILURE";
		case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
		case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
		case -15: return "CL_COMPILE_PROGRAM_FAILURE";
		case -16: return "CL_LINKER_NOT_AVAILABLE";
		case -17: return "CL_LINK_PROGRAM_FAILURE";
		case -18: return "CL_DEVICE_PARTITION_FAILED";
		case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

		// compile-time errors
		case -30: return "CL_INVALID_VALUE";
		case -31: return "CL_INVALID_DEVICE_TYPE";
		case -32: return "CL_INVALID_PLATFORM";
		case -33: return "CL_INVALID_DEVICE";
		case -34: return "CL_INVALID_CONTEXT";
		case -35: return "CL_INVALID_QUEUE_PROPERTIES";
		case -36: return "CL_INVALID_COMMAND_QUEUE";
		case -37: return "CL_INVALID_HOST_PTR";
		case -38: return "CL_INVALID_MEM_OBJECT";
		case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
		case -40: return "CL_INVALID_IMAGE_SIZE";
		case -41: return "CL_INVALID_SAMPLER";
		case -42: return "CL_INVALID_BINARY";
		case -43: return "CL_INVALID_BUILD_OPTIONS";
		case -44: return "CL_INVALID_PROGRAM";
		case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
		case -46: return "CL_INVALID_KERNEL_NAME";
		case -47: return "CL_INVALID_KERNEL_DEFINITION";
		case -48: return "CL_INVALID_KERNEL";
		case -49: return "CL_INVALID_ARG_INDEX";
		case -50: return "CL_INVALID_ARG_VALUE";
		case -51: return "CL_INVALID_ARG_SIZE";
		case -52: return "CL_INVALID_KERNEL_ARGS";
		case -53: return "CL_INVALID_WORK_DIMENSION";
		case -54: return "CL_INVALID_WORK_GROUP_SIZE";
		case -55: return "CL_INVALID_WORK_ITEM_SIZE";
		case -56: return "CL_INVALID_GLOBAL_OFFSET";
		case -57: return "CL_INVALID_EVENT_WAIT_LIST";
		case -58: return "CL_INVALID_EVENT";
		case -59: return "CL_INVALID_OPERATION";
		case -60: return "CL_INVALID_GL_OBJECT";
		case -61: return "CL_INVALID_BUFFER_SIZE";
		case -62: return "CL_INVALID_MIP_LEVEL";
		case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
		case -64: return "CL_INVALID_PROPERTY";
		case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
		case -66: return "CL_INVALID_COMPILER_OPTIONS";
		case -67: return "CL_INVALID_LINKER_OPTIONS";
		case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

		// extension errors
		case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
		case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
		case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
		case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
		case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
		case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
		default: return "Unknown OpenCL error";
	}
}
