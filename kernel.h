#ifndef KERNELH
#define KERNELH

#define GLEW_STATIC
#include <GL\glew.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda_gl_interop.h>

#include <curand_kernel.h>
#include "hitable.h"
#include "camera.h"

namespace CUDAtrace {
	void createWorld(hitable **d_list, hitable **d_world, camera **d_camera, vec3 camera_position, GLuint *fbTexture, cudaGraphicsResource_t *viewCudaResource);
	void updateWorld(camera **d_camera, vec3 position, vec3 look);
	void rayTrace(int width, int height, int sample_count, int rand_seed, hitable **world, camera **camera, curandState *rand_state, GLuint *fbTexture, cudaGraphicsResource_t *viewCudaResource);
	void cleanup(hitable **d_list, hitable **d_world, camera **d_camera, GLuint *pbo);
}

#endif