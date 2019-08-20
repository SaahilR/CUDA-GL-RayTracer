#include "kernel.h"

// CUDA header includes
#include "sphere.h"
#include "hitable_list.h"
#include "material.h"

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__)

void check_cuda(cudaError_t result, char const * const func, const char * const file, int const line) {
	if (result) {
		std::cerr << "CUDA Error: " << static_cast<unsigned int>(result) << " at " << file << ":" << line << " '" << func << "' \n";
		system("pause");
		cudaDeviceReset();
		exit(99);
	}
}

__device__ vec3 color(const ray& r,
		hitable **world,
		curandState *local_rand_state) {
	ray cur_ray = r;
	vec3 cur_attenuation = vec3(1.0f, 1.0f, 1.0f);

	for (int i = 0; i < 4; i++) {
		hit_record rec;
		if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
			ray scattered;
			vec3 attenuation;
			if (rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
				cur_attenuation *= attenuation;
				cur_ray = scattered;
			}
			else {
				return vec3(0.0f, 0.0f, 0.0f);
			}
		}
		else {
			vec3 unit_direction = unit_vector(cur_ray.direction());
			float t = 0.5f * (unit_direction.y() + 1.0f);
			vec3 c = (1.0f - t) * vec3(1.0f, 1.0f, 1.0f) + t * vec3(0.5f, 0.7f, 1.0f);
			return cur_attenuation * c;
		}
	}
	return vec3(0.0f, 0.0f, 0.0f);
}

__global__ void create_world(hitable **d_list,
		hitable **d_world,
		camera **d_camera,
		vec3 camera_position) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		d_list[0] = new sphere(vec3(0.0f, 0.0f, -1.0f), 0.5f,
			new lambertian(vec3(0.8f, 0.3f, 0.3f)));
		d_list[1] = new sphere(vec3(0.0f, -100.5, -1.0f), 100.0f,
			new lambertian(vec3(0.3f, 0.9f, 0.3f)));
		d_list[2] = new sphere(vec3(1.0f, 0.0f, -1.0f), 0.5f,
			new metal(vec3(0.8f, 0.6f, 0.2f), 1.0f));
		d_list[3] = new sphere(vec3(-1.0f, 0.0f, -1.0f), 0.5f,
			new metal(vec3(0.2f, 0.1f, 0.9f), 0.2f));
		d_list[4] = new sphere(vec3(-1, 0, -1), -0.45,
			new dielectric(1.1));

		*d_world = new hitable_list(d_list, 5);
		*d_camera = new camera(camera_position, vec3(0, 0, -1), vec3(0, 1, 0), 90.0f, 4.0f / 3.0f);
	}
}

__global__ void update_world(camera **d_camera, vec3 position, vec3 look) {
	(*d_camera)->setOrigin(position);
	(*d_camera)->setLook(look);
	(*d_camera)->updateCamera();
}

__global__ void free_world(hitable **d_list,
		hitable **d_world,
		camera **d_camera) {
	for (int i = 0; i < 4; i++) {
		delete ((sphere *)d_list[i])->mat_ptr;
		delete d_list[i];
	}
	delete *d_world;
	delete *d_camera;
}

// Initialize the random value function for CUDA
__global__ void render_init(int width, int height, curandState *rand_state, int seed) {
	int index = (threadIdx.x + blockIdx.x * blockDim.x);
	int x = index % width;
	int y = (height - 1) - (index / width);
	if (x >= width || y >= height) return;

	curand_init(1984 + index + seed, 0, 0, &rand_state[index]);
}

__global__ void render(int width,
		int height,
		int sample_count,
		hitable **world,
		camera **cam,
		curandState *rand_state,
		cudaSurfaceObject_t surface) {
	int index = (threadIdx.x + blockIdx.x * blockDim.x);
	int x = index % width;
	int y = (height - 1) - (index / width);
	if (x >= width || y >= height) return;

	curandState local_rand_state = rand_state[index];

	vec3 fragColor(0.0f, 0.0f, 0.0f);
	for (int sampleId = 0; sampleId < sample_count; sampleId++) {
		float u = float(x + curand_uniform(&local_rand_state)) / float(width);
		float v = float(y + curand_uniform(&local_rand_state)) / float(height);

		ray r = (*cam)->get_ray(u, v);
		fragColor += color(r, world, &local_rand_state);
	}
	rand_state[index] = local_rand_state;

	fragColor /= sample_count;
	fragColor = vec3(255 * sqrt(fragColor.r()), 255 * sqrt(fragColor.g()), 255 * sqrt(fragColor.b()));
	uchar4 readValue;
	surf2Dread(&readValue, surface, x * sizeof(uchar4), y, cudaBoundaryModeClamp);
	uchar4 data = make_uchar4((fragColor.r() + readValue.x) / 2, (fragColor.g() + readValue.y) / 2, (fragColor.b() + readValue.z) / 2, 255);
	
	surf2Dwrite(data, surface, x * sizeof(data), y, cudaBoundaryModeClamp);
}

namespace CUDAtrace {
	void createWorld(hitable **d_list,
			hitable **d_world,
			camera **d_camera,
			vec3 camera_position,
			GLuint *fbTexture,
			cudaGraphicsResource_t *viewCudaResource) {
		// Run kernel to generate object in the world
		create_world <<<1, 1>>> (d_list, d_world, d_camera, camera_position);
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());
		//printf("World objects created\n");
	}

	void updateWorld(camera **d_camera, vec3 position, vec3 look) {
		update_world <<<1, 1 >>> (d_camera, position, look);
	}

	void rayTrace(int width,
			int height,
			int sample_count,
			int rand_seed,
			hitable **world,
			camera **camera,
			curandState *rand_state,
			GLuint *fbTexture,
			cudaGraphicsResource_t *viewCudaResource) {
		int deviceId;
		cudaDeviceProp deviceProp;
		cudaGetDevice(&deviceId);
		cudaGetDeviceProperties(&deviceProp, deviceId);
		int warp = 16 * deviceProp.warpSize;
		
		dim3 blocks = (((width * height) / warp));
		dim3 threads = (warp);
		//std::cout << blocks.x << " " << threads.x << std::endl;
		render_init <<<blocks, threads>>>(width, height, rand_state, rand_seed);
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());
		//printf("Random Values Generated\n");

		// CUDA GL Interop
		checkCudaErrors(cudaGraphicsMapResources(1, viewCudaResource));
		
		cudaArray_t viewCudaArray;
		checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&viewCudaArray, *viewCudaResource, 0, 0));
		
		cudaResourceDesc viewCudaArrayResourceDesc;
		memset(&viewCudaArrayResourceDesc, 0, sizeof(viewCudaArrayResourceDesc));
		viewCudaArrayResourceDesc.resType = cudaResourceTypeArray;
		viewCudaArrayResourceDesc.res.array.array = viewCudaArray;
			
		cudaSurfaceObject_t viewCudaSurfaceObject;
		checkCudaErrors(cudaCreateSurfaceObject(&viewCudaSurfaceObject, &viewCudaArrayResourceDesc));
		
		render <<<blocks, threads>>>(width,
				height,
				sample_count,
				world,
				camera,
				rand_state,
				viewCudaSurfaceObject);
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());
		
		checkCudaErrors(cudaDestroySurfaceObject(viewCudaSurfaceObject));
		checkCudaErrors(cudaGraphicsUnmapResources(1, viewCudaResource));
		checkCudaErrors(cudaStreamSynchronize(0));
		//printf("CUDA Device has finished working\n");
	}

	// CUDA cleanup of allocated data for camera and world information
	void cleanup(hitable **d_list,
			hitable **d_world,
			camera **d_camera,
			GLuint *fb_Texture) {
		checkCudaErrors(cudaDeviceSynchronize());
		free_world <<<1, 1>>> (d_list, d_world, d_camera);
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaFree(d_list));
		checkCudaErrors(cudaFree(d_world));
		checkCudaErrors(cudaFree(d_camera));

	}
}