#include "kernel.h"

#define GLEW_STATIC
#include "GL\glew.h"
#include "GLFW\glfw3.h"

#include "vector.h"
#include "ray.h"
#include "user_input.h"

#include <iostream>
#include <fstream>

void getCUDAinfo() {
	// Cuda Device Information
	int deviceId;
	cudaDeviceProp deviceProp;
	cudaGetDevice(&deviceId);
	cudaGetDeviceProperties(&deviceProp, deviceId);
	std::cout << "Device name: " << deviceProp.name << std::endl;
	std::cout << "" << deviceProp.warpSize << std::endl;
	std::cout << "" << deviceProp.major << "." << deviceProp.minor << std::endl;
}

void KeyCallback(GLFWwindow *window, int key, int scancode, int action, int mode);

int main(int argc, char *argv[]) {
	const int width = 640, const height = 480, const sample_count = 1;
	const int numPixels = width * height;
	const int framebufferSize = numPixels * sizeof(vec3);
	
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
	glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);

	GLFWwindow *window = glfwCreateWindow(width, height, "Ray Tracing", NULL, NULL);
	if (!window) exit(EXIT_FAILURE);
	
	glfwMakeContextCurrent(window);
	glfwSetKeyCallback(window, KeyCallback);

	glewExperimental = GL_TRUE;
	if (glewInit() != GLEW_OK) exit(EXIT_FAILURE);

	GLuint fbTexture;
	cudaGraphicsResource_t viewCudaResource;

	glEnable(GL_TEXTURE_2D);
	glGenTextures(1, &fbTexture);
	glBindTexture(GL_TEXTURE_2D, fbTexture);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	glBindTexture(GL_TEXTURE_2D, 0);

	// run a bind text kernel
	hitable **d_list;
	cudaMalloc((void **)&d_list, 5 * sizeof(hitable *));
	hitable **d_world;
	cudaMalloc((void **)&d_world, sizeof(hitable *));
	camera **d_camera;
	cudaMallocManaged(&d_camera, sizeof(camera *));
	curandState *d_rand_state;
	cudaMalloc((void **)&d_rand_state, numPixels * sizeof(curandState));

	// Register the GL texture to a CUDA Resource Object
	cudaGraphicsGLRegisterImage(&viewCudaResource, fbTexture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);
	
	// Create the world objects
	CUDAtrace::createWorld(d_list, d_world, d_camera, vec3(1.0f, 2.0f, -4.0f), &fbTexture, &viewCudaResource);

	// Render a single frame
	CUDAtrace::rayTrace(width,
			height,
			sample_count,
			0,
			d_world,
			d_camera,
			d_rand_state,
			&fbTexture,
			&viewCudaResource);

	GLfloat deltaTime = 0.0f;
	GLfloat lastFrame = 0.0f;
	user_input inputHandler;
	// Render loop
	while (!glfwWindowShouldClose(window)) {
		GLfloat currentFrame = (GLfloat)glfwGetTime();
		deltaTime = currentFrame - lastFrame;
		lastFrame = currentFrame;

		glClear(GL_COLOR_BUFFER_BIT);
		glClearColor(0.3f, 0.5f, 0.8f, 1.0f);

		glfwPollEvents();
		//inputHandler.ProcessInputs(deltaTime, **d_camera);
		CUDAtrace::updateWorld(d_camera, vec3(2.0f, 1.0f, 1.0f), vec3(0.0f, -1.0f, 0.0f));
		CUDAtrace::rayTrace(width,
			height,
			sample_count,
			1000 * currentFrame,
			d_world,
			d_camera,
			d_rand_state,
			&fbTexture,
			&viewCudaResource);
		// run the CUDA kernel
		// Render using immediate mode
		glBindTexture(GL_TEXTURE_2D, fbTexture);
		glBegin(GL_QUADS);
		glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0f, -1.0f);
		glTexCoord2f(1.0f, 0.0f); glVertex2f(+1.0f, -1.0f);
		glTexCoord2f(1.0f, 1.0f); glVertex2f(+1.0f, +1.0f);
		glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0f, +1.0f);
		glEnd();
		glBindTexture(GL_TEXTURE_2D, 0);
		glFinish();

		glfwSwapBuffers(window);
	}

	CUDAtrace::cleanup(d_list, d_world, d_camera, &fbTexture);
	cudaDeviceReset();
}

void KeyCallback(GLFWwindow *window, int key, int scancode, int action, int mode) {
	if (GLFW_KEY_ESCAPE == key && GLFW_PRESS == action) {
		glfwSetWindowShouldClose(window, GL_TRUE);
	}

}