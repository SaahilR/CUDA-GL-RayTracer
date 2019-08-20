#ifndef CAMERAH
#define CAMERAH

#include "ray.h"

#define M_PI 3.141592653

class camera {
public:
	__device__ camera(vec3 origin, vec3 look, vec3 up, float vFov, float aspect) : vFov(vFov), aspect(aspect), origin(origin), look(look), up(up) {
		vec3 u, v, w;
		float theta = vFov * M_PI / 180;
		float half_height = tan(theta / 2);
		float half_width = aspect * half_height;

		w = unit_vector(origin - look);
		u = unit_vector(cross(up, w));
		v = cross(w, u);

		lower_left_corner = vec3(-half_width, -half_height, -1.0f);
		lower_left_corner = origin - half_width * u - half_height * v - w;
		horizontal = 2.0f * half_width * u;
		vertical = 2.0f * half_height * v;
	}
	__device__ ray get_ray(float u, float v) {
		return ray(origin, lower_left_corner + u * horizontal + v * vertical - origin);
	}

	__device__ void updateCamera() {
		vec3 u, v, w;

		float theta = vFov * M_PI / 180;
		float half_height = tan(theta / 2);
		float half_width = aspect * half_height;

		w = unit_vector(origin - look);
		u = unit_vector(cross(up, w));
		v = cross(w, u);

		lower_left_corner = vec3(-half_width, -half_height, -1.0f);
		lower_left_corner = origin - half_width * u - half_height * v - w;
		horizontal = 2.0f * half_width * u;
		vertical = 2.0f * half_height * v;
	}

	__device__ void setOrigin(vec3 newOrigin) { origin = newOrigin; }
	__device__ vec3 getOrigin() { return origin; }
	
	__device__ void setLook(vec3 newLook) { look = newLook; }
	__device__ vec3 getLook() { return look; }
	
	float vFov;
	float aspect;
	vec3 origin;
	vec3 look;
	vec3 up;
	vec3 lower_left_corner;
	vec3 horizontal;
	vec3 vertical;
};

#endif