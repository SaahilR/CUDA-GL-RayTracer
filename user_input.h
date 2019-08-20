#ifndef USERINPUTH
#define USERINPUTH

#include "GLFW\glfw3.h"
#include "camera.h"

class user_input {
public:
	bool keys[1024];

	user_input() {}

	void ProcessInputs(GLfloat deltaTime, camera cam);
	void ProcessKeyboard(GLfloat deltaTime, camera cam);
	void ProcessMouse(GLfloat dletaTime, camera cam);

};

void user_input::ProcessInputs(GLfloat deltaTime, camera cam) {
	ProcessKeyboard(deltaTime, cam);
	ProcessMouse(deltaTime, cam);
}

void user_input::ProcessKeyboard(GLfloat deltaTime, camera cam) {
	float speed = deltaTime * 1.0f;
	if (keys[GLFW_KEY_W] || keys[GLFW_KEY_UP]) {
		cam.setOrigin(cam.getOrigin() + cam.getLook() * speed);
	}
	if (keys[GLFW_KEY_S] || keys[GLFW_KEY_DOWN]) {
		cam.setOrigin(cam.getOrigin() - cam.getLook() * speed);
	}
	if (keys[GLFW_KEY_A] || keys[GLFW_KEY_LEFT]) {

	}
	if (keys[GLFW_KEY_D] || keys[GLFW_KEY_RIGHT]) {

	}
	if (keys[GLFW_KEY_SPACE]) {
		cam.setOrigin(cam.getOrigin() + vec3(0.0f, 1.0f, 0.0) * speed);
	}
	if (keys[GLFW_KEY_LEFT_SHIFT]) {
		cam.setOrigin(cam.getOrigin() - vec3(0.0f, 1.0f, 0.0) * speed);
	}
}

void user_input::ProcessMouse(GLfloat deltaTime, camera cam) {

}

#endif 



