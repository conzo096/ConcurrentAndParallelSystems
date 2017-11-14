#pragma once
#include <glm\glm.hpp>
#include <glm\gtc\matrix_transform.hpp>
#include <glm\gtx\euler_angles.hpp>
#include <GLFW\glfw3.h>

// Camera to allow user to navigate around scene.
class Camera
{
private:

	// Window it is connected too.
	GLFWwindow* window;

	// Where the camera is in world space.
	glm::vec3 position = glm::vec3(0);
	// Where the camera is looking in world space.
	glm::vec3 target = glm::vec3(0,0,-1);

	// Up direction of the camera.
	glm::vec3 up = glm::vec3(0, 1,0);
	// translation of the camera since the last update.
	glm::vec3 translation;

	// x & y rotation.
	float pitch = 0, yaw = 0;
	
	// View and projection matrices. ( Required for shader manipulation).
	glm::mat4 view;
	glm::mat4 projection;


	void CheckForMovement(float deltaTime);
public:

	void SetPosition(glm::vec3 pos)
	{
		position = pos;
	}
	void SetTarget(glm::vec3 tar)
	{
		target = tar;
	}

	void SetWindow(GLFWwindow* &attachedWindow)
	{
		window = attachedWindow;
	}
	glm::mat4 GetView() { return view; }
	glm::mat4 GetProjection() { return projection; }
	glm::vec3 GetPosition() { return position; }
	void SetProjection(float fov, float aspectRatio, float nearPlane, float farPlane)
	{
		projection = glm::perspective(fov, aspectRatio, nearPlane, farPlane);
	}

	void Update(float update);

	void Rotate(float deltaYaw, float deltaPitch);


};