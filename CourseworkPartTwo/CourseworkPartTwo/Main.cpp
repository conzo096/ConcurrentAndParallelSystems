// A single particle instance.
//struct Particle
//{
//	double mass;
//	double forceX, forceY;
//	double velocityX, velocityY;
//	double positionX, positionY;
//	double size = 100;
//	// Colour of the particle.
//	int r, g, b, a;
//
//	// Add force of contacting particle to this particle.
//	void AddForce(Particle& other)
//	{
//		double EPS = 3E4;      // softening parameter (just to avoid infinities)
//		double dx = other.positionX - positionX;
//		double dy = other.positionY - positionY;
//		double dist = sqrt(dx*dx + dy*dy);
//
//		double F = (GRAVITY * mass * other.mass) / (dist*dist + EPS*EPS);
//		forceX += F * dx / dist;
//		forceY += F * dy / dist;
//	}
//
//	// Update the particle
//	void Update(double deltaTime)
//	{
//		//velocityX += deltaTime * forceX / mass;
//		//velocityY += deltaTime * forceY / mass;
//		//positionX += deltaTime * forceX;
//		//positionY += deltaTime * forceY;
//
//		velocityX = 0;
//		velocityY  = 0;
//		positionX = 0;
//		positionY = 0;
//
//	}
//	
//	void ResetForce()
//	{
//		forceX = 0;
//		forceY = 0;
//	}
//
//	void PrintParticle()
//	{
//		std::printf("posX == %f posY == %f \n veloX == %f veloY == %f mass == %f \n", positionX, positionY, velocityX, velocityY, mass);
//
//	}
//};


#include "GLShader.h"
#include <GL\GLU.h>
#include <GLFW\glfw3.h>
#include <chrono>
#include <glm\gtc\type_ptr.hpp>
#include <glm\gtc\matrix_transform.hpp>
#include <glm\common.hpp>
#include <glm\gtx\norm.hpp>
#include "Camera.h"
#include <fstream>
#include "Main.h"
#include <iostream>
#include <ctime>
#include <math.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>

#define MAXPARTICLES 200

Camera camera;
GLShader shader;
GLFWwindow* window;


// CPU representation of a particle
struct Particle
{
	glm::vec3 pos, speed;
	unsigned char r, g, b, a; // Color
	float size, angle, weight;
	float life; // Remaining life of the particle. if <0 : dead and unused.
	float cameradistance; // *Squared* distance to the camera. if dead : -1.0f

	bool operator<(const Particle& that) const
	{
		// Sort in reverse order : far particles drawn first.
		return this->cameradistance > that.cameradistance;
	}
};

Particle ParticlesContainer[MAXPARTICLES];
int LastUsedParticle = 0;

// Finds a Particle in ParticlesContainer which isn't used yet.
// (i.e. life < 0);
int FindUnusedParticle() {

	for (int i = LastUsedParticle; i<MAXPARTICLES; i++) {
		if (ParticlesContainer[i].life < 0) {
			LastUsedParticle = i;
			return i;
		}
	}

	for (int i = 0; i<LastUsedParticle; i++) {
		if (ParticlesContainer[i].life < 0) {
			LastUsedParticle = i;
			return i;
		}
	}

	return 0; // All particles are taken, override the first one
}

void SortParticles() {
	std::sort(&ParticlesContainer[0], &ParticlesContainer[MAXPARTICLES]);
}

int main(void)
{
	// Initialise GLFW
	if (!glfwInit())
	{
		fprintf(stderr, "Failed to initialize GLFW\n");
		getchar();
		return -1;
	}

	glfwWindowHint(GLFW_SAMPLES, 4);
	glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // To make MacOS happy; should not be needed
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	// Open a window and create its OpenGL context
	window = glfwCreateWindow(1024, 768, "Tutorial 18 - Particules", NULL, NULL);
	if (window == NULL) {
		fprintf(stderr, "Failed to open GLFW window. If you have an Intel GPU, they are not 3.3 compatible. Try the 2.1 version of the tutorials.\n");
		getchar();
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window);

	// Initialize GLEW
	glewExperimental = true; // Needed for core profile
	if (glewInit() != GLEW_OK) {
		fprintf(stderr, "Failed to initialize GLEW\n");
		getchar();
		glfwTerminate();
		return -1;
	}

	// Ensure we can capture the escape key being pressed below
	glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);
	// Hide the mouse and enable unlimited mouvement
	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

	// Set the mouse at the center of the screen
	glfwPollEvents();
	glfwSetCursorPos(window, 1024 / 2, 768 / 2);

	// Dark blue background
	glClearColor(0.0f, 0.0f, 0.4f, 0.0f);

	// Enable depth test
	glEnable(GL_DEPTH_TEST);
	// Accept fragment if it closer to the camera than the former one
	glDepthFunc(GL_LESS);

	GLuint VertexArrayID;
	glGenVertexArrays(1, &VertexArrayID);
	glBindVertexArray(VertexArrayID);


	// Create and compile our GLSL program from the shaders
	shader.SetProgram();
	shader.AddShaderFromFile("Quad.vert", GLShader::VERTEX);
	shader.AddShaderFromFile("Quad.frag", GLShader::FRAGMENT);
	shader.Link();


	camera.SetProjection(glm::quarter_pi<float>(), 1920 / 1080, 2.414f, 1000);
	camera.SetWindow(window);
	camera.SetPosition(glm::vec3(0, 0, 20));

	// Vertex shader
	GLuint CameraRight_worldspace_ID = glGetUniformLocation(shader.GetId(), "CameraRight_worldspace");
	GLuint CameraUp_worldspace_ID = glGetUniformLocation(shader.GetId(), "CameraUp_worldspace");
	GLuint ViewProjMatrixID = glGetUniformLocation(shader.GetId(), "VP");


	static GLfloat* g_particule_position_size_data = new GLfloat[MAXPARTICLES * 4];
	static GLubyte* g_particule_color_data = new GLubyte[MAXPARTICLES * 4];

	for (int i = 0; i<MAXPARTICLES; i++) {
		ParticlesContainer[i].life = -1.0f;
		ParticlesContainer[i].cameradistance = -1.0f;
	}

	//GLuint Texture = loadDDS("particle.DDS");

	// The VBO containing the 4 vertices of the particles.
	// Thanks to instancing, they will be shared by all particles.
	static const GLfloat g_vertex_buffer_data[] =
	{
		-0.5f, -0.5f, 0.0f,
		0.5f, -0.5f, 0.0f,
		-0.5f,  0.5f, 0.0f,
		0.5f,  0.5f, 0.0f,
	};
	GLuint billboard_vertex_buffer;
	glGenBuffers(1, &billboard_vertex_buffer);
	glBindBuffer(GL_ARRAY_BUFFER, billboard_vertex_buffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(g_vertex_buffer_data), g_vertex_buffer_data, GL_STATIC_DRAW);

	// The VBO containing the positions and sizes of the particles
	GLuint particles_position_buffer;
	glGenBuffers(1, &particles_position_buffer);
	glBindBuffer(GL_ARRAY_BUFFER, particles_position_buffer);
	// Initialize with empty (NULL) buffer : it will be updated later, each frame.
	glBufferData(GL_ARRAY_BUFFER, MAXPARTICLES * 4 * sizeof(GLfloat), NULL, GL_STREAM_DRAW);

	// The VBO containing the colors of the particles
	GLuint particles_color_buffer;
	glGenBuffers(1, &particles_color_buffer);
	glBindBuffer(GL_ARRAY_BUFFER, particles_color_buffer);
	// Initialize with empty (NULL) buffer : it will be updated later, each frame.
	glBufferData(GL_ARRAY_BUFFER, MAXPARTICLES * 4 * sizeof(GLubyte), NULL, GL_STREAM_DRAW);



	double lastTime = glfwGetTime();
	do
	{
		// Clear the screen
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		double currentTime = glfwGetTime();
		double delta = currentTime - lastTime;
		lastTime = currentTime;

		float ratio_width = glm::quarter_pi<float>() / static_cast<float>(1920);
		float ratio_height = glm::quarter_pi<float>() / static_cast<float>(1080);

		double xpos, ypos;
		glfwGetCursorPos(window, &xpos, &ypos);
		glfwSetCursorPos(window, 1920.0 / 2, 1080.0 / 2);
		// Calculate delta of cursor positions from last frame
		double delta_x = xpos - 1920.0 / 2;
		double delta_y = ypos - 1080.0 / 2;
		// Multiply deltas by ratios - gets actual change in orientation
		delta_x *= ratio_width;
		delta_y *= ratio_height;
		camera.Rotate(static_cast<float>(delta_x), static_cast<float>(-delta_y)); // flipped y to revert the invert.
		camera.Update(delta);
		glm::mat4 ProjectionMatrix = camera.GetProjection();
		glm::mat4 ViewMatrix = camera.GetView();

		// We will need the camera's position in order to sort the particles
		// w.r.t the camera's distance.
		// There should be a getCameraPosition() function in common/controls.cpp, 
		// but this works too.
		glm::vec3 CameraPosition(glm::inverse(ViewMatrix)[3]);

		glm::mat4 ViewProjectionMatrix = ProjectionMatrix * ViewMatrix;


		// Generate 10 new particule each millisecond,
		// but limit this to 16 ms (60 fps), or if you have 1 long frame (1sec),
		// newparticles will be huge and the next frame even longer.
		int newparticles = (int)(delta*10000.0);
		if (newparticles > (int)(0.016f*10000.0))
			newparticles = (int)(0.016f*10000.0);

		for (int i = 0; i<newparticles; i++) {
			int particleIndex = FindUnusedParticle();
			ParticlesContainer[particleIndex].life = 5.0f; // This particle will live 5 seconds.
			ParticlesContainer[particleIndex].pos = glm::vec3(0, 0, -20.0f);

			float spread = 1.5f;
			glm::vec3 maindir = glm::vec3(0.0f, 10.0f, 0.0f);
			// Very bad way to generate a random direction; 
			// See for instance http://stackoverflow.com/questions/5408276/python-uniform-spherical-distribution instead,
			// combined with some user-controlled parameters (main direction, spread, etc)
			glm::vec3 randomdir = glm::vec3(
				(rand() % 2000 - 1000.0f) / 1000.0f,
				(rand() % 2000 - 1000.0f) / 1000.0f,
				(rand() % 2000 - 1000.0f) / 1000.0f
			);

			ParticlesContainer[particleIndex].speed = maindir + randomdir*spread;


			// Very bad way to generate a random color
			ParticlesContainer[particleIndex].r = rand() % 256;
			ParticlesContainer[particleIndex].g = rand() % 256;
			ParticlesContainer[particleIndex].b = rand() % 256;
			ParticlesContainer[particleIndex].a = (rand() % 256) / 3;

			ParticlesContainer[particleIndex].size = (rand() % 1000) / 2000.0f + 0.1f;

		}
		// Simulate all particles
		int ParticlesCount = 0;
		for (int i = 0; i<MAXPARTICLES; i++)
		{
			Particle& p = ParticlesContainer[i]; // shortcut
			if (p.life > 0.0f)
			{
				// Decrease life
				p.life -= delta;
				if (p.life > 0.0f)
				{
					// Simulate simple physics : gravity only, no collisions
					p.speed += glm::vec3(0.0f, -9.81f, 0.0f) * (float)delta * 0.5f;
					p.pos += p.speed * (float)delta;
					p.cameradistance = glm::length2(p.pos - CameraPosition);
					//ParticlesContainer[i].pos += glm::vec3(0.0f,10.0f, 0.0f) * (float)delta;

					// Fill the GPU buffer
					g_particule_position_size_data[4 * ParticlesCount + 0] = p.pos.x;
					g_particule_position_size_data[4 * ParticlesCount + 1] = p.pos.y;
					g_particule_position_size_data[4 * ParticlesCount + 2] = p.pos.z;

					g_particule_position_size_data[4 * ParticlesCount + 3] = p.size;

					g_particule_color_data[4 * ParticlesCount + 0] = p.r;
					g_particule_color_data[4 * ParticlesCount + 1] = p.g;
					g_particule_color_data[4 * ParticlesCount + 2] = p.b;
					g_particule_color_data[4 * ParticlesCount + 3] = p.a;
				}
				else
				{
					// Particles that just died will be put at the end of the buffer in SortParticles();
					p.cameradistance = -1.0f;
				}
				ParticlesCount++;
			}
		}

		SortParticles();
		glBindBuffer(GL_ARRAY_BUFFER, particles_position_buffer);
		glBufferData(GL_ARRAY_BUFFER, MAXPARTICLES * 4 * sizeof(GLfloat), NULL, GL_STREAM_DRAW);
		glBufferSubData(GL_ARRAY_BUFFER, 0, ParticlesCount * sizeof(GLfloat) * 4, g_particule_position_size_data);

		glBindBuffer(GL_ARRAY_BUFFER, particles_color_buffer);
		glBufferData(GL_ARRAY_BUFFER, MAXPARTICLES * 4 * sizeof(GLubyte), NULL, GL_STREAM_DRAW);
		glBufferSubData(GL_ARRAY_BUFFER, 0, ParticlesCount * sizeof(GLubyte) * 4, g_particule_color_data);
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

		// Use our shader
		shader.Use();
		glUniform3f(CameraRight_worldspace_ID, ViewMatrix[0][0], ViewMatrix[1][0], ViewMatrix[2][0]);
		glUniform3f(CameraUp_worldspace_ID, ViewMatrix[0][1], ViewMatrix[1][1], ViewMatrix[2][1]);

		glUniformMatrix4fv(ViewProjMatrixID, 1, GL_FALSE, &ViewProjectionMatrix[0][0]);

		
		glEnableVertexAttribArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, billboard_vertex_buffer);
		glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,0,(void*)0);

		// 2nd attribute buffer : positions of particles' centers
		glEnableVertexAttribArray(1);
		glBindBuffer(GL_ARRAY_BUFFER, particles_position_buffer);
		glVertexAttribPointer(1,4,GL_FLOAT,GL_FALSE,0,(void*)0);

		// 3rd attribute buffer : particles' colors
		glEnableVertexAttribArray(2);
		glBindBuffer(GL_ARRAY_BUFFER, particles_color_buffer);
		glVertexAttribPointer(2,4,GL_UNSIGNED_BYTE,GL_TRUE,0,(void*)0);

		
		glVertexAttribDivisor(0, 0);
		glVertexAttribDivisor(1, 1); 
		glVertexAttribDivisor(2, 1);
		glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, ParticlesCount);

		glDisableVertexAttribArray(0);
		glDisableVertexAttribArray(1);
		glDisableVertexAttribArray(2);

		// Swap buffers
		glfwSwapBuffers(window);
		glfwPollEvents();

	} // Check if the ESC key was pressed or the window was closed
	while (glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS &&
		glfwWindowShouldClose(window) == 0);


	delete[] g_particule_position_size_data;

	// Cleanup VBO and shader
	glDeleteBuffers(1, &particles_color_buffer);
	glDeleteBuffers(1, &particles_position_buffer);
	glDeleteBuffers(1, &billboard_vertex_buffer);
	glDeleteProgram(shader.GetId());
	glDeleteVertexArrays(1, &VertexArrayID);


	// Close OpenGL window and terminate GLFW
	glfwTerminate();

	return 0;
}

