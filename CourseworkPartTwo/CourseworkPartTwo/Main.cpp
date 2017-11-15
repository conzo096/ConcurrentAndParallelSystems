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
#include<random>
#include<cmath>
#include<chrono>
#include <iostream>
#include <ctime>
#include <math.h>
#include <algorithm>
#include "Texture.h"
#include <glm\gtx\string_cast.hpp>
#define MAXPARTICLES 10

Texture tex;
Camera camera;
GLShader shader;
GLFWwindow* window;
GLuint CameraRight_worldspace_ID;
GLuint CameraUp_worldspace_ID;
GLuint ViewProjMatrixID;
// CPU representation of a particle
struct Particle
{
	glm::dvec3 pos;
	glm::dvec3 force;
	unsigned char r, g, b, a; 
	float size;
	float cameradistance; 
	glm::dvec3 velocity;     
	double mass;     

	void AddForce(Particle& b)
	{
		double dist = glm::distance(pos, b.pos);
		// Nan will appear if dist = 0;
		if (dist == 0)
			dist = 0.0001;
		//6.673e-11
		double F = (6.673e-11) * (mass * b.mass / (dist*dist));
		force += F*glm::normalize(b.pos-pos);
	}

	void Update(double deltaTime)
	{
		velocity += (force / mass);
		pos += deltaTime * velocity;
	}

	void ResetForce()
	{
		force = glm::dvec3(0);
	}
	bool operator<(const Particle& that) const
	{
		// Sort in reverse order : far particles drawn first.
		return this->cameradistance > that.cameradistance;
	}

	void Print()
	{
		std::cout << glm::to_string(pos) << std::endl;
	}
};

Particle ParticlesContainer[MAXPARTICLES];
static GLfloat* g_particule_position_size_data = new GLfloat[MAXPARTICLES * 4];
static GLubyte* g_particule_color_data = new GLubyte[MAXPARTICLES * 4];
GLuint particles_position_buffer;
GLuint particles_color_buffer;
GLuint billboard_vertex_buffer;
int LastUsedParticle = 0;


void SortParticles()
{
	std::sort(&ParticlesContainer[0], &ParticlesContainer[MAXPARTICLES]);
}


void Update(double deltaTime)
{
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
	camera.Update(deltaTime);
	
	
	// Simulate all particles

	for (int i = 0; i<MAXPARTICLES; i++)
	{
		Particle& p = ParticlesContainer[i]; 
		p.ResetForce();
		for (int j = 0; j < MAXPARTICLES; j++)
		{
			if (i != j)
			{
				p.AddForce(ParticlesContainer[j]);
			}
		}
	}
	for (int i = 0; i < MAXPARTICLES; i++)
	{
		Particle& p = ParticlesContainer[i]; // shortcut
		p.Update(1e11* deltaTime);
		p.cameradistance = glm::length2(p.pos - glm::dvec3(camera.GetPosition()));
		p.Print();
		// Fill the GPU buffer
		g_particule_position_size_data[4 * i + 0] = p.pos.x;
		g_particule_position_size_data[4 * i + 1] = p.pos.y;
		g_particule_position_size_data[4 * i + 2] = p.pos.z;

		g_particule_position_size_data[4 * i + 3] = p.size;

		g_particule_color_data[4 * i + 0] = p.r;
		g_particule_color_data[4 * i + 1] = p.g;
		g_particule_color_data[4 * i + 2] = p.b;
		g_particule_color_data[4 * i + 3] = p.a;
	}
	SortParticles();

	// Move this outta update.
	glBindBuffer(GL_ARRAY_BUFFER, particles_position_buffer);
	glBufferData(GL_ARRAY_BUFFER, MAXPARTICLES * 4 * sizeof(GLfloat), NULL, GL_STREAM_DRAW);
	glBufferSubData(GL_ARRAY_BUFFER, 0, MAXPARTICLES * sizeof(GLfloat) * 4, g_particule_position_size_data);

	glBindBuffer(GL_ARRAY_BUFFER, particles_color_buffer);
	glBufferData(GL_ARRAY_BUFFER, MAXPARTICLES * 4 * sizeof(GLubyte), NULL, GL_STREAM_DRAW);
	glBufferSubData(GL_ARRAY_BUFFER, 0, MAXPARTICLES * sizeof(GLubyte) * 4, g_particule_color_data);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
}


void Render()
{
	// Clear the screen
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
//	glClearColor(1, 1, 1, 1);
	glm::mat4 ProjectionMatrix = camera.GetProjection();
	glm::mat4 ViewMatrix = camera.GetView();
	glm::mat4 ViewProjectionMatrix = ProjectionMatrix * ViewMatrix;

	// Use our shader
	shader.Use();
	glUniform3f(CameraRight_worldspace_ID, ViewMatrix[0][0], ViewMatrix[1][0], ViewMatrix[2][0]);
	glUniform3f(CameraUp_worldspace_ID, ViewMatrix[0][1], ViewMatrix[1][1], ViewMatrix[2][1]);
	glUniformMatrix4fv(ViewProjMatrixID, 1, GL_FALSE, &ViewProjectionMatrix[0][0]);


	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, tex.id);
	glUniform1i(glGetUniformLocation(shader.GetId(), "tex"), 1);


	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, billboard_vertex_buffer);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);

	// 2nd attribute buffer : positions of particles' centers
	glEnableVertexAttribArray(1);
	glBindBuffer(GL_ARRAY_BUFFER, particles_position_buffer);
	glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, (void*)0);

	// 3rd attribute buffer : particles' colors
	glEnableVertexAttribArray(2);
	glBindBuffer(GL_ARRAY_BUFFER, particles_color_buffer);
	glVertexAttribPointer(2, 4, GL_UNSIGNED_BYTE, GL_TRUE, 0, (void*)0);


	glVertexAttribDivisor(0, 0);
	glVertexAttribDivisor(1, 1);
	glVertexAttribDivisor(2, 1);
	glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, MAXPARTICLES);

	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);
	glDisableVertexAttribArray(2);

	// Swap buffers
	glfwSwapBuffers(window);
	glfwPollEvents();
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
	window = glfwCreateWindow(1920, 1080, "N-Body Simulation", NULL, NULL);
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
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	GLuint VertexArrayID;
	glGenVertexArrays(1, &VertexArrayID);
	glBindVertexArray(VertexArrayID);


	// Create and compile our GLSL program from the shaders
	shader.SetProgram();
	shader.AddShaderFromFile("Quad.vert", GLShader::VERTEX);
	shader.AddShaderFromFile("Quad.frag", GLShader::FRAGMENT);
	shader.Link();


	camera.SetProjection(glm::quarter_pi<float>(), 1920 / 1080, 2.414f, 100000);
	camera.SetWindow(window);
	camera.SetPosition(glm::vec3(0, 0, 20));

	// Vertex shader
	CameraRight_worldspace_ID = glGetUniformLocation(shader.GetId(), "CameraRight_worldspace");
	CameraUp_worldspace_ID = glGetUniformLocation(shader.GetId(), "CameraUp_worldspace");
	ViewProjMatrixID = glGetUniformLocation(shader.GetId(), "VP");

	double lastTime = glfwGetTime();

	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::mt19937 generator(seed);
	std::uniform_real_distribution<double> uniform01(0.0, 1.0);

	double radius = 1e18;        // radius of universe
	double solarmass = 1.98892e30;
	for (int i = 1; i < MAXPARTICLES; i++)
	{
		double theta = 2 * glm::pi<double>() * uniform01(generator);
		double phi = acos(1 - 2 * uniform01(generator));
		double x = sin(phi) * cos(theta) * radius;
		double y = sin(phi) * sin(theta) * radius;
		double z = cos(phi) * radius;
		
		ParticlesContainer[i].pos = glm::dvec3(x,y,z);
		ParticlesContainer[i].velocity = glm::dvec3(0,0,0);
		ParticlesContainer[i].r = rand() % 256;
		ParticlesContainer[i].g = rand() % 256;
		ParticlesContainer[i].b = rand() % 256;
		ParticlesContainer[i].a = 255;
		ParticlesContainer[i].mass = rand()*solarmass * 10 + 1e20;
		ParticlesContainer[i].size = 5;
	}
	//Put the central mass in
	ParticlesContainer[0].pos = glm::dvec3(0, 0, 0);
	ParticlesContainer[0].velocity = glm::dvec3(0, 0, 0);
	ParticlesContainer[0].mass = 1e6*solarmass;
	ParticlesContainer[0].r = 255;
	ParticlesContainer[0].g = 0;
	ParticlesContainer[0].b = 0;
	ParticlesContainer[0].a = 255;
	ParticlesContainer[0].size = (rand() % 1000) / 2000.0f + 0.1f;
	tex = Texture("circle.png");

	//camera.SetPosition(ParticlesContainer[0].pos);

	// The VBO containing the 4 vertices of the particles.
	// Thanks to instancing, they will be shared by all particles.
	static const GLfloat g_vertex_buffer_data[] =
	{
		-0.5f, -0.5f, 0.0f,
		0.5f, -0.5f, 0.0f,
		-0.5f,  0.5f, 0.0f,
		0.5f,  0.5f, 0.0f,
	};
	glGenBuffers(1, &billboard_vertex_buffer);
	glBindBuffer(GL_ARRAY_BUFFER, billboard_vertex_buffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(g_vertex_buffer_data), g_vertex_buffer_data, GL_STATIC_DRAW);

	// The VBO containing the positions and sizes of the particles
	glGenBuffers(1, &particles_position_buffer);
	glBindBuffer(GL_ARRAY_BUFFER, particles_position_buffer);
	// Initialize with empty (NULL) buffer : it will be updated later, each frame.
	glBufferData(GL_ARRAY_BUFFER, MAXPARTICLES * 4 * sizeof(GLfloat), NULL, GL_STREAM_DRAW);

	// The VBO containing the colors of the particles
	glGenBuffers(1, &particles_color_buffer);
	glBindBuffer(GL_ARRAY_BUFFER, particles_color_buffer);
	// Initialize with empty (NULL) buffer : it will be updated later, each frame.
	glBufferData(GL_ARRAY_BUFFER, MAXPARTICLES * 4 * sizeof(GLubyte), NULL, GL_STREAM_DRAW);

	do
	{

		double currentTime = glfwGetTime();
		double delta = currentTime - lastTime;
		Update(delta);
		Render();
		lastTime = currentTime;

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

