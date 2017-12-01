#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include <iostream>
#include <vector>
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
#include<random>
#include<cmath>
#include<chrono>
#include <iostream>
#include <ctime>
#include <math.h>
#include <algorithm>
#include "Texture.h"
#include <omp.h>
#include <thread>
#include <curand.h>
#include <curand_kernel.h>

// Number of particles to be generated.
#define MAXPARTICLES 10000
#define THREADSPERBLOCK 64
std::string filePath("CudaUpd1000.csv");
// Gravational constant
#define G 6.673e-3 //6.673e-11;

// Get the number of threads this hardware can support.
int numThreads = std::thread::hardware_concurrency();
// Texture to make the particle a round shape.
Texture tex;
// User controlled camera - Use WASD.
Camera camera;
// Simple shader which renders billboarded particles.
GLShader shader;
GLFWwindow* window;

// Uniform locations for shader.
GLuint CameraRight_worldspace_ID;
GLuint CameraUp_worldspace_ID;
GLuint ViewProjMatrixID;

int nBlocks = (MAXPARTICLES + THREADSPERBLOCK - 1) / THREADSPERBLOCK;

void cuda_info()
{
	int device;
	cudaGetDevice(&device);

	cudaDeviceProp properties;
	cudaGetDeviceProperties(&properties, device);

	std::cout << "Name: " << properties.name << std::endl;
	std::cout << "CUDA Capability: " << properties.major <<  "." << properties.minor << std::endl;
	std::cout << "Cores: " << properties.multiProcessorCount << std::endl;
	std::cout << "Memory: : " << properties.totalGlobalMem / (1024*1024) << "MB" << std::endl;
	std::cout << "Clock freq: " << properties.clockRate/1000 << "MHz" << std::endl;
}


// This class represents the particle.
struct Particle
{
	// Position of the shader.
	glm::vec3 pos;
	// Force acting on the particle - is reset each update.
	glm::vec3 force;
	// Colour of the particle.
	unsigned char r, g, b, a;
	// Size of the particle - does not represent mass!
	float size;
	// How far away from the camera is it? This is used to sort the particle list on how close it is to camera.
	float cameradistance;
	// Speed and direction of the particle.
	glm::vec3 velocity;
	float mass;

	// Caclulate the force that another particle is having on this particle.
	void AddForce(Particle& b)
	{
		// Get the distance between the two particles.
		float dist = glm::distance(pos, b.pos);
		// Add a condition to prevent Nan - Is there a better approach to this? 
		if (dist == 0)
			dist = 0.000001;
		float F = G * (mass * b.mass / (dist*dist));
		force += F * (b.pos - pos) / dist;
	}

	// Update this particle. 
	void Update(float deltaTime)
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
};

Particle ParticlesContainer[MAXPARTICLES];
static GLfloat* g_particule_position_size_data = new GLfloat[MAXPARTICLES * 4];
static GLubyte* g_particule_color_data = new GLubyte[MAXPARTICLES * 4];
GLuint particles_position_buffer;
GLuint particles_color_buffer;
GLuint billboard_vertex_buffer;

Particle *buffer_particles;
auto data_size = sizeof(Particle) * MAXPARTICLES;

std::ofstream myfile;
__device__ void random(float* result, int MAX,int) {
	/* CUDA's random number library uses curandState_t to keep track of the seed value
	we will store a random state for every thread  */
	curandState_t state;

	/* we have to initialize the state */
	curand_init(0, /* the seed controls the sequence of random values that are produced */
		0, /* the sequence number is only important with multiple cores */
		0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
		&state);

	/* curand works like rand - except that it takes a state as a parameter */
	*result = curand(&state) % MAX;
}

__device__ void random(double* result)
{
	curandState_t state;
	curand_init(0,0, 0, &state);
	/* curand works like rand - except that it takes a state as a parameter */
	*result = curand(&state);
}
__device__ void random(unsigned char* result, int MAX,int i) {
	/* CUDA's random number library uses curandState_t to keep track of the seed value
	we will store a random state for every thread  */
	curandState_t state;

	/* we have to initialize the state */
	curand_init(i, /* the seed controls the sequence of random values that are produced */
		0, /* the sequence number is only important with multiple cores */
		0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
		&state);

	/* curand works like rand - except that it takes a state as a parameter */
	*result = curand(&state) % MAX;
}

__global__ void LoadParticlesGPU(Particle* particles, int radius,int numParticles)
{
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

	float x = (i* (numParticles/radius));
	float y = (i + (numParticles / radius));
	float z = (i * (numParticles / radius));

	particles[i].pos = glm::vec3(x,y,z);
	particles[i].velocity = glm::dvec3(0);
	random(&particles[i].r, 256,i);
	random(&particles[i].g, 256, i);
	random(&particles[i].b, 256, i);
	particles[i].a = 1;
	random(&particles[i].mass, 26,i);

	particles[i].mass =10;
	particles[i].size = 5;


}

//Updates particles based on their total force.
__global__ void SimulateParticlesGPU(Particle* particles, int numParticles)
{
	//unsigned int i = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned int j = blockIdx.x*blockDim.x + threadIdx.x;
	for (int i = 0; i<numParticles; i++)
	{
		if (i != j)
		{
			// Get the distance between the two particles.
			float dist = glm::distance(particles[j].pos, particles[i].pos);
			// Add a condition to prevent Nan - Is there a better approach to this? 
			if (dist < particles[i].size*particles[j].size)
				dist = particles[i].size*particles[j].size;
			float F = G * (particles[j].mass * particles[i].mass / (dist*dist));
			particles[j].force += F * (particles[i].pos - particles[j].pos) / dist;
		}
	}
}

//Updates particles based on their total force.
__global__ void UpdateParticlesGPU(Particle* particles,float deltaTime)
{
	unsigned int j = blockIdx.x*blockDim.x + threadIdx.x;
	particles[j].velocity += (particles[j].force / particles[j].mass);
	particles[j].pos += deltaTime * particles[j].velocity;
	particles[j].force = glm::dvec3(0);
}




// Sort the particle order by closest to camera.
void SortParticles()
{
	for (int i = 0; i < MAXPARTICLES; i++)
	{
		ParticlesContainer[i].cameradistance = glm::distance(camera.GetPosition(), ParticlesContainer[i].pos);
	}
	std::sort(&ParticlesContainer[0], &ParticlesContainer[MAXPARTICLES]);
}


// Update each particle, against all other particles in the scene.
void SimulateParticles()
{
	int i;
	#pragma omp parallel for num_threads(numThreads)  private(i)
	for (i = 0; i<MAXPARTICLES; i++)
	{
		// Get particle and reset its current force.
		Particle& p = ParticlesContainer[i];
		p.ResetForce();
		for (int j = 0; j < MAXPARTICLES; j++)
		{
			// Update particle as long as it is not itself.
			if (i != j)
			{
				p.AddForce(ParticlesContainer[j]);
			}
		}
	}
}


// Calculate the new position of all the particles, depending on the force applied to them.
void UpdateParticles(double deltaTime)
{
	//	#pragma omp parallel for num_threads(numThreads)  private(i)
	for (int i = 0; i < MAXPARTICLES; i++)
	{
		Particle& p = ParticlesContainer[i];
		// Update position of particle.
		p.Update(deltaTime);
		// calculate camera distance.
		p.cameradistance = glm::length2(p.pos - camera.GetPosition());
		// Update GPU buffer with new positions.
		g_particule_position_size_data[4 * i + 0] = p.pos.x;
		g_particule_position_size_data[4 * i + 1] = p.pos.y;
		g_particule_position_size_data[4 * i + 2] = p.pos.z;
		g_particule_position_size_data[4 * i + 3] = p.size;

		//// Update GPU buffer with colour positions.
		//g_particule_color_data[4 * i + 0] = p.r;
		//g_particule_color_data[4 * i + 1] = p.g;
		//g_particule_color_data[4 * i + 2] = p.b;
		//g_particule_color_data[4 * i + 3] = p.a;

	}

	// Now for each particle, update their position and velocity.
	for (int i = 0; i < MAXPARTICLES; i++)
	{
		g_particule_position_size_data[4 * i + 0] = ParticlesContainer[i].pos.x;
		g_particule_position_size_data[4 * i + 1] = ParticlesContainer[i].pos.y;
		g_particule_position_size_data[4 * i + 2] = ParticlesContainer[i].pos.z;
		g_particule_position_size_data[4 * i + 3] = ParticlesContainer[i].size;
	}
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
	SimulateParticlesGPU << <nBlocks, THREADSPERBLOCK >> > (buffer_particles, MAXPARTICLES);
	cudaDeviceSynchronize();
	UpdateParticlesGPU << <nBlocks, THREADSPERBLOCK >> >(buffer_particles, deltaTime);
	cudaDeviceSynchronize();
	// Copy required data back to their buffers.
	cudaMemcpy(&ParticlesContainer[0], buffer_particles, data_size, cudaMemcpyDeviceToHost);
	//UpdateParticles(deltaTime);

	SortParticles();
	// Now for each particle, update their position and velocity.
	for (int i = 0; i < MAXPARTICLES; i++)
	{
		g_particule_position_size_data[4 * i + 0] = ParticlesContainer[i].pos.x;
		g_particule_position_size_data[4 * i + 1] = ParticlesContainer[i].pos.y;
		g_particule_position_size_data[4 * i + 2] = ParticlesContainer[i].pos.z;
		g_particule_position_size_data[4 * i + 3] = ParticlesContainer[i].size;

		g_particule_color_data[4 * i + 0] = ParticlesContainer[i].r;
		g_particule_color_data[4 * i + 1] = ParticlesContainer[i].g;
		g_particule_color_data[4 * i + 2] = ParticlesContainer[i].b;
		g_particule_color_data[4 * i + 3] = ParticlesContainer[i].a;
	}
}


void Render()
{
	//SortParticles();

	// Update the OpenGL buffers with updated particle positions.
	glBindBuffer(GL_ARRAY_BUFFER, particles_position_buffer);
	glBufferData(GL_ARRAY_BUFFER, MAXPARTICLES * 4 * sizeof(GLfloat), NULL, GL_STREAM_DRAW);
	glBufferSubData(GL_ARRAY_BUFFER, 0, MAXPARTICLES * sizeof(GLfloat) * 4, g_particule_position_size_data);

	glBindBuffer(GL_ARRAY_BUFFER, particles_color_buffer);
	glBufferData(GL_ARRAY_BUFFER, MAXPARTICLES * 4 * sizeof(GLubyte), NULL, GL_STREAM_DRAW);
	glBufferSubData(GL_ARRAY_BUFFER, 0, MAXPARTICLES * sizeof(GLubyte) * 4, g_particule_color_data);

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
	myfile.open(filePath);
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
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
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

	double radius = 100;        
	int i;
	#pragma omp parallel for num_threads(numThreads) private(i)
	for (i = 1; i < MAXPARTICLES; i++)
	{
		double theta = 2 * glm::pi<double>() * uniform01(generator);
		double phi = acos(1 - 2 * uniform01(generator));
		double x = sin(phi) * cos(theta) * radius;
		double y = sin(phi) * sin(theta) * radius;
		double z = cos(phi) * radius;

		ParticlesContainer[i].pos = glm::dvec3(x, y, z);
		ParticlesContainer[i].velocity = glm::dvec3(0);
		ParticlesContainer[i].r = rand() % 256;
		ParticlesContainer[i].g = rand() % 256;
		ParticlesContainer[i].b = rand() % 256;
		ParticlesContainer[i].a = 255;
		ParticlesContainer[i].mass = rand() % 26 + 10;
		ParticlesContainer[i].size = 5;

		// Update GPU buffer with colour positions.
		g_particule_color_data[4 * i + 0] = ParticlesContainer[i].r;
		g_particule_color_data[4 * i + 1] = ParticlesContainer[i].g;
		g_particule_color_data[4 * i + 2] = ParticlesContainer[i].b;
		g_particule_color_data[4 * i + 3] = ParticlesContainer[i].a;
	}

	//Put the central mass in
	ParticlesContainer[0].pos = glm::dvec3(0, 0, 0);
	ParticlesContainer[0].velocity = glm::dvec3(0, 0, 0);
	ParticlesContainer[0].mass = 1;
	ParticlesContainer[0].r = 255;
	ParticlesContainer[0].g = 0;
	ParticlesContainer[0].b = 0;
	ParticlesContainer[0].a = 255;
	ParticlesContainer[0].size = (rand() % 1000) / 2000.0f + 0.1f;

	tex = Texture("circle.png");

	// Configuring CUDA.
	cudaSetDevice(0);
	cuda_info();
	// Create host memory.
	cudaMalloc((void**)&buffer_particles, data_size);
	
	// Send data to GPU.
	cudaMemcpy(buffer_particles, &ParticlesContainer[0], data_size, cudaMemcpyHostToDevice);
	//LoadParticlesGPU<<<2, MAXPARTICLES / 2 >>>(buffer_particles, radius, MAXPARTICLES);
	//cudaDeviceSynchronize();
	//cudaMemcpy(ParticlesContainer, &buffer_particles[0], data_size, cudaMemcpyDeviceToHost);
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
	delete[] g_particule_color_data;
	// Cleanup VBO and shader
	glDeleteBuffers(1, &particles_color_buffer);
	glDeleteBuffers(1, &particles_position_buffer);
	glDeleteBuffers(1, &billboard_vertex_buffer);
	glDeleteProgram(shader.GetId());
	glDeleteVertexArrays(1, &VertexArrayID);


	cudaFree(buffer_particles);
	// Close OpenGL window and terminate GLFW
	glfwTerminate();
	myfile.close();
	return 0;
}
