#include <iostream>
#include <ctime>
#include <math.h>
#include <cstdio>
#include <cstdlib>
#define GRAVITY -9.18
#define MAXPARTICLES 500

// A single particle instance.
struct Particle
{
	double mass;
	double forceX, forceY;
	double velocityX, velocityY;
	double positionX, positionY;
	// Colour of the particle.
	int r, g, b, a;

	// Add force of contacting particle to this particle.
	void AddForce(Particle& other)
	{
		double EPS = 3E4;      // softening parameter (just to avoid infinities)
		double dx = other.positionX - positionX;
		double dy = other.positionY - positionY;
		double dist = sqrt(dx*dx + dy*dy);

		double F = (GRAVITY * mass * other.mass) / (dist*dist + EPS*EPS);
		forceX += F * dx / dist;
		forceY += F * dy / dist;
	}

	// Update the particle
	void Update(double deltaTime)
	{
		velocityX += deltaTime * forceX / mass;
		velocityY += deltaTime * forceY / mass;
		positionX += deltaTime * forceX;
		positionY += deltaTime * forceY;
	}
	
	void ResetForce()
	{
		forceX = 0;
		forceY = 0;
	}

	void PrintParticle()
	{
		std::printf("posX == %f posY == %f veloX == %f veloY == %f mass == %f\n", positionX, positionY, velocityX, velocityY, mass);
	}
};










void main()
{
	// Create particle array.
	Particle particles[MAXPARTICLES];
	// Populate array with particles moving in random positions.
	double x, y;
	for (int i = 0; i < MAXPARTICLES; i++)
	{


	}

}