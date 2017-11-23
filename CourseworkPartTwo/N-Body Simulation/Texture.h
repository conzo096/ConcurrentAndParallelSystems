#pragma once
#include <GLFW\glfw3.h>
#include "stb_image.h"
class Texture
{
public:
	// The OpenGL ID of the texture data
	GLuint id = 0;
	// The width of the texture
	int width = 0;
	// The height of the texture
	int height = 0;
	GLenum type;


	Texture() {}

	Texture(GLuint w, GLuint h)
	{
		// Initialise texture with OpenGL
		glGenTextures(1, &id);
		type = GL_TEXTURE_2D;
		width = w;
		height = h;
	}



	Texture(const char* loc)
	{
		glGenTextures(1, &id);
		type = GL_TEXTURE_2D;
		glBindTexture(type, id);
		// set the texture wrapping parameters
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);	// set texture wrapping to GL_REPEAT (default wrapping method)
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
		// set texture filtering parameters
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		// load image, create texture and generate mipmaps
		int nrChannels;
		stbi_set_flip_vertically_on_load(true); // tell stb_image.h to flip loaded texture's on the y-axis.
		unsigned char *data = stbi_load(loc, &width, &height, &nrChannels, 0);
		if (data)
		{
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
			glGenerateMipmap(GL_TEXTURE_2D);
		}
		else
		{
			std::cout << "Failed to load texture" << std::endl;
		}
		stbi_image_free(data);
	}
};