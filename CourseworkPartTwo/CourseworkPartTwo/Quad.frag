#version 440

// Interpolated values from the vertex shaders
in vec2 UV;
in vec4 particlecolor;

// Ouput data
out vec4 color;

uniform sampler2D myTextureSampler;

void main()
{
	color = vec4(255,0,0,1);
	color.a = 1;
}