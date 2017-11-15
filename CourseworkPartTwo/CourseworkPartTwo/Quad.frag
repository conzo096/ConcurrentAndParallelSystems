#version 440

// Interpolated values from the vertex shaders
in vec2 UV;
in vec4 particlecolor;

// Ouput data
out vec4 color;

uniform sampler2D tex;

void main()
{
	color =  texture(tex,UV) * particlecolor;
	//color.a = 1;
}