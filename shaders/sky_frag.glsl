#version 450 core

// Sun/moon disc fragment: flat color, no lighting or fog.

layout(location = 0) in vec3 inColor;
layout(location = 1) in vec2 inUV; // unused (shared vertex layout)
layout(location = 0) out vec4 outColor;

void main() {
    outColor = vec4(inColor, 1.0);
}
