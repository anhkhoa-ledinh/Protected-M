#version 330 core

uniform vec3 mistColor;
uniform float mistOpacity;

in vec2 vUV;
out vec4 fragColor;

void main() {
    vec2 centered = vUV - 0.5;
    float dist = length(centered) * 1.4;
    float falloff = 1.0 - smoothstep(0.0, 1.0, dist);
    fragColor = vec4(mistColor, mistOpacity * falloff);
}
