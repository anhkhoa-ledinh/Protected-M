#version 330 core

in vec2 vUV;
out vec4 FragColor;

uniform sampler2D uVideo;
uniform float uAlphaThreshold;

void main() {
    vec4 c = texture(uVideo, vUV);
    // Skip only near-transparent fragments (fill/perf); the feathered alpha is
    // blended back-to-front so layer edges fade smoothly instead of hard-cutting.
    if (c.a < uAlphaThreshold) {
        discard;
    }
    FragColor = vec4(c.rgb, c.a);
}
