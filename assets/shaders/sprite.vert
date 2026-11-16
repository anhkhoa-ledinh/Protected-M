#version 330 core

// Bo: quad LUON QUAY MAT VE CAMERA. Dung goc quad o khong gian view (camera
// nhin theo -z), nen chi can cong offset vao vi tri tam da bien doi.
layout(location = 0) in vec2 aCorner;
layout(location = 1) in vec2 aUV;

uniform mat4 uView;
uniform mat4 uProjection;
uniform vec3 uCenter;
uniform float uSize;

out vec2 vUV;

void main() {
    vUV = aUV;
    vec4 centre_view = uView * vec4(uCenter, 1.0);
    centre_view.xy += aCorner * uSize;
    gl_Position = uProjection * centre_view;
}
