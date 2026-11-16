#version 330 core

// Giao dien album: mot quad co texture nhan voi mot mau. Hinh to mau phang
// dung texture 1x1 trang, nen chi can MOT shader cho ca anh, chu va khung.
in vec2 vUV;
out vec4 FragColor;

uniform sampler2D uTexture;
uniform vec4 uTint;

void main() {
    FragColor = texture(uTexture, vUV) * uTint;
}
