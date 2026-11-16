#version 330 core

in vec2 vUV;
out vec4 FragColor;

uniform sampler2D uTexture;
uniform vec4 uTint;
// TASK_bugs_look_0130: rgb + do manh nguy trang. a=0 giu nguyen pixel sprite
// hom nay (mac dinh moi lan goi khong truyen uCamo).
uniform vec4 uCamo;

void main() {
    vec4 texel = texture(uTexture, vUV) * uTint;
    // Bo pixel gan nhu trong suot de khong ghi depth cho vung rong cua sprite.
    if (texel.a < 0.02) discard;
    // TASK_bug_art_0530: tron phang (mix thang voi mau nguy trang) bien ca
    // con bo thanh MOT KHOI MAU PHANG. Nhan mau nguy trang theo do sang cua
    // chinh pixel sprite thi van giu duoc net/bong cua hoa tiet got (giu
    // "shading") thay vi xoa sach no.
    float luma = dot(texel.rgb, vec3(0.299, 0.587, 0.114));
    vec3 camo = uCamo.rgb * (0.55 + 0.9 * luma);
    texel.rgb = mix(texel.rgb, camo, uCamo.a);
    FragColor = texel;
}
