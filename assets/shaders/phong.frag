#version 330 core

in vec3 vWorldPos;
in vec3 vNormal;

uniform vec3 uBaseColor;
uniform vec3 uViewPos;
uniform vec3 uDirLightDir;
uniform vec3 uDirLightColor;
uniform vec3 uPointLightPos;
uniform vec3 uPointLightColor;
uniform float uShininess;
uniform float uAmbient;

out vec4 FragColor;

vec3 phongLight(
    vec3 lightDir,
    vec3 lightColor,
    vec3 normal,
    vec3 viewDir,
    vec3 baseColor
) {
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 reflectDir = reflect(-lightDir, normal);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), uShininess);
    return (uAmbient * baseColor)
         + (diff * baseColor * lightColor)
         + (spec * lightColor * 0.35);
}

void main() {
    vec3 normal = normalize(vNormal);
    vec3 viewDir = normalize(uViewPos - vWorldPos);

    vec3 dirDir = normalize(-uDirLightDir);
    vec3 color = phongLight(dirDir, uDirLightColor, normal, viewDir, uBaseColor);

    vec3 pointDir = normalize(uPointLightPos - vWorldPos);
    float distance = length(uPointLightPos - vWorldPos);
    float attenuation = 1.0 / (1.0 + 0.09 * distance + 0.032 * distance * distance);
    color += phongLight(pointDir, uPointLightColor * attenuation, normal, viewDir, uBaseColor);

    FragColor = vec4(color, 1.0);
}
