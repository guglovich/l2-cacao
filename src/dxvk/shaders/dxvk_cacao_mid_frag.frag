#version 450

#extension GL_EXT_samplerless_texture_functions : require
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_nonuniform_qualifier : require

#include "dxvk_color_space.glsl"

layout(set = 0, binding = 0) uniform sampler s_samplers[];
layout(set = 1, binding = 0) uniform texture2D s_scene;
layout(set = 1, binding = 1) uniform texture2D s_ao;

layout(push_constant) uniform push_block {
  ivec2 srcExtent;
  uint  samplerScene;
  float darkMask;
  float contactT0;
  float contactT1;
  float contactStrength;
  float debugView;
};

layout(location = 0) in vec2 i_coord;
layout(location = 0) out vec4 o_color;

void main() {
  ivec2 pixel = clamp(ivec2(i_coord * vec2(srcExtent)), ivec2(0), srcExtent - 1);
  vec2 uv = (vec2(pixel) + 0.5) / vec2(srcExtent);

  vec4 scene = texture(sampler2D(s_scene, s_samplers[samplerScene]), uv);
  float ao = texelFetch(s_ao, pixel, 0).r;

  if (debugView > 0.5) {
    o_color = vec4(ao, ao, ao, 1.0);
    return;
  }

  // Same composite math as the present blitter, performed in linear space
  // for exact visual parity (the back buffer is gamma-encoded).
  vec3 lin = srgb_to_linear(scene.rgb);
  float lum = dot(lin, vec3(0.2126, 0.7152, 0.0722));
  float mask = 1.0 - darkMask * (1.0 - min(lum * 3.0, 1.0));
  float aoApplied = mix(1.0, ao, mask);

  if (contactStrength > 0.0) {
    float occ     = 1.0 - aoApplied;
    float contact = smoothstep(contactT0, contactT1, occ) * contactStrength;
    aoApplied = 1.0 - contact;
  }

  o_color = vec4(linear_to_srgb(lin * aoApplied), scene.a);
}
