#include "dxvk_color_space.glsl"

#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_samplerless_texture_functions : require

layout(constant_id = 0) const uint c_samples = 0u;
layout(constant_id = 1) const bool c_gamma = false;

layout(constant_id = 2) const uint c_src_color_space = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
layout(constant_id = 3) const bool c_src_is_srgb = true;
layout(constant_id = 4) const uint c_dst_color_space = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
layout(constant_id = 5) const bool c_dst_is_srgb = true;
layout(constant_id = 6) const bool c_composite_hud = false;
layout(constant_id = 7) const bool c_composite_cursor = false;
layout(constant_id = 8) const bool c_composite_ao = false;
layout(constant_id = 9) const bool c_ao_debug_view = false;  // show raw AO buffer

layout(set = 0, binding = 0) uniform sampler s_samplers[];

layout(set = 1, binding = 0) uniform texture2D s_image;
layout(set = 1, binding = 0) uniform texture2DMS s_image_ms;
layout(set = 1, binding = 1) uniform texture1D s_gamma;
layout(set = 1, binding = 2) uniform texture2D s_hud;
layout(set = 1, binding = 3) uniform texture2D s_cursor;
layout(set = 1, binding = 4) uniform texture2D s_ao;

layout(push_constant)
uniform present_info_t {
  ivec2 src_offset;
  ivec2 src_extent;
  ivec2 dst_offset;
  ivec2 cursor_offset;
  ivec2 cursor_extent;
  uint sampler_gamma;
  uint sampler_cursor;
  float ao_dark_mask;
  float ao_contact_t0;
  float ao_contact_t1;
  float ao_contact_strength;
};


vec4 blend_sc_rgb(vec4 dst, vec4 src) {
  return mix(dst, vec4(src.rgb, 1.0f), src.aaaa);
}


vec4 blend_linear_sdr(vec4 dst, vec4 src) {
  src.rgb = nits_to_sc_rgb(src.rgb * SDR_NITS);
  return blend_sc_rgb(dst, src);
}


vec4 composite_image(vec4 color, ivec2 srcCoord) {
  ivec2 coord = ivec2(gl_FragCoord.xy);

  if (c_composite_ao) {
    // The AO buffer is allocated at the SOURCE (back buffer) size, so it must
    // be sampled with the SOURCE pixel coordinate, not the destination
    // gl_FragCoord: under present scaling (e.g. a 1920x1058 game frame
    // presented into a 1920x1080 swap chain) gl_FragCoord-based sampling
    // shifts the AO vertically by up to the letterbox height, and the
    // out-of-source rows (the black bars) fetch AO for scene rows, which
    // shows up as a static strip glued to the screen edge.
    float ao = 1.0f;
    if (all(greaterThanEqual(srcCoord, ivec2(0)))
     && all(lessThan(srcCoord, src_extent)))
      ao = texelFetch(s_ao, srcCoord, 0).r;
    if (c_ao_debug_view)
      color.rgb = vec3(ao);  // show raw AO greyscale, skip HUD/cursor
    else {
      // Baked-lighting compensation (UE2 lightmaps already contain AO):
      // scale down the AO effect where the scene is already dark so that
      // baked shadow areas are not darkened twice. Full effect on lit pixels.
      float lum  = dot(color.rgb, vec3(0.2126, 0.7152, 0.0722));
      float mask = 1.0 - ao_dark_mask * (1.0 - min(lum * 3.0, 1.0));
      float aoApplied = mix(1.0, ao, mask);  // CACAO visibility: 1.0 = lit, 0.0 = occluded

      // Contact mode: apply AO only where occlusion is significant.
      // Open ground stays untouched (no global wash); wall bases, crevices
      // and object contacts get full-strength localized shadows.
      if (ao_contact_strength > 0.0) {
        float occ     = 1.0 - aoApplied;
        float contact = smoothstep(ao_contact_t0, ao_contact_t1, occ)
                      * ao_contact_strength;
        color.rgb *= 1.0 - contact;
      } else {
        color.rgb *= aoApplied;
      }
    }
  }

  if (c_composite_hud)
    color = blend_linear_sdr(color, texelFetch(s_hud, coord, 0));

  if (c_composite_cursor) {
    ivec2 rel_ofs = coord - cursor_offset;

    if (max(rel_ofs.x, rel_ofs.y) >= 0 && all(lessThan(rel_ofs, cursor_extent))) {
      color = blend_linear_sdr(color,
        texture(sampler2D(s_cursor, s_samplers[sampler_cursor]),
        vec2(rel_ofs) / vec2(cursor_extent)));
    }
  }

  return color;
}


vec4 input_to_sc_rgb(vec4 color) {
  switch (c_src_color_space) {
    default:
    case VK_COLOR_SPACE_PASS_THROUGH_EXT:
      return color;

    case VK_COLOR_SPACE_SRGB_NONLINEAR_KHR: {
      if (!c_src_is_srgb)
        color.rgb = srgb_to_linear(color.rgb);

      color.rgb = nits_to_sc_rgb(color.rgb * SDR_NITS);
      return color;
    }

    case VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT:
      return color;

    case VK_COLOR_SPACE_HDR10_ST2084_EXT:
      color.rgb = nits_to_sc_rgb(pq_to_nits(color.rgb));
      color.rgb = rec2020_to_rec709 * color.rgb;
      return color;
  }
}


vec4 sc_rgb_to_output(vec4 color) {
  if (c_gamma) {
    // If we need to apply a gamma curve, convert to sRGB, perform
    // the lookup, and then convert back to scRGB as necessary
    if (c_src_color_space != VK_COLOR_SPACE_PASS_THROUGH_EXT) {
      color.rgb = sc_rgb_to_nits(color.rgb) / SDR_NITS;
      color.rgb = linear_to_srgb(color.rgb);
    } else if (c_src_is_srgb) {
      color.rgb = linear_to_srgb(color.rgb);
    }

    color.rgb = vec3(
      texture(sampler1D(s_gamma, s_samplers[sampler_gamma]), color.r).r,
      texture(sampler1D(s_gamma, s_samplers[sampler_gamma]), color.g).g,
      texture(sampler1D(s_gamma, s_samplers[sampler_gamma]), color.b).b);

    if (c_dst_color_space != VK_COLOR_SPACE_PASS_THROUGH_EXT) {
      color.rgb = srgb_to_linear(color.rgb);
      color.rgb = nits_to_sc_rgb(color.rgb * SDR_NITS);
    } else if (c_dst_is_srgb) {
      color.rgb = srgb_to_linear(color.rgb);
    }
  }

  switch (c_dst_color_space) {
    default:
    case VK_COLOR_SPACE_PASS_THROUGH_EXT:
      // If we applied a gamma curve, the output is already correct
      if (!c_gamma && c_src_is_srgb != c_dst_is_srgb) {
        color.rgb = c_src_is_srgb
          ? linear_to_srgb(color.rgb)
          : srgb_to_linear(color.rgb);
      }

      return color;

    case VK_COLOR_SPACE_SRGB_NONLINEAR_KHR: {
      color.rgb = sc_rgb_to_nits(color.rgb) / SDR_NITS;

      if (!c_dst_is_srgb)
        color.rgb = linear_to_srgb(color.rgb);

      return color;
    }

    case VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT:
      return color;

    case VK_COLOR_SPACE_HDR10_ST2084_EXT:
      color.rgb = rec709_to_rec2020 * color.rgb;
      color.rgb = nits_to_pq(sc_rgb_to_nits(color.rgb));
      return color;
  }
}
