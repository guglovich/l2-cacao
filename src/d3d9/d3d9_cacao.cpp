/**
 * d3d9_cacao.cpp
 * FidelityFX CACAO (Combined Adaptive Compute Ambient Occlusion)
 * integrated as a DXVK built-in post-process pass.
 *
 * Quality path: Q3 (non-adaptive) for High.
 * Resources are lazily allocated on first Apply() and reallocated if
 * the resolution changes.
 *
 * Binding layout (set=0) matches ffx_cacao_callbacks_glsl.h:
 *   0-8    : SRV textures
 *   1000-1004 : immutable samplers
 *   2000-2008 : UAV storage images / buffers
 *   3000   : constants UBO
 *
 * Because DXVK's createBuiltInPipelineLayout() assumes densely-packed
 * binding indices starting at 0, we bypass it entirely and use raw
 * Vulkan API (vkd()) for descriptor pools, layouts, and pipelines.
 * Compute dispatch is injected via D3D9DeviceEx::EmitCs() / DxvkContext::beginExternalRendering().
 *
 * Descriptor safety: sets are pre-allocated (double-buffered, kFrameLag=2).
 * On each Apply() call the current frame's sets are updated with vkUpdateDescriptorSets
 * BEFORE recording commands that use them. With GPU latency ≤ 1 frame, the
 * previous usage of the same set (2 frames prior) is complete. ✓
 */

#include "d3d9_cacao.h"
#include "d3d9_device.h"
#include "d3d9_cacao_log.h"
#include "../dxvk/hud/dxvk_hud_cacao.h"

#include "../dxvk/dxvk_device.h"
#include "../dxvk/dxvk_context.h"
#include "../dxvk/dxvk_cmdlist.h"
#include "../dxvk/dxvk_image.h"
#include "../dxvk/dxvk_buffer.h"
#include "../dxvk/dxvk_util.h"

#include "cacao_spirv/dxvk_cacao_mid_vert_pass.h"
#include "cacao_spirv/dxvk_cacao_mid_frag_pass.h"

#include <cmath>
#include <cstring>
#include <algorithm>
#include <vector>
#include <cstdio>

// ── Compiled SPIR-V headers (compiled from FidelityFX SDK GLSL) ───────────────
#include "cacao_spirv/ffx_cacao_prepare_native_depths_and_mips_pass.h"
#include "cacao_spirv/ffx_cacao_prepare_native_normals_pass.h"
#include "cacao_spirv/ffx_cacao_generate_q0_pass.h"
#include "cacao_spirv/ffx_cacao_generate_q1_pass.h"
#include "cacao_spirv/ffx_cacao_generate_q2_pass.h"
#include "cacao_spirv/ffx_cacao_generate_q3_pass.h"
#include "cacao_spirv/ffx_cacao_generate_importance_map_pass.h"
#include "cacao_spirv/ffx_cacao_generate_importance_map_a_pass.h"
#include "cacao_spirv/ffx_cacao_generate_importance_map_b_pass.h"
#include "cacao_spirv/ffx_cacao_edge_sensitive_blur_8_pass.h"
#include "cacao_spirv/ffx_cacao_apply_pass.h"
#include "cacao_spirv/ffx_cacao_clear_load_counter_pass.h"

namespace dxvk {

// ── Dispatch size helper ──────────────────────────────────────────────────────
static inline uint32_t dispatchSize(uint32_t tileSize, uint32_t totalSize) {
  return (totalSize + tileSize - 1) / tileSize;
}

// ── Sampling pattern data (from FidelityFX SDK) ──────────────────────────────
static const float kPatternData[4][5][4] = {
  { // pass 0
    { 0.238946f,  0.971099f, 0.101341f, 0.0f },
    { 0.878523f,  0.477706f, 0.310988f, 0.0f },
    { 0.553760f, -0.832650f, 0.410492f, 0.0f },
    {-0.290014f,  0.957116f, 0.600498f, 0.0f },
    {-0.527154f, -0.849897f, 0.900124f, 0.0f },
  },
  { // pass 1
    { 0.741154f,  0.671310f, 0.101341f, 0.0f },
    {-0.249655f,  0.968374f, 0.310988f, 0.0f },
    {-0.522019f, -0.852977f, 0.410492f, 0.0f },
    { 0.984386f, -0.176126f, 0.600498f, 0.0f },
    { 0.055423f, -0.998463f, 0.900124f, 0.0f },
  },
  { // pass 2
    {-0.532827f,  0.846182f, 0.101341f, 0.0f },
    { 0.950377f, -0.311031f, 0.310988f, 0.0f },
    {-0.932081f, -0.362358f, 0.410492f, 0.0f },
    {-0.093048f, -0.995660f, 0.600498f, 0.0f },
    { 0.997260f,  0.073968f, 0.900124f, 0.0f },
  },
  { // pass 3
    { 0.208650f, -0.977996f, 0.101341f, 0.0f },
    {-0.927940f, -0.372724f, 0.310988f, 0.0f },
    { 0.485253f,  0.874353f, 0.410492f, 0.0f },
    {-0.951165f,  0.308768f, 0.600498f, 0.0f },
    {-0.248050f,  0.968691f, 0.900124f, 0.0f },
  },
};

// ── Buffer size info (mirrors FfxCacaoBufferSizeInfo) ────────────────────────
struct BufSizeInfo {
  uint32_t depthBufferWidth, depthBufferHeight;
  uint32_t ssaoBufferWidth,  ssaoBufferHeight;
  uint32_t deinterleavedDepthBufferWidth,  deinterleavedDepthBufferHeight;
  uint32_t importanceMapWidth, importanceMapHeight;
  uint32_t inputOutputBufferWidth, inputOutputBufferHeight;
};

static void calcBufSizeInfo(uint32_t w, uint32_t h, BufSizeInfo* bsi) {
  const uint32_t hw = (w + 1) / 2;
  const uint32_t hh = (h + 1) / 2;
  const uint32_t qw = (hw + 1) / 2;
  const uint32_t qh = (hh + 1) / 2;

  bsi->depthBufferWidth  = w;
  bsi->depthBufferHeight = h;
  bsi->ssaoBufferWidth   = hw;
  bsi->ssaoBufferHeight  = hh;
  bsi->deinterleavedDepthBufferWidth  = hw;
  bsi->deinterleavedDepthBufferHeight = hh;
  bsi->importanceMapWidth  = qw;
  bsi->importanceMapHeight = qh;
  bsi->inputOutputBufferWidth  = w;
  bsi->inputOutputBufferHeight = h;
}

// ── Binding slot constants (must match GLSL defines) ─────────────────────────
static constexpr uint32_t B_SRV_LOAD_COUNTER          =    0;
static constexpr uint32_t B_SRV_DEINTERLEAVED_DEPTHS  =    1;
static constexpr uint32_t B_SRV_DEINTERLEAVED_NORMALS =    2;
static constexpr uint32_t B_SRV_SSAO_PING             =    3;
static constexpr uint32_t B_SRV_IMPORTANCE_MAP        =    4;
static constexpr uint32_t B_SRV_DEPTH_IN              =    5;
static constexpr uint32_t B_SRV_IMPORTANCE_MAP_PONG   =    7;  // g_ImportanceMapPong sampled image (required by importance_map_a/b_pass SPIR-V)
static constexpr uint32_t B_SRV_SSAO_PING_APPLY       =    8;
static constexpr uint32_t B_SAMPLER_POINT_CLAMP       = 1000;
static constexpr uint32_t B_SAMPLER_POINT_MIRROR      = 1001;
static constexpr uint32_t B_SAMPLER_LINEAR_CLAMP      = 1002;
static constexpr uint32_t B_SAMPLER_VIEWSPACE_DEPTH   = 1003;
static constexpr uint32_t B_SAMPLER_REAL_POINT_CLAMP  = 1004;
static constexpr uint32_t B_UAV_SSAO_PING             = 2000;
static constexpr uint32_t B_UAV_DEPTH_MIPS            = 2001;
static constexpr uint32_t B_UAV_DEINT_DEPTHS          = 2002;
static constexpr uint32_t B_UAV_DEINT_NORMALS         = 2003;
static constexpr uint32_t B_UAV_LOAD_COUNTER          = 2004;
static constexpr uint32_t B_UAV_IMPORTANCE_MAP        = 2005;
static constexpr uint32_t B_UAV_IMPORTANCE_MAP_PONG   = 2006;
static constexpr uint32_t B_UAV_SSAO_PONG             = 2007;
static constexpr uint32_t B_UAV_OUTPUT                = 2008;
static constexpr uint32_t B_CB_CACAO                  = 3000;

// Set indices within a single frame's descriptor set block
static constexpr uint32_t SET_IDX_DEPTHS  = 0;
static constexpr uint32_t SET_IDX_NORMALS = 1;
static constexpr uint32_t SET_IDX_Q3      = 2;
static constexpr uint32_t SET_IDX_BLUR_0  = 3;
static constexpr uint32_t SET_IDX_BLUR_1  = 4;
static constexpr uint32_t SET_IDX_BLUR_2  = 5;
static constexpr uint32_t SET_IDX_BLUR_3  = 6;
static constexpr uint32_t SET_IDX_APPLY   = 7;
// (= kSetsPerFrame = 8)


// ── Constructor / destructor ──────────────────────────────────────────────────
D3D9CacaoRenderer::D3D9CacaoRenderer(DxvkDevice* device, CacaoQuality quality)
  : m_device(device), m_quality(quality) {
  try {
    InitPipelines();
    m_initialized = true;
    hud::g_cacaoActive.store(true);
    hud::g_cacaoEnabled.store(m_enabled);
    hud::g_cacaoQuality.store((int)m_quality);
    CacaoLogger::get().log("[INIT] CACAO pipelines initialized OK");
  } catch (const std::exception& e) {
    Logger::err(str::format("CACAO: pipeline init failed: ", e.what()));
    CacaoLogger::get().log(str::format("[INIT] CACAO pipeline init FAILED: ", e.what()));
  }
}

D3D9CacaoRenderer::~D3D9CacaoRenderer() {
  DestroyResources();

  auto vkd = m_device->vkd();
  VkDevice dev = vkd->device();

  if (m_descPool)
    vkd->vkDestroyDescriptorPool(dev, m_descPool, nullptr);

  for (auto& pl : m_pipelines)
    if (pl) vkd->vkDestroyPipeline(dev, pl, nullptr);
  for (auto& pl : m_pipelineLayouts)
    if (pl) vkd->vkDestroyPipelineLayout(dev, pl, nullptr);
  for (auto& sl : m_setLayouts)
    if (sl) vkd->vkDestroyDescriptorSetLayout(dev, sl, nullptr);
  for (auto& s : m_samplers)
    if (s) vkd->vkDestroySampler(dev, s, nullptr);
}

// ── Vulkan helpers ────────────────────────────────────────────────────────────

struct BindDef {
  uint32_t             binding;
  VkDescriptorType     type;
  uint32_t             count;
  VkShaderStageFlags   stage;
};

static VkDescriptorSetLayout makeSetLayout(
    const Rc<vk::DeviceFn>& vkd,
    const std::vector<BindDef>& defs,
    const VkSampler* immutableSamplers = nullptr,
    uint32_t         immutableSamplerBase = 0,
    uint32_t         immutableSamplerCount = 0)
{
  std::vector<VkDescriptorSetLayoutBinding> lb;
  lb.reserve(defs.size());
  for (auto& d : defs) {
    VkDescriptorSetLayoutBinding b = {};
    b.binding         = d.binding;
    b.descriptorType  = d.type;
    b.descriptorCount = d.count;
    b.stageFlags      = d.stage;
    if (immutableSamplers && d.type == VK_DESCRIPTOR_TYPE_SAMPLER
        && d.binding >= immutableSamplerBase
        && d.binding < immutableSamplerBase + immutableSamplerCount)
    {
      b.pImmutableSamplers = immutableSamplers + (d.binding - immutableSamplerBase);
    }
    lb.push_back(b);
  }

  VkDescriptorSetLayoutCreateInfo ci = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
  ci.bindingCount = uint32_t(lb.size());
  ci.pBindings    = lb.data();

  VkDescriptorSetLayout layout = VK_NULL_HANDLE;
  if (vkd->vkCreateDescriptorSetLayout(vkd->device(), &ci, nullptr, &layout) != VK_SUCCESS)
    throw DxvkError("CACAO: failed to create descriptor set layout");
  return layout;
}

static VkPipelineLayout makePipelineLayout(
    const Rc<vk::DeviceFn>& vkd,
    VkDescriptorSetLayout setLayout)
{
  VkPipelineLayoutCreateInfo ci = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
  ci.setLayoutCount = 1;
  ci.pSetLayouts    = &setLayout;

  VkPipelineLayout layout = VK_NULL_HANDLE;
  if (vkd->vkCreatePipelineLayout(vkd->device(), &ci, nullptr, &layout) != VK_SUCCESS)
    throw DxvkError("CACAO: failed to create pipeline layout");
  return layout;
}

static VkPipeline makeComputePipeline(
    const Rc<vk::DeviceFn>& vkd,
    VkPipelineLayout pipelineLayout,
    const uint32_t* spirv,
    size_t spirvBytes)
{
  VkShaderModuleCreateInfo smci = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
  smci.codeSize = spirvBytes;
  smci.pCode    = spirv;

  VkShaderModule shmod = VK_NULL_HANDLE;
  if (vkd->vkCreateShaderModule(vkd->device(), &smci, nullptr, &shmod) != VK_SUCCESS)
    throw DxvkError("CACAO: failed to create shader module");

  VkComputePipelineCreateInfo pci = { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
  pci.layout            = pipelineLayout;
  pci.basePipelineIndex = -1;
  pci.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  pci.stage.module = shmod;
  pci.stage.pName  = "main";
  pci.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;

  VkPipeline pipeline = VK_NULL_HANDLE;
  VkResult vr = vkd->vkCreateComputePipelines(vkd->device(),
    VK_NULL_HANDLE, 1, &pci, nullptr, &pipeline);

  vkd->vkDestroyShaderModule(vkd->device(), shmod, nullptr);

  if (vr != VK_SUCCESS)
    throw DxvkError("CACAO: failed to create compute pipeline");
  return pipeline;
}

static VkSampler makeSampler(
    const Rc<vk::DeviceFn>& vkd,
    VkFilter        filter,
    VkSamplerAddressMode addressMode,
    float           maxLod = 0.0f)
{
  VkSamplerCreateInfo sci = { VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
  sci.magFilter        = filter;
  sci.minFilter        = filter;
  sci.mipmapMode       = filter == VK_FILTER_LINEAR
                           ? VK_SAMPLER_MIPMAP_MODE_LINEAR
                           : VK_SAMPLER_MIPMAP_MODE_NEAREST;
  sci.addressModeU     = addressMode;
  sci.addressModeV     = addressMode;
  sci.addressModeW     = addressMode;
  sci.maxLod           = maxLod;

  VkSampler s = VK_NULL_HANDLE;
  if (vkd->vkCreateSampler(vkd->device(), &sci, nullptr, &s) != VK_SUCCESS)
    throw DxvkError("CACAO: failed to create sampler");
  return s;
}

// ── InitPipelines ─────────────────────────────────────────────────────────────
void D3D9CacaoRenderer::InitPipelines() {
  auto vkd = m_device->vkd();
  VkDevice dev = vkd->device();

  // ── Samplers ─────────────────────────────────────────────────────────────
  m_samplers[0] = makeSampler(vkd, VK_FILTER_NEAREST, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
  m_samplers[1] = makeSampler(vkd, VK_FILTER_NEAREST, VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT);
  m_samplers[2] = makeSampler(vkd, VK_FILTER_LINEAR,  VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, 1000.0f);
  m_samplers[3] = makeSampler(vkd, VK_FILTER_LINEAR,  VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
  m_samplers[4] = makeSampler(vkd, VK_FILTER_NEAREST, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);

  // ── Descriptor pool ───────────────────────────────────────────────────────
  // Pre-allocate kTotalSets sets for double-buffered per-frame use.
  const VkDescriptorPoolSize poolSizes[] = {
    { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  kTotalSets * 16 },
    { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  kTotalSets * 16 },
    { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, kTotalSets      },
    { VK_DESCRIPTOR_TYPE_SAMPLER,        kTotalSets * 5  },
  };
  VkDescriptorPoolCreateInfo dpi = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
  dpi.maxSets       = kTotalSets;
  dpi.poolSizeCount = uint32_t(std::size(poolSizes));
  dpi.pPoolSizes    = poolSizes;
  if (vkd->vkCreateDescriptorPool(dev, &dpi, nullptr, &m_descPool) != VK_SUCCESS)
    throw DxvkError("CACAO: failed to create descriptor pool");

  // ── Sampler bindings shared by all passes ─────────────────────────────────
  const std::vector<BindDef> samplerDefs = {
    { B_SAMPLER_POINT_CLAMP,      VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { B_SAMPLER_POINT_MIRROR,     VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { B_SAMPLER_LINEAR_CLAMP,     VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { B_SAMPLER_VIEWSPACE_DEPTH,  VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { B_SAMPLER_REAL_POINT_CLAMP, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT },
  };

  auto buildLayout = [&](std::vector<BindDef> passDefs) -> VkDescriptorSetLayout {
    for (auto& s : samplerDefs) passDefs.push_back(s);
    passDefs.push_back({ B_CB_CACAO, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT });
    return makeSetLayout(vkd, passDefs, m_samplers.data(), B_SAMPLER_POINT_CLAMP, 5);
  };

  // PASS_PREPARE_DEPTHS
  m_setLayouts[CACAO_PASS_PREPARE_DEPTHS] = buildLayout({
    { B_SRV_DEPTH_IN,     VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT },
    { B_UAV_DEPTH_MIPS,   VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  4, VK_SHADER_STAGE_COMPUTE_BIT },
    { B_UAV_DEINT_DEPTHS, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT },
  });

  // PASS_PREPARE_NORMALS
  m_setLayouts[CACAO_PASS_PREPARE_NORMALS] = buildLayout({
    { B_SRV_DEPTH_IN,       VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { B_UAV_DEINT_NORMALS,  VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
  });

  // PASS_GENERATE_Qx — all four generate variants (q0/q1/q2/q3 SPIR-V) share
  // the exact same descriptor interface (bindings 0-4 + UAV 2000 verified via
  // spirv-dis), so each pass gets its own identical layout object (separate
  // handles: the destructor destroys every slot).
  {
    std::vector<BindDef> genDefs = {
      { B_SRV_LOAD_COUNTER,          VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT },
      { B_SRV_DEINTERLEAVED_DEPTHS,  VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT },
      { B_SRV_DEINTERLEAVED_NORMALS, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT },
      { B_SRV_SSAO_PING,             VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT },
      { B_SRV_IMPORTANCE_MAP,        VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT },
      { B_UAV_SSAO_PING,             VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT },
    };
    m_setLayouts[CACAO_PASS_GENERATE_Q3] = buildLayout(genDefs);
    m_setLayouts[CACAO_PASS_GENERATE_Q0] = buildLayout(genDefs);
    m_setLayouts[CACAO_PASS_GENERATE_Q1] = buildLayout(genDefs);
    m_setLayouts[CACAO_PASS_GENERATE_Q2] = buildLayout(genDefs);
  }

  // PASS_IMPORTANCE_MAP — full bindings used by all three importance-map passes.
  // Create SEPARATE layout objects for A and B to avoid double-destroy in destructor.
  {
    std::vector<BindDef> impDefs = {
      { B_SRV_SSAO_PING,            VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT },
      { B_SRV_IMPORTANCE_MAP,       VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT },
      { B_SRV_IMPORTANCE_MAP_PONG,  VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT },  // binding 7: g_ImportanceMapPong
      { B_UAV_LOAD_COUNTER,         VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT },
      { B_UAV_IMPORTANCE_MAP,       VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT },
      { B_UAV_IMPORTANCE_MAP_PONG,  VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT },
    };
    m_setLayouts[CACAO_PASS_IMPORTANCE_MAP]   = buildLayout(impDefs);
    m_setLayouts[CACAO_PASS_IMPORTANCE_MAP_A] = buildLayout(impDefs);  // separate handle
    m_setLayouts[CACAO_PASS_IMPORTANCE_MAP_B] = buildLayout(impDefs);  // separate handle
  }

  // PASS_EDGE_BLUR: ping → pong
  m_setLayouts[CACAO_PASS_EDGE_BLUR] = buildLayout({
    { B_SRV_SSAO_PING_APPLY,  VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT },
    { B_UAV_SSAO_PONG,        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT },
  });

  // PASS_APPLY: ssao → aoFinal
  m_setLayouts[CACAO_PASS_APPLY] = buildLayout({
    { B_SRV_SSAO_PING_APPLY,  VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT },
    { B_UAV_OUTPUT,           VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT },
  });

  // ── Pipeline layouts (one per pass) ───────────────────────────────────────
  for (uint32_t i = 0; i < CACAO_PASS_COUNT; i++)
    m_pipelineLayouts[i] = makePipelineLayout(vkd, m_setLayouts[i]);

  // ── Compute pipelines ──────────────────────────────────────────────────────
#define MAKE_PIPELINE(pass, spv) \
  m_pipelines[pass] = makeComputePipeline(vkd, m_pipelineLayouts[pass], \
    spv, sizeof(spv))

  MAKE_PIPELINE(CACAO_PASS_PREPARE_DEPTHS,    FFX_CACAO_PREPARE_NATIVE_DEPTHS_AND_MIPS_PASS_SPV);
  MAKE_PIPELINE(CACAO_PASS_PREPARE_NORMALS,   FFX_CACAO_PREPARE_NATIVE_NORMALS_PASS_SPV);
  MAKE_PIPELINE(CACAO_PASS_GENERATE_Q3,       FFX_CACAO_GENERATE_Q3_PASS_SPV);
  MAKE_PIPELINE(CACAO_PASS_GENERATE_Q0,       FFX_CACAO_GENERATE_Q0_PASS_SPV);
  MAKE_PIPELINE(CACAO_PASS_GENERATE_Q1,       FFX_CACAO_GENERATE_Q1_PASS_SPV);
  MAKE_PIPELINE(CACAO_PASS_GENERATE_Q2,       FFX_CACAO_GENERATE_Q2_PASS_SPV);
  MAKE_PIPELINE(CACAO_PASS_IMPORTANCE_MAP,    FFX_CACAO_GENERATE_IMPORTANCE_MAP_PASS_SPV);
  MAKE_PIPELINE(CACAO_PASS_IMPORTANCE_MAP_A,  FFX_CACAO_GENERATE_IMPORTANCE_MAP_A_PASS_SPV);
  MAKE_PIPELINE(CACAO_PASS_IMPORTANCE_MAP_B,  FFX_CACAO_GENERATE_IMPORTANCE_MAP_B_PASS_SPV);
  MAKE_PIPELINE(CACAO_PASS_EDGE_BLUR,         FFX_CACAO_EDGE_SENSITIVE_BLUR_8_PASS_SPV);
  MAKE_PIPELINE(CACAO_PASS_APPLY,             FFX_CACAO_APPLY_PASS_SPV);

#undef MAKE_PIPELINE

  // ── Pre-allocate descriptor sets (double-buffered) ────────────────────────
  // Layout sequence per frame: DEPTHS, NORMALS, Q3, BLUR×4, APPLY
  VkDescriptorSetLayout frameLayouts[kSetsPerFrame] = {
    m_setLayouts[CACAO_PASS_PREPARE_DEPTHS],
    m_setLayouts[CACAO_PASS_PREPARE_NORMALS],
    m_setLayouts[CACAO_PASS_GENERATE_Q3],
    m_setLayouts[CACAO_PASS_EDGE_BLUR],
    m_setLayouts[CACAO_PASS_EDGE_BLUR],
    m_setLayouts[CACAO_PASS_EDGE_BLUR],
    m_setLayouts[CACAO_PASS_EDGE_BLUR],
    m_setLayouts[CACAO_PASS_APPLY],
  };

  VkDescriptorSetLayout allLayouts[kTotalSets];
  for (uint32_t f = 0; f < kFrameLag; f++)
    for (uint32_t s = 0; s < kSetsPerFrame; s++)
      allLayouts[f * kSetsPerFrame + s] = frameLayouts[s];

  VkDescriptorSetAllocateInfo ai = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
  ai.descriptorPool     = m_descPool;
  ai.descriptorSetCount = kTotalSets;
  ai.pSetLayouts        = allLayouts;
  if (vkd->vkAllocateDescriptorSets(dev, &ai, m_descSets.data()) != VK_SUCCESS)
    throw DxvkError("CACAO: failed to pre-allocate descriptor sets");
}

// ── InitResources ──────────────────────────────────────────────────────────────
void D3D9CacaoRenderer::InitResources(uint32_t w, uint32_t h) {
  DestroyResources();
  m_width  = w;
  m_height = h;

  BufSizeInfo bsi;
  calcBufSizeInfo(w, h, &bsi);

  const uint32_t hw = bsi.ssaoBufferWidth;
  const uint32_t hh = bsi.ssaoBufferHeight;
  const uint32_t qw = bsi.importanceMapWidth;
  const uint32_t qh = bsi.importanceMapHeight;

  m_deinterleavedDepths = CreateIntermediateImage(hw, hh,
    VK_FORMAT_R32_SFLOAT, 4, 4,
    VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
    | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);

  m_deinterleavedNormals = CreateIntermediateImage(hw, hh,
    VK_FORMAT_R32G32B32A32_SFLOAT, 4, 1,
    VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
    | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);

  m_ssaoPing = CreateIntermediateImage(hw, hh,
    VK_FORMAT_R32G32_SFLOAT, 4, 1,
    VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
    | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);

  m_ssaoPong = CreateIntermediateImage(hw, hh,
    VK_FORMAT_R32G32_SFLOAT, 4, 1,
    VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
    | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);

  m_importanceMap = CreateIntermediateImage(qw, qh,
    VK_FORMAT_R32_SFLOAT, 1, 1,
    VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);

  m_importanceMapPong = CreateIntermediateImage(qw, qh,
    VK_FORMAT_R32_SFLOAT, 1, 1,
    VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);

  // AO output: full-res, R16G16B16A16_SFLOAT (CACAO Apply writes AO into R)
  // Double-buffered to avoid cross-frame race conditions
  for (uint32_t i = 0; i < kFrameLag; i++) {
    m_aoFinals[i] = CreateIntermediateImage(w, h,
      VK_FORMAT_R16G16B16A16_SFLOAT, 1, 1,
      VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
      | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
  }

  // Persistent SAMPLED_IMAGE views of m_aoFinals for use in the present blitter
  // One view per frame to avoid layout transition conflicts
  {
    DxvkImageViewKey aoSrvKey = {};
    aoSrvKey.viewType   = VK_IMAGE_VIEW_TYPE_2D;
    aoSrvKey.usage      = VK_IMAGE_USAGE_SAMPLED_BIT;
    aoSrvKey.format     = VK_FORMAT_R16G16B16A16_SFLOAT;
    aoSrvKey.layout     = VK_IMAGE_LAYOUT_GENERAL;
    aoSrvKey.aspects    = VK_IMAGE_ASPECT_COLOR_BIT;
    aoSrvKey.mipIndex   = 0u;
    aoSrvKey.mipCount   = 1u;
    aoSrvKey.layerIndex = 0u;
    aoSrvKey.layerCount = 1u;
    for (uint32_t i = 0; i < kFrameLag; i++) {
      m_aoFinalSrvs[i] = m_aoFinals[i]->createView(aoSrvKey);
    }
  }

  // Load counter: 1D R32_UINT storage image (adaptive sampling counter)
  // Note: Must be 1D to match SPIR-V shader expectation (Dim = 1D)
  {
    DxvkImageCreateInfo ici = {};
    ici.type        = VK_IMAGE_TYPE_1D;
    ici.format      = VK_FORMAT_R32_UINT;
    ici.flags       = 0;
    ici.sampleCount = VK_SAMPLE_COUNT_1_BIT;
    ici.extent      = { 1, 1, 1 };
    ici.numLayers   = 1;
    ici.mipLevels   = 1;
    ici.usage       = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    ici.stages      = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    ici.access      = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    ici.layout      = VK_IMAGE_LAYOUT_GENERAL;
    ici.tiling      = VK_IMAGE_TILING_OPTIMAL;
    m_loadCounter = m_device->createImage(ici, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  }

  // Cached static views: depend only on our own intermediates, so they are
  // created once per resolution instead of ~15 times per Apply.
  {
    auto mkView = [](const Rc<DxvkImage>& img, VkFormat fmt, VkImageViewType type,
                     VkImageUsageFlagBits usage, uint32_t layers,
                     uint32_t baseMip = 0, uint32_t mipCnt = 1) {
      DxvkImageViewKey key = {};
      key.viewType   = type;
      key.usage      = usage;
      key.format     = fmt;
      key.layout     = VK_IMAGE_LAYOUT_GENERAL;
      key.aspects    = VK_IMAGE_ASPECT_COLOR_BIT;
      key.mipIndex   = uint8_t(baseMip);
      key.mipCount   = uint8_t(mipCnt);
      key.layerIndex = 0u;
      key.layerCount = uint16_t(layers);
      return img->createView(key);
    };

    m_vDeintDUav = mkView(m_deinterleavedDepths, VK_FORMAT_R32_SFLOAT,
      VK_IMAGE_VIEW_TYPE_2D_ARRAY, VK_IMAGE_USAGE_STORAGE_BIT, 4);
    m_vDeintDSrv = mkView(m_deinterleavedDepths, VK_FORMAT_R32_SFLOAT,
      VK_IMAGE_VIEW_TYPE_2D_ARRAY, VK_IMAGE_USAGE_SAMPLED_BIT, 4);
    for (uint32_t m = 0; m < 4; m++)
      m_vDeintDMips[m] = mkView(m_deinterleavedDepths, VK_FORMAT_R32_SFLOAT,
        VK_IMAGE_VIEW_TYPE_2D_ARRAY, VK_IMAGE_USAGE_STORAGE_BIT, 4, m, 1);
    m_vDeintNUav = mkView(m_deinterleavedNormals, VK_FORMAT_R32G32B32A32_SFLOAT,
      VK_IMAGE_VIEW_TYPE_2D_ARRAY, VK_IMAGE_USAGE_STORAGE_BIT, 4);
    m_vDeintNSrv = mkView(m_deinterleavedNormals, VK_FORMAT_R32G32B32A32_SFLOAT,
      VK_IMAGE_VIEW_TYPE_2D_ARRAY, VK_IMAGE_USAGE_SAMPLED_BIT, 4);
    m_vPingUav   = mkView(m_ssaoPing, VK_FORMAT_R32G32_SFLOAT,
      VK_IMAGE_VIEW_TYPE_2D_ARRAY, VK_IMAGE_USAGE_STORAGE_BIT, 4);
    m_vPingSrv   = mkView(m_ssaoPing, VK_FORMAT_R32G32_SFLOAT,
      VK_IMAGE_VIEW_TYPE_2D_ARRAY, VK_IMAGE_USAGE_SAMPLED_BIT, 4);
    m_vPongUav   = mkView(m_ssaoPong, VK_FORMAT_R32G32_SFLOAT,
      VK_IMAGE_VIEW_TYPE_2D_ARRAY, VK_IMAGE_USAGE_STORAGE_BIT, 4);
    m_vPongSrv   = mkView(m_ssaoPong, VK_FORMAT_R32G32_SFLOAT,
      VK_IMAGE_VIEW_TYPE_2D_ARRAY, VK_IMAGE_USAGE_SAMPLED_BIT, 4);
    m_vImpSrv    = mkView(m_importanceMap, VK_FORMAT_R32_SFLOAT,
      VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_USAGE_SAMPLED_BIT, 1);
    m_vLoadSrv   = mkView(m_loadCounter, VK_FORMAT_R32_UINT,
      VK_IMAGE_VIEW_TYPE_1D, VK_IMAGE_USAGE_SAMPLED_BIT, 1);
    for (uint32_t i = 0; i < kFrameLag; i++)
      m_vAoUavs[i] = mkView(m_aoFinals[i], VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_USAGE_STORAGE_BIT, 1);
  }

  // Constants UBO (double-buffered to avoid race conditions)
  DxvkBufferCreateInfo cbci = {};
  cbci.size   = sizeof(CacaoConstants);
  cbci.usage  = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
  cbci.stages = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
  cbci.access = VK_ACCESS_UNIFORM_READ_BIT;

  for (uint32_t i = 0; i < kFrameLag; i++) {
    m_constantBuffers[i] = m_device->createBuffer(cbci,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  }

  // Readback buffer: strip @0/16/64 + full-buffer dump regions:
  //   aoFinal 1920x1080 RGBA16F (16.6 MB)  @ 0
  //   deintD  layer0 960x540  R32F  (2.1 MB)  @ 16 MB
  //   ssao    layer0 960x540  RG32F (4.1 MB)  @ 19 MB
  DxvkBufferCreateInfo rbci = {};
  rbci.size   = 32u * 1024u * 1024u;
  rbci.usage  = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  rbci.stages = VK_PIPELINE_STAGE_TRANSFER_BIT;
  rbci.access = VK_ACCESS_TRANSFER_WRITE_BIT;
  m_readbackBuffer  = m_device->createBuffer(rbci,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  m_readbackPending = false;
}

// ── DestroyResources ───────────────────────────────────────────────────────────
void D3D9CacaoRenderer::DestroyResources() {
  DestroyMidComposite();
  m_deinterleavedDepths  = nullptr;
  m_deinterleavedNormals = nullptr;
  m_ssaoPing             = nullptr;
  m_ssaoPong             = nullptr;
  m_importanceMap        = nullptr;
  m_importanceMapPong    = nullptr;
  m_vDeintDUav = nullptr;
  m_vDeintDSrv = nullptr;
  m_vDeintNUav = nullptr;
  m_vDeintNSrv = nullptr;
  for (uint32_t m = 0; m < 4; m++)
    m_vDeintDMips[m] = nullptr;
  m_vPingSrv = nullptr;
  m_vPingUav = nullptr;
  m_vPongSrv = nullptr;
  m_vPongUav = nullptr;
  m_vImpSrv  = nullptr;
  m_vLoadSrv = nullptr;
  for (uint32_t i = 0; i < kFrameLag; i++) {
    m_aoFinalSrvs[i] = nullptr;
    m_vAoUavs[i]     = nullptr;
    m_aoFinals[i]    = nullptr;
  }
  m_loadCounter          = nullptr;
  for (uint32_t i = 0; i < kFrameLag; i++) {
    m_constantBuffers[i] = nullptr;
  }
  m_readbackBuffer       = nullptr;
  m_readbackPending      = false;
  m_width  = 0;
  m_height = 0;
  m_frameIndex = 0;
}

// ── CreateIntermediateImage ────────────────────────────────────────────────────
Rc<DxvkImage> D3D9CacaoRenderer::CreateIntermediateImage(
    uint32_t w, uint32_t h,
    VkFormat format,
    uint32_t layers,
    uint32_t mips,
    VkImageUsageFlags extraUsage)
{
  DxvkImageCreateInfo ici = {};
  ici.type        = VK_IMAGE_TYPE_2D;
  ici.format      = format;
  ici.flags       = 0;
  ici.sampleCount = VK_SAMPLE_COUNT_1_BIT;
  ici.extent      = { w, h, 1 };
  ici.numLayers   = layers;
  ici.mipLevels   = mips;
  ici.usage       = extraUsage;
  ici.stages      = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
  ici.access      = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
  ici.layout      = VK_IMAGE_LAYOUT_GENERAL;
  ici.tiling      = VK_IMAGE_TILING_OPTIMAL;

  return m_device->createImage(ici, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
}

// ── UpdateConstants ────────────────────────────────────────────────────────────
void D3D9CacaoRenderer::UpdateConstants(
    CacaoConstants* c,
    const float*    proj,  // 16 floats, row-major (D3D9 convention)
    uint32_t        w,
    uint32_t        h)
{
  BufSizeInfo bsi;
  calcBufSizeInfo(w, h, &bsi);

  std::memset(c, 0, sizeof(*c));

  // Row-major D3D9 projection: proj[row*4+col]
  const float depthLinearizeMul = -proj[14];
  float       depthLinearizeAdd =  proj[10];
  if (depthLinearizeMul * depthLinearizeAdd < 0.0f)
    depthLinearizeAdd = -depthLinearizeAdd;
  // D3D9 depth convention: 0 = near, 1 = far. DXVK stores the raw D3D9 depth
  // unmodified (no reversed-depth viewport for D3D9 — verified: no min/maxDepth
  // flip in d3d9 viewport setup, and Clear writes 1.0 = far). viewZ(d) =
  // mul / (add - d) with mul = near*far/(far-near), add = far/(far-near).
  //
  // HISTORY (don't re-flip): a 06.09 build flipped these constants believing
  // DXVK stores D3D9 depth reversed ("empty gave viewZ=near"). That observation
  // came from an uninitialized buffer. The 29.10 full-buffer dump proved it
  // wrong: a REAL scene read raw depth 0.995 uniformly (= viewZ ~1900 units,
  // the login backdrop) — with the flipped constants that scene was unpacked
  // to viewZ ~10 (at the near plane), normals became garbage, SSAO ~0 and the
  // composite went black on every frame that had real depth content.
  c->DepthUnpackConsts[0] = depthLinearizeMul;
  c->DepthUnpackConsts[1] = depthLinearizeAdd;

  const float tanHalfFOVX = 1.0f / proj[0];
  const float tanHalfFOVY = 1.0f / proj[5];
  c->CameraTanHalfFOV[0] = tanHalfFOVX;
  c->CameraTanHalfFOV[1] = tanHalfFOVY;

  // NDCToView: matches ffx_cacao.cpp SDK reference exactly.
  // NDCToViewMul[1] is negative: UV.y=0 (top) maps to positive view-space Y (up).
  // D3D9 NDC Y: +1 top, -1 bottom. View-space Y: +1 up. SDK uses -2*t / +t pair.
  c->NDCToViewMul[0] =  tanHalfFOVX *  2.0f;
  c->NDCToViewMul[1] =  tanHalfFOVY * -2.0f;  // SDK: CameraTanHalfFOV[1] * -2.0f
  c->NDCToViewAdd[0] = -tanHalfFOVX;
  c->NDCToViewAdd[1] =  tanHalfFOVY;           // SDK: CameraTanHalfFOV[1] * 1.0f

  const float ratio  = 1.0f;
  const float border = 0.0f;
  c->DepthBufferUVToViewMul[0] = c->NDCToViewMul[0] / ratio;
  c->DepthBufferUVToViewMul[1] = c->NDCToViewMul[1] / ratio;
  c->DepthBufferUVToViewAdd[0] = c->NDCToViewAdd[0] - c->NDCToViewMul[0] * border / ratio;
  c->DepthBufferUVToViewAdd[1] = c->NDCToViewAdd[1] - c->NDCToViewMul[1] * border / ratio;

  // worldRadius: scale m_radius (config multiplier) by zNear extracted from projection.
  // Guard against degenerate projection (depthLinearizeMul == 0) to avoid div-by-zero below.
  const float worldRadius = (depthLinearizeMul > 0.0f)
      ? m_radius * depthLinearizeMul
      : m_radius;  // fallback: use raw radius if projection is degenerate
  m_worldRadiusScale = depthLinearizeMul;  // diagnostics: radius scale factor
  c->EffectRadius                       = worldRadius;
  c->EffectShadowStrength               = m_strength * 2.0f;  // user 0-2 → internal 0-4
  c->EffectShadowPow                    = 1.5f;
  c->EffectShadowClamp                  = m_shadowClamp;
  // Fadeout, SDK semantics (fadeOutFrom / fadeOutTo): FULL effect up to
  // worldRadius, linear falloff to zero at worldRadius*40 (=1600 units for
  // radius 4). The old linear-from-zero 20x fade (zero at 800) suppressed
  // most of the mid-range world geometry — in an open L2 world the visible
  // terrain is mostly 300..1500 units away, so the AO read as exactly
  // neutral everywhere. Sky/far background still fades out completely.
  c->EffectFadeOutMul                   = -1.0f / (worldRadius * 39.0f);
  c->EffectFadeOutAdd                   = 1.0f + 1.0f / 39.0f;
  c->EffectHorizonAngleThreshold        = 0.06f;
  c->EffectSamplingRadiusNearLimitRec   = 1.0f / (worldRadius * 1.2f / tanHalfFOVY);
  c->DepthPrecisionOffsetMod            = 0.9992f;
  c->NegRecEffectRadius                 = -1.0f / worldRadius;
  c->AdaptiveSampleCountLimit           = 0.45f;
  c->InvSharpness                       = 0.0f;
  c->BlurNumPasses                      = 4;
  c->BilateralSigmaSquared              = 5.0f;
  c->BilateralSimilarityDistanceSigma   = 0.1f;
  c->DetailAOStrength                   = 0.5f;
  c->NormalsUnpackMul                   = 2.0f;
  c->NormalsUnpackAdd                   = -1.0f;

  std::memcpy(c->PatternRotScaleMatrices, kPatternData, sizeof(kPatternData));

  std::memset(c->NormalsWorldToViewspaceMatrix, 0, sizeof(c->NormalsWorldToViewspaceMatrix));
  c->NormalsWorldToViewspaceMatrix[0][0] = 1.0f;
  c->NormalsWorldToViewspaceMatrix[1][1] = 1.0f;
  c->NormalsWorldToViewspaceMatrix[2][2] = 1.0f;
  c->NormalsWorldToViewspaceMatrix[3][3] = 1.0f;

  c->SSAOBufferDimensions[0] = float(bsi.ssaoBufferWidth);
  c->SSAOBufferDimensions[1] = float(bsi.ssaoBufferHeight);
  c->SSAOBufferInverseDimensions[0] = 1.0f / c->SSAOBufferDimensions[0];
  c->SSAOBufferInverseDimensions[1] = 1.0f / c->SSAOBufferDimensions[1];

  c->DepthBufferDimensions[0] = float(w);
  c->DepthBufferDimensions[1] = float(h);
  c->DepthBufferInverseDimensions[0] = 1.0f / float(w);
  c->DepthBufferInverseDimensions[1] = 1.0f / float(h);

  c->DepthBufferOffset[0] = 0;
  c->DepthBufferOffset[1] = 0;

  for (int p = 0; p < 4; p++) {
    c->PerPassFullResUVOffset[p * 4 + 0] = float(p % 2) / float(bsi.ssaoBufferWidth);
    c->PerPassFullResUVOffset[p * 4 + 1] = float(p / 2) / float(bsi.ssaoBufferHeight);
    c->PerPassFullResUVOffset[p * 4 + 2] = 0.0f;
    c->PerPassFullResUVOffset[p * 4 + 3] = 0.0f;
  }

  c->InputOutputBufferDimensions[0] = float(w);
  c->InputOutputBufferDimensions[1] = float(h);
  c->InputOutputBufferInverseDimensions[0] = 1.0f / float(w);
  c->InputOutputBufferInverseDimensions[1] = 1.0f / float(h);

  c->ImportanceMapDimensions[0] = float(bsi.importanceMapWidth);
  c->ImportanceMapDimensions[1] = float(bsi.importanceMapHeight);
  c->ImportanceMapInverseDimensions[0] = 1.0f / c->ImportanceMapDimensions[0];
  c->ImportanceMapInverseDimensions[1] = 1.0f / c->ImportanceMapDimensions[1];

  c->DeinterleavedDepthBufferDimensions[0] = float(bsi.deinterleavedDepthBufferWidth);
  c->DeinterleavedDepthBufferDimensions[1] = float(bsi.deinterleavedDepthBufferHeight);
  c->DeinterleavedDepthBufferInverseDimensions[0] = 1.0f / c->DeinterleavedDepthBufferDimensions[0];
  c->DeinterleavedDepthBufferInverseDimensions[1] = 1.0f / c->DeinterleavedDepthBufferDimensions[1];

  c->DeinterleavedDepthBufferOffset[0] = 0.0f;
  c->DeinterleavedDepthBufferOffset[1] = 0.0f;
  c->DeinterleavedDepthBufferNormalisedOffset[0] = 0.0f;
  c->DeinterleavedDepthBufferNormalisedOffset[1] = 0.0f;

  c->LoadCounterAvgDiv = 9.0f /
    float(bsi.importanceMapWidth * bsi.importanceMapHeight * 255);
}

// ── Fill AO buffer with constant value (for UI frames to avoid flickering) ──
void D3D9CacaoRenderer::FillAoBuffer(D3D9DeviceEx* device, float value) {
  if (!m_initialized || !m_enabled || !m_aoFinals[m_frameIndex])
    return;

  auto vkd = m_device->vkd();
  Rc<DxvkImage> aoFinal = m_aoFinals[m_frameIndex];

  device->EmitCs([
    vkd, aoFinal = std::move(aoFinal), value
  ] (DxvkContext* cs) {
    cs->beginExternalRendering();
    Rc<DxvkCommandList> cmd = cs->getCommandListForCompute();
    VkDevice vkdev = vkd->device();

    // Transition to TRANSFER_DST for clear
    VkImageMemoryBarrier2 b = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
    b.srcStageMask  = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    b.srcAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    b.dstStageMask  = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    b.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    b.oldLayout     = VK_IMAGE_LAYOUT_GENERAL;
    b.newLayout     = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    b.image         = aoFinal->handle();
    b.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    VkDependencyInfo di = { VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    di.imageMemoryBarrierCount = 1;
    di.pImageMemoryBarriers    = &b;
    cmd->cmdPipelineBarrier(DxvkCmdBuffer::ExecBuffer, &di);

    // Clear AO buffer with value
    VkClearColorValue clearVal = {};
    clearVal.float32[0] = value;  // R = AO value
    clearVal.float32[1] = 0.0f;   // G
    clearVal.float32[2] = 0.0f;   // B
    clearVal.float32[3] = 1.0f;   // A
    VkImageSubresourceRange range = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    cmd->cmdClearColorImage(DxvkCmdBuffer::ExecBuffer,
      aoFinal->handle(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clearVal, 1, &range);

    // Transition back to GENERAL for fragment shader read
    b.srcStageMask  = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    b.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    b.dstStageMask  = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
    b.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
    b.oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    b.newLayout     = VK_IMAGE_LAYOUT_GENERAL;
    cmd->cmdPipelineBarrier(DxvkCmdBuffer::ExecBuffer, &di);

    cmd->track(aoFinal, DxvkAccess::Write);
  });
}


// ── WriteDumpFiles: convert staging buffer contents to PGM images ────────────
void D3D9CacaoRenderer::WriteDumpFiles(uint32_t slot) {
  if (!m_readbackBuffer || m_width == 0)
    return;

  const uint8_t* base = reinterpret_cast<const uint8_t*>(m_readbackBuffer->mapPtr(0));

  BufSizeInfo bsi;
  calcBufSizeInfo(m_width, m_height, &bsi);

  auto half2f = [](uint16_t h) -> float {
    uint32_t sign = (h >> 15) & 0x1, ex = (h >> 10) & 0x1f, mt = h & 0x3ff;
    uint32_t f = (ex == 0) ? ((sign << 31) | (mt << 13))
               : ((sign << 31) | ((ex == 31 ? 0x7f800000u : ((ex + 112u) << 23)) | (mt << 13)));
    float r; std::memcpy(&r, &f, 4); return r;
  };

  const char* tag = slot == 0 ? "A" : "B";

  auto writePgm = [&](const char* name, uint32_t w, uint32_t h, const std::vector<uint8_t>& px) {
    FILE* f = fopen(name, "wb");
    if (!f) return;
    fprintf(f, "P5\n%u %u\n255\n", w, h);
    fwrite(px.data(), 1, px.size(), f);
    fclose(f);
  };

  // 1) aoFinal: RGBA16F, R = visibility
  {
    std::vector<uint8_t> px(size_t(m_width) * m_height);
    const uint16_t* src = reinterpret_cast<const uint16_t*>(base + 9u * 1024u * 1024u);
    for (uint32_t i = 0; i < px.size(); i++)
      px[i] = uint8_t(dxvk::clamp(half2f(src[i * 4 + 0]), 0.0f, 1.0f) * 255.0f);
    char name[64];
    snprintf(name, sizeof(name), "cacaodump_%s_ao.pgm", tag);
    writePgm(name, m_width, m_height, px);
  }

  // 2) deinterleaved depth layer 0: R32F viewZ, log scale
  {
    std::vector<uint8_t> px(size_t(bsi.deinterleavedDepthBufferWidth) * bsi.deinterleavedDepthBufferHeight);
    const float* src = reinterpret_cast<const float*>(base + 1u * 1024u * 1024u);
    const float maxZ = 30000.0f, invLog = 1.0f / std::log(1.0f + maxZ);
    for (uint32_t i = 0; i < px.size(); i++) {
      float z = src[i] < 0.0f ? 0.0f : src[i];
      // Clamp: sky (~far, viewZ ~66000) exceeds maxZ and would wrap around
      // uint8_t to near-black, looking like garbage in the dump.
      float v = std::log(1.0f + z) * invLog * 255.0f;
      px[i] = uint8_t(v > 255.0f ? 255.0f : v);
    }
    char name[64];
    snprintf(name, sizeof(name), "cacaodump_%s_depth.pgm", tag);
    writePgm(name, bsi.deinterleavedDepthBufferWidth, bsi.deinterleavedDepthBufferHeight, px);
  }

  // 3) blurred SSAO layer 0: RG32F (R = visibility, G = packed edges)
  {
    std::vector<uint8_t> px(size_t(bsi.ssaoBufferWidth) * bsi.ssaoBufferHeight);
    const float* src = reinterpret_cast<const float*>(base + 4u * 1024u * 1024u);
    for (uint32_t i = 0; i < px.size(); i++)
      px[i] = uint8_t(dxvk::clamp(src[i * 2 + 0], 0.0f, 1.0f) * 255.0f);
    char name[64];
    snprintf(name, sizeof(name), "cacaodump_%s_ssao.pgm", tag);
    writePgm(name, bsi.ssaoBufferWidth, bsi.ssaoBufferHeight, px);
  }

  CacaoLogger::get().log(str::format(
    "[DUMP] wrote cacaodump_", tag, "_*.pgm (ao ", m_width, "x", m_height,
    ", depth/ssao ", bsi.ssaoBufferWidth, "x", bsi.ssaoBufferHeight, ")"));
}

// ── Apply ──────────────────────────────────────────────────────────────────────
void D3D9CacaoRenderer::Apply(
    D3D9DeviceEx*  device,
    Rc<DxvkImage>  depthImage,
    const float*   projMatrix4x4,
    uint32_t       width,
    uint32_t       height,
    bool           debugFill)
{
  // Update HUD globals on every call so toggle is reflected immediately
  hud::g_cacaoEnabled.store(m_enabled);
  hud::g_cacaoQuality.store((int)m_quality);

  if (!m_initialized || !m_enabled)
    return;

  if (!depthImage)
    return;

  // ── CPU readback: multi-stage probe from previous frame's GPU copy ──────
  if (m_readbackPending && m_readbackBuffer) {
    m_readbackPending = false;
    const uint8_t* base = reinterpret_cast<const uint8_t*>(m_readbackBuffer->mapPtr(0));

    // Convert float16 to float32 manually (simple half-to-float)
    auto half2float = [](uint16_t h) -> float {
      uint32_t sign     = (h >> 15) & 0x1;
      uint32_t exponent = (h >> 10) & 0x1f;
      uint32_t mantissa = h & 0x3ff;
      uint32_t f;
      if (exponent == 0)       f = (sign << 31) | (mantissa << 13);
      else if (exponent == 31) f = (sign << 31) | 0x7f800000 | (mantissa << 13);
      else                     f = (sign << 31) | ((exponent + 112) << 23) | (mantissa << 13);
      float result; std::memcpy(&result, &f, 4); return result;
    };

    float viewDepth = 0.0f, ssaoR = 0.0f, ssaoG = 0.0f;
    std::memcpy(&viewDepth, base + 0, 4);
    std::memcpy(&ssaoR,     base + 16, 4);
    std::memcpy(&ssaoG,     base + 20, 4);
    uint32_t snapRaw = 0;
    std::memcpy(&snapRaw,   base + 32, 4);
    float snapD = float(snapRaw & 0xFFFFFF) / 16777215.0f;

    const uint16_t* strip = reinterpret_cast<const uint16_t*>(base + 64);
    float aoMin = 1e9f, aoMax = -1e9f;
    for (uint32_t i = 0; i < 64; i++) {
      float ao = half2float(strip[i * 4 + 0]);
      aoMin = std::min(aoMin, ao);
      aoMax = std::max(aoMax, ao);
    }
    CacaoLogger::get().log(str::format(
      "[READBACK] viewZ=", viewDepth, " ssao=(", ssaoR, ", ", ssaoG, ")",
      " snapD=", snapD, " (0x", std::hex, snapRaw, std::dec, ")",
      " aoStrip[min=", aoMin, " max=", aoMax, "]",
      " first4=", half2float(strip[0]), ",", half2float(strip[4]), ",",
      half2float(strip[8]), ",", half2float(strip[12])));

    // World-diagnostics rows (offsets 1024/1152/1536, see copy side).
    {
      const float* depthRow = reinterpret_cast<const float*>(base + 1024);
      float dMin = 1e9f, dMax = -1e9f;
      for (uint32_t i = 0; i < 16; i++) {
        dMin = std::min(dMin, depthRow[i]);
        dMax = std::max(dMax, depthRow[i]);
      }
      const float* ssaoRow = reinterpret_cast<const float*>(base + 1152);
      float sMin = 1e9f, sMax = -1e9f;
      for (uint32_t i = 0; i < 16; i++) {
        sMin = std::min(sMin, ssaoRow[i * 2 + 0]);
        sMax = std::max(sMax, ssaoRow[i * 2 + 0]);
      }
      const uint16_t* strip2 = reinterpret_cast<const uint16_t*>(base + 1536);
      float a2Min = 1e9f, a2Max = -1e9f;
      for (uint32_t i = 0; i < 64; i++) {
        float ao = half2float(strip2[i * 4 + 0]);
        a2Min = std::min(a2Min, ao);
        a2Max = std::max(a2Max, ao);
      }
      CacaoLogger::get().log(str::format(
        "[READBACK2] depthRow[min=", dMin, " max=", dMax, "]",
        " ssaoRow[min=", sMin, " max=", sMax, "]",
        " aoTop[min=", a2Min, " max=", a2Max, "]",
        " radius=", m_radius * m_worldRadiusScale,
        " fadeZero=", m_radius * m_worldRadiusScale * 40.0f,
        " msaa=", m_lastDepthSampleCount));
    }
  }

  m_applyCount++;

  bool useSnapInput = depthImage.ptr() == m_depthCopy.ptr();
  if (m_applyCount % 60 == 0)
    CacaoLogger::get().log(str::format("[SRC] snapInput=", useSnapInput,
      " depthImg=", depthImage.ptr(), " snap=", m_depthCopy.ptr()));

  bool dumpEnabled = m_dumpEnabled && m_readbackBuffer != nullptr;

  // ── Host-side write of a completed full-buffer dump ────────────────────
  if (m_dumpWritePending && m_applyCount >= m_dumpWriteAt) {
    m_dumpWritePending = false;
    WriteDumpFiles(m_dumpWriteSlot);
  }

  // Issue a new dump when enabled (alternates A/B slots), throttled because
  // the full-buffer GPU copy and the host-side PGM write are expensive.
  bool doDump = false;
  if (dumpEnabled && !m_dumpWritePending && m_applyCount >= m_dumpWriteAt
   && std::chrono::steady_clock::now() - m_dumpLastIssue >= kDumpMinInterval) {
    doDump             = true;
    m_dumpLastIssue    = std::chrono::steady_clock::now();
    m_probeBurst       = 12;
    m_dumpWritePending = true;
    m_dumpWriteAt      = m_applyCount + kFrameLag;
    m_dumpWriteSlot    = m_dumpSlot;
    m_dumpSlot        ^= 1u;
    CacaoLogger::get().log(str::format(
      "[DUMP] issuing full-buffer dump slot=", m_dumpWriteSlot == 0 ? "A" : "B",
      " apply=", m_applyCount));
  }

  // Log first successful Apply call with diagnostic info
  static bool s_firstApply = true;
  if (s_firstApply) {
    s_firstApply = false;
    CacaoLogger::get().log(str::format("[APPLY] CACAO running at ", width, "x", height));
    CacaoLogger::get().log(str::format("[DIAG]  depth format=", (int)depthImage->info().format,
      " layout=", (int)depthImage->info().layout,
      " usage=", (int)depthImage->info().usage,
      " msaa=", (int)depthImage->info().sampleCount));
    CacaoLogger::get().log(str::format("[DIAG]  proj[0]=",  projMatrix4x4[0],
      " proj[5]=",  projMatrix4x4[5],
      " proj[10]=", projMatrix4x4[10],
      " proj[14]=", projMatrix4x4[14]));
    const float mul = -projMatrix4x4[14];
    const float add = projMatrix4x4[10];
    CacaoLogger::get().log(str::format("[DIAG]  depthLinearizeMul=", mul,
      " depthLinearizeAdd=", add,
      " (after sign fix: ", (mul * add < 0.0f ? -add : add), ")"));
    CacaoLogger::get().log(str::format("[DIAG]  tanHalfFOVX=", 1.0f/projMatrix4x4[0],
      " tanHalfFOVY=", 1.0f/projMatrix4x4[5],
      " effectRadius=", m_radius));
    if (debugFill)
      CacaoLogger::get().log("[DIAG]  debugFill=ON: AO will be constant 0.5");
  }

  // Reallocate intermediate resources if resolution changed
  if (width != m_width || height != m_height)
    InitResources(width, height);

  // Check if constant buffer is initialized
  if (!m_constantBuffers[m_frameIndex])
    return;

  // Upload constants to frame-specific buffer (coherent memory, safe before EmitCs)
  {
    CacaoConstants consts;
    UpdateConstants(&consts, projMatrix4x4, width, height);
    std::memcpy(m_constantBuffers[m_frameIndex]->mapPtr(0), &consts, sizeof(consts));
  }
  m_lastDepthSampleCount = depthImage->info().sampleCount;

  BufSizeInfo bsi;
  calcBufSizeInfo(width, height, &bsi);

  const uint32_t frameBase = m_frameIndex * kSetsPerFrame;

  // Capture resources by Rc<> (keeps alive through GPU execution)
  Rc<DxvkImage>  deintD    = m_deinterleavedDepths;
  Rc<DxvkImage>  deintN    = m_deinterleavedNormals;
  Rc<DxvkImage>  ssaoPing  = m_ssaoPing;
  Rc<DxvkImage>  ssaoPong  = m_ssaoPong;
  Rc<DxvkImage>  impMap    = m_importanceMap;
  Rc<DxvkImage>  impMapP   = m_importanceMapPong;
  Rc<DxvkImage>  aoFinal   = m_aoFinals[m_frameIndex];  // Frame-specific AO buffer
  Rc<DxvkImageView> aoUavView = m_vAoUavs[m_frameIndex];

  Rc<DxvkImageView> vDeintDUav = m_vDeintDUav;
  Rc<DxvkImageView> vDeintDSrv = m_vDeintDSrv;
  Rc<DxvkImageView> vDeintNUav = m_vDeintNUav;
  Rc<DxvkImageView> vDeintNSrv = m_vDeintNSrv;
  std::array<Rc<DxvkImageView>, 4> vDeintDMips = m_vDeintDMips;
  Rc<DxvkImageView> vPingSrv = m_vPingSrv;
  Rc<DxvkImageView> vPingUav = m_vPingUav;
  Rc<DxvkImageView> vPongSrv = m_vPongSrv;
  Rc<DxvkImageView> vPongUav = m_vPongUav;
  Rc<DxvkImageView> vImpSrv  = m_vImpSrv;
  Rc<DxvkImageView> vLoadSrv = m_vLoadSrv;
  Rc<DxvkImage>  loadCnt   = m_loadCounter;
  Rc<DxvkBuffer> cb        = m_constantBuffers[m_frameIndex];  // Frame-specific constant buffer
  Rc<DxvkImage>  depthCopy = m_depthCopy;  // Capture for depth copy target

  // Create depth copy target if needed (before EmitCs, on host thread)
  // Keep same format as original depth for compatibility with vkCmdCopyImage
  if (!depthCopy || depthCopy->info().extent.width != depthImage->info().extent.width
      || depthCopy->info().extent.height != depthImage->info().extent.height
      || depthCopy->info().format != depthImage->info().format) {
    DxvkImageCreateInfo rci = {};
    rci.type        = VK_IMAGE_TYPE_2D;
    rci.format      = depthImage->info().format;  // Same format as source
    rci.sampleCount = VK_SAMPLE_COUNT_1_BIT;
    rci.extent      = depthImage->info().extent;
    rci.numLayers   = 1;
    rci.mipLevels   = 1;
    rci.usage       = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    rci.stages      = VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    rci.access      = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
    rci.layout      = VK_IMAGE_LAYOUT_GENERAL;
    rci.tiling      = VK_IMAGE_TILING_OPTIMAL;
    depthCopy = m_device->createImage(rci, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    m_depthCopy = depthCopy;  // Store for reuse
    CacaoLogger::get().log(str::format("[DEPTH-COPY] Created copy target: ",
      rci.extent.width, "x", rci.extent.height, " fmt=", (int)rci.format));
  }

  // Capture Vulkan objects and descriptor sets for this frame
  auto   vkd        = m_device->vkd();
  auto   pipelines  = m_pipelines.data();
  auto   pipeLayouts= m_pipelineLayouts.data();
  // Descriptor sets for this frame (frameBase..frameBase+kSetsPerFrame-1)
  VkDescriptorSet frameSets[kSetsPerFrame];
  for (uint32_t i = 0; i < kSetsPerFrame; i++)
    frameSets[i] = m_descSets[frameBase + i];

  uint32_t w          = width;
  uint32_t h          = height;
  BufSizeInfo bsiCopy = bsi;
  CacaoQuality quality = m_quality;

  // Decide whether to issue a readback this frame. The dump needs the
  // staging buffer too, but the full-buffer copy must not run every apply
  // (25 MB over PCIe per present was a major FPS loss).
  m_readbackFrameCount++;
  bool burst = m_probeBurst > 0;
  if (burst)
    m_probeBurst--;
  bool doReadback = ((m_readbackFrameCount % kReadbackInterval == 0) || burst) && m_readbackBuffer;
  bool needStaging = (doReadback || doDump) && m_readbackBuffer;
  Rc<DxvkBuffer> readbackBuf = needStaging ? m_readbackBuffer : nullptr;
  if (doReadback)
    m_readbackPending = true;

  device->EmitCs([
    vkd, depthImage = std::move(depthImage), depthCopy,
    deintD, deintN, ssaoPing, ssaoPong, impMap, impMapP, aoFinal, aoUavView, loadCnt, cb,
    pipelines, pipeLayouts,
    vDeintDUav, vDeintDSrv, vDeintNUav, vDeintNSrv, vDeintDMips,
    vPingSrv, vPingUav, vPongSrv, vPongUav, vImpSrv, vLoadSrv,
    frameSets /* copied by value (array of VkDescriptorSet handles) */,
    w, h, bsiCopy, quality, debugFill,
    readbackBuf, doReadback, doDump, useSnapInput
  ] (DxvkContext* cs) mutable {

    cs->beginExternalRendering();
    Rc<DxvkCommandList> cmd = cs->getCommandListForCompute();
    VkDevice vkdev = vkd->device();


    // ── Helper: create an image view ──────────────────────────────────────
    // usage: pass VK_IMAGE_USAGE_SAMPLED_BIT for SRV, VK_IMAGE_USAGE_STORAGE_BIT for UAV
    auto makeView = [&](
        const Rc<DxvkImage>& img,
        VkFormat fmt,
        VkImageViewType vtype,
        VkImageUsageFlagBits usage,
        VkImageLayout layout,
        uint32_t baseLayer, uint32_t layerCnt,
        uint32_t baseMip = 0, uint32_t mipCnt = 1,
        VkImageAspectFlags aspects = VK_IMAGE_ASPECT_COLOR_BIT) -> Rc<DxvkImageView>
    {
      DxvkImageViewKey key;
      key.viewType   = vtype;
      key.usage      = usage;
      key.format     = fmt;
      key.layout     = layout;
      key.aspects    = aspects;
      key.mipIndex   = uint8_t(baseMip);
      key.mipCount   = uint8_t(mipCnt);
      key.layerIndex = uint16_t(baseLayer);
      key.layerCount = uint16_t(layerCnt);
      return img->createView(key);
    };

    // ── Helper: memory barrier between compute passes ─────────────────────
    auto computeBarrier = [&]() {
      VkMemoryBarrier2 mb = { VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
      mb.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
      mb.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
      mb.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
      mb.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
      VkDependencyInfo di = { VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
      di.memoryBarrierCount = 1;
      di.pMemoryBarriers    = &mb;
      cmd->cmdPipelineBarrier(DxvkCmdBuffer::ExecBuffer, &di);
    };

    // ── Helper: initial layout transition from UNDEFINED to GENERAL ──────
    // Intermediate images are created with UNDEFINED initial layout from
    // Vulkan's perspective (no explicit transition occurred at creation).
    // This helper performs the required initial transition before first use.
    auto initToGeneral = [&](const Rc<DxvkImage>& img,
                             uint32_t              layers = 1,
                             uint32_t              mips   = 1,
                             VkImageAspectFlags    aspect = VK_IMAGE_ASPECT_COLOR_BIT) {
      VkImageMemoryBarrier2 b = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
      b.srcStageMask  = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
      b.srcAccessMask = 0;
      b.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
      b.dstAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_SHADER_READ_BIT;
      b.oldLayout     = VK_IMAGE_LAYOUT_UNDEFINED;
      b.newLayout     = VK_IMAGE_LAYOUT_GENERAL;
      b.image         = img->handle();
      b.subresourceRange = { aspect, 0, mips, 0, layers };
      VkDependencyInfo di = { VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
      di.imageMemoryBarrierCount = 1;
      di.pImageMemoryBarriers    = &b;
      cmd->cmdPipelineBarrier(DxvkCmdBuffer::ExecBuffer, &di);
    };

    // ── Helper: update a single image descriptor write ────────────────────
    auto imgWrite = [&](VkDescriptorSet ds, uint32_t binding,
                        VkDescriptorType type, const VkDescriptorImageInfo& info,
                        uint32_t arrayElement = 0) {
      VkWriteDescriptorSet w = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
      w.dstSet          = ds;
      w.dstBinding      = binding;
      w.dstArrayElement = arrayElement;
      w.descriptorCount = 1;
      w.descriptorType  = type;
      w.pImageInfo      = &info;
      vkd->vkUpdateDescriptorSets(vkdev, 1, &w, 0, nullptr);
    };

    auto bufWrite = [&](VkDescriptorSet ds, uint32_t binding,
                        const VkDescriptorBufferInfo& info) {
      VkWriteDescriptorSet w = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
      w.dstSet         = ds;
      w.dstBinding     = binding;
      w.descriptorCount= 1;
      w.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
      w.pBufferInfo    = &info;
      vkd->vkUpdateDescriptorSets(vkdev, 1, &w, 0, nullptr);
    };

    // ── Shared CB info ─────────────────────────────────────────────────────
    VkDescriptorBufferInfo cbInfo = {};
    cbInfo.buffer = cb->getSliceInfo().buffer;
    cbInfo.offset = cb->getSliceInfo().offset;
    cbInfo.range  = sizeof(CacaoConstants);

    constexpr uint32_t TG = 8;

    // ── Debug fill: skip CACAO, write 0.5 to aoFinal to test composite ─────
    if (debugFill) {
      initToGeneral(aoFinal);
      VkClearColorValue clearVal = {};
      clearVal.float32[0] = clearVal.float32[1] = clearVal.float32[2] = 0.5f;
      clearVal.float32[3] = 1.0f;
      VkImageSubresourceRange range = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
      cmd->cmdClearColorImage(DxvkCmdBuffer::ExecBuffer,
        aoFinal->handle(), VK_IMAGE_LAYOUT_GENERAL, &clearVal, 1, &range);
      // Barrier: clear write → fragment shader read
      VkImageMemoryBarrier2 b = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
      b.srcStageMask  = VK_PIPELINE_STAGE_2_CLEAR_BIT;
      b.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
      b.dstStageMask  = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
      b.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
      b.oldLayout = b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
      b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      b.image = aoFinal->handle();
      b.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
      VkDependencyInfo di = { VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
      di.imageMemoryBarrierCount = 1;
      di.pImageMemoryBarriers    = &b;
      cmd->cmdPipelineBarrier(DxvkCmdBuffer::ExecBuffer, &di);
      cmd->track(aoFinal, DxvkAccess::Write);
      return;
    }

    // ── Depth buffer: Read directly from GENERAL layout (no transition needed) ──
    // DXVK creates depth images with SAMPLED_BIT, so we can read from GENERAL layout.
    // This avoids layout transition conflicts with DXVK's internal depth tracking.

    // ── Initial layout transition: UNDEFINED → GENERAL for all intermediate images ──
    // Required because Vulkan validation layers track subresource layouts as UNDEFINED
    // until an explicit transition occurs. DXVK internally stores the target layout
    // from createInfo, but the actual GPU-side transition never happened at creation.
    initToGeneral(deintD,   4, 4);
    initToGeneral(deintN,   4, 1);
    initToGeneral(ssaoPing, 4, 1);
    initToGeneral(ssaoPong, 4, 1);
    initToGeneral(impMap,   1, 1);
    initToGeneral(impMapP,  1, 1);
    initToGeneral(aoFinal,  1, 1);
    initToGeneral(loadCnt,  1, 1);

    // ── Neutral-init importance map and load counter ─────────────────────
    // The Q3 shader's adaptive branch (quality >= 3) samples
    // g_ImportanceMap and g_LoadCounter, but the non-adaptive pipeline
    // never runs the importance-map generation passes. Reading them
    // uninitialized produced garbage extra taps (checkerboard artifacts
    // surviving the blur). Zero importance = "no additional taps";
    // zero counter keeps the importance limiter well-defined.
    {
      VkClearColorValue zeroF = {};   // impMap is R32_SFLOAT
      VkClearColorValue zeroU = {};   // loadCnt is R32_UINT
      VkImageSubresourceRange range = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
      cmd->cmdClearColorImage(DxvkCmdBuffer::ExecBuffer,
        impMap->handle(),  VK_IMAGE_LAYOUT_GENERAL, &zeroF, 1, &range);
      cmd->cmdClearColorImage(DxvkCmdBuffer::ExecBuffer,
        loadCnt->handle(), VK_IMAGE_LAYOUT_GENERAL, &zeroU, 1, &range);

      // Barrier: transfer writes above -> compute reads below
      VkImageMemoryBarrier2 cb = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
      cb.srcStageMask  = VK_PIPELINE_STAGE_2_CLEAR_BIT;
      cb.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
      cb.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
      cb.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
      cb.oldLayout     = VK_IMAGE_LAYOUT_GENERAL;
      cb.newLayout     = VK_IMAGE_LAYOUT_GENERAL;
      cb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      cb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      VkDependencyInfo cdi = { VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
      VkImageMemoryBarrier2 cbs[2] = { cb, cb };
      cbs[0].image = impMap->handle();
      cbs[0].subresourceRange = range;
      cbs[1].image = loadCnt->handle();
      cbs[1].subresourceRange = range;
      cdi.imageMemoryBarrierCount = 2;
      cdi.pImageMemoryBarriers    = cbs;
      cmd->cmdPipelineBarrier(DxvkCmdBuffer::ExecBuffer, &cdi);
    }

    // Use original depth image format for shader sampling
    const VkFormat fmtDepthSrv = depthImage->info().format;

    // Depth SRV shared by PASS 1 and PASS 2 (created once per Apply)
    auto depthSrv = makeView(depthImage, fmtDepthSrv, VK_IMAGE_VIEW_TYPE_2D,
      VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_LAYOUT_GENERAL,
      0, 1, 0, 1, VK_IMAGE_ASPECT_DEPTH_BIT);

    // ════════════════════════════════════════════════════════════════════════
    // PASS 1 — Prepare native depths and mips
    // ════════════════════════════════════════════════════════════════════════
    {
      VkDescriptorSet ds = frameSets[SET_IDX_DEPTHS];

      VkDescriptorImageInfo depthSrvInfo = { {}, depthSrv->handle(), VK_IMAGE_LAYOUT_GENERAL };
      imgWrite(ds, B_SRV_DEPTH_IN, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, depthSrvInfo);

      VkDescriptorImageInfo deintDUavInfo = { {}, vDeintDUav->handle(), VK_IMAGE_LAYOUT_GENERAL };
      imgWrite(ds, B_UAV_DEINT_DEPTHS, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, deintDUavInfo);

      // depth mips UAV: 4 array elements, one per mip level
      // (descriptor array binding with count=4, each element = one mip view)
      std::array<VkDescriptorImageInfo, 4> mipInfos = {};
      for (uint32_t m = 0; m < 4; m++)
        mipInfos[m] = { {}, vDeintDMips[m]->handle(), VK_IMAGE_LAYOUT_GENERAL };
      VkWriteDescriptorSet mipWrite = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
      mipWrite.dstSet          = ds;
      mipWrite.dstBinding      = B_UAV_DEPTH_MIPS;
      mipWrite.descriptorCount = 4;
      mipWrite.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
      mipWrite.pImageInfo      = mipInfos.data();
      vkd->vkUpdateDescriptorSets(vkdev, 1, &mipWrite, 0, nullptr);

      bufWrite(ds, B_CB_CACAO, cbInfo);

      cmd->cmdBindPipeline(DxvkCmdBuffer::ExecBuffer,
        VK_PIPELINE_BIND_POINT_COMPUTE, pipelines[CACAO_PASS_PREPARE_DEPTHS]);
      VkBindDescriptorSetsInfo bindInfo = { VK_STRUCTURE_TYPE_BIND_DESCRIPTOR_SETS_INFO };
      bindInfo.stageFlags          = VK_SHADER_STAGE_COMPUTE_BIT;
      bindInfo.layout            = pipeLayouts[CACAO_PASS_PREPARE_DEPTHS];
      bindInfo.firstSet          = 0;
      bindInfo.descriptorSetCount = 1;
      bindInfo.pDescriptorSets   = &ds;
      cmd->cmdBindDescriptorSets(DxvkCmdBuffer::ExecBuffer, &bindInfo);
      cmd->cmdDispatch(DxvkCmdBuffer::ExecBuffer,
        dispatchSize(TG, bsiCopy.deinterleavedDepthBufferWidth),
        dispatchSize(TG, bsiCopy.deinterleavedDepthBufferHeight),
        1);
    }
    computeBarrier();

    // ════════════════════════════════════════════════════════════════════════
    // PASS 2 — Prepare normals from depth
    // ════════════════════════════════════════════════════════════════════════
    {
      VkDescriptorSet ds = frameSets[SET_IDX_NORMALS];

      // Use original depth image for normals computation (GENERAL layout)
      VkDescriptorImageInfo depthSrvInfo = { {}, depthSrv->handle(), VK_IMAGE_LAYOUT_GENERAL };
      imgWrite(ds, B_SRV_DEPTH_IN, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, depthSrvInfo);

      VkDescriptorImageInfo deintNUavInfo = { {}, vDeintNUav->handle(), VK_IMAGE_LAYOUT_GENERAL };
      imgWrite(ds, B_UAV_DEINT_NORMALS, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, deintNUavInfo);

      bufWrite(ds, B_CB_CACAO, cbInfo);

      cmd->cmdBindPipeline(DxvkCmdBuffer::ExecBuffer,
        VK_PIPELINE_BIND_POINT_COMPUTE, pipelines[CACAO_PASS_PREPARE_NORMALS]);
      VkBindDescriptorSetsInfo bindInfo = { VK_STRUCTURE_TYPE_BIND_DESCRIPTOR_SETS_INFO };
      bindInfo.stageFlags          = VK_SHADER_STAGE_COMPUTE_BIT;
      bindInfo.layout            = pipeLayouts[CACAO_PASS_PREPARE_NORMALS];
      bindInfo.firstSet          = 0;
      bindInfo.descriptorSetCount = 1;
      bindInfo.pDescriptorSets   = &ds;
      cmd->cmdBindDescriptorSets(DxvkCmdBuffer::ExecBuffer, &bindInfo);
      cmd->cmdDispatch(DxvkCmdBuffer::ExecBuffer,
        dispatchSize(TG, bsiCopy.ssaoBufferWidth),
        dispatchSize(TG, bsiCopy.ssaoBufferHeight),
        1);
    }
    computeBarrier();

    // ════════════════════════════════════════════════════════════════════════
    // PASS 3 — Generate AO (z=4: all 4 deinterleaved layers at once)
    //          Reads: deintD (SRV), deintN (SRV), ssaoPong (SRV, unused in non-adaptive)
    //          Writes: ssaoPing (UAV)
    //
    //          Pipeline is selected per quality level. The q3 SPIR-V is the
    //          ADAPTIVE variant: its non-adaptive branch is skipped at runtime
    //          (qualityLevel==3), and the adaptive branch reads the "base pass"
    //          via LoadBasePassSSAOPass from the pong buffer. This port never
    //          runs the q3 base/importance passes, so pong holds the previous
    //          frame's blur intermediate — a temporal feedback loop that made
    //          the output oscillate (period-2 flicker). The q0/q1/q2 variants
    //          contain the same adaptive code but with a compile-time-dead
    //          branch (verified in SPIR-V), so they are fully deterministic.
    //          Until the full adaptive pipeline is ported, Medium and High both
    //          use the q2 non-adaptive variant.
    // ════════════════════════════════════════════════════════════════════════
    {
      const CacaoPass generatePass =
          quality == CacaoQuality::Lowest ? CACAO_PASS_GENERATE_Q0
        : quality == CacaoQuality::Low    ? CACAO_PASS_GENERATE_Q1
        :                                   CACAO_PASS_GENERATE_Q2;

      VkDescriptorSet ds = frameSets[SET_IDX_Q3];

      VkDescriptorImageInfo deintDInfo  = { {}, vDeintDSrv->handle(),  VK_IMAGE_LAYOUT_GENERAL };
      VkDescriptorImageInfo deintNInfo  = { {}, vDeintNSrv->handle(),  VK_IMAGE_LAYOUT_GENERAL };
      VkDescriptorImageInfo pingUavInfo = { {}, vPingUav->handle(),    VK_IMAGE_LAYOUT_GENERAL };
      VkDescriptorImageInfo pongSrvInfo = { {}, vPongSrv->handle(),    VK_IMAGE_LAYOUT_GENERAL };
      VkDescriptorImageInfo impInfo     = { {}, vImpSrv->handle(),     VK_IMAGE_LAYOUT_GENERAL };
      VkDescriptorImageInfo loadInfo    = { {}, vLoadSrv->handle(),    VK_IMAGE_LAYOUT_GENERAL };

      imgWrite(ds, B_SRV_LOAD_COUNTER,          VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  loadInfo);
      imgWrite(ds, B_SRV_DEINTERLEAVED_DEPTHS,  VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  deintDInfo);
      imgWrite(ds, B_SRV_DEINTERLEAVED_NORMALS, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  deintNInfo);
      imgWrite(ds, B_SRV_SSAO_PING,             VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  pongSrvInfo); // read pong
      imgWrite(ds, B_SRV_IMPORTANCE_MAP,        VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  impInfo);
      imgWrite(ds, B_UAV_SSAO_PING,             VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  pingUavInfo); // write ping
      bufWrite(ds, B_CB_CACAO, cbInfo);

      cmd->cmdBindPipeline(DxvkCmdBuffer::ExecBuffer,
        VK_PIPELINE_BIND_POINT_COMPUTE, pipelines[generatePass]);
      VkBindDescriptorSetsInfo bindInfo = { VK_STRUCTURE_TYPE_BIND_DESCRIPTOR_SETS_INFO };
      bindInfo.stageFlags          = VK_SHADER_STAGE_COMPUTE_BIT;
      bindInfo.layout            = pipeLayouts[generatePass];
      bindInfo.firstSet          = 0;
      bindInfo.descriptorSetCount = 1;
      bindInfo.pDescriptorSets   = &ds;
      cmd->cmdBindDescriptorSets(DxvkCmdBuffer::ExecBuffer, &bindInfo);
      cmd->cmdDispatch(DxvkCmdBuffer::ExecBuffer,
        dispatchSize(TG, bsiCopy.ssaoBufferWidth),
        dispatchSize(TG, bsiCopy.ssaoBufferHeight),
        4);  // z=4: dispatch once per deinterleaved layer
    }
    computeBarrier();

    // ════════════════════════════════════════════════════════════════════════
    // PASS 4 — Edge-sensitive blur (ping→pong, alternating, numBlurPasses times)
    // ════════════════════════════════════════════════════════════════════════
    const uint32_t numBlurPasses = (quality == CacaoQuality::Lowest) ? 2u : 4u;

    for (uint32_t blurPass = 0; blurPass < numBlurPasses; blurPass++) {
      bool even = (blurPass % 2) == 0;
      Rc<DxvkImage>& src = even ? ssaoPing : ssaoPong;
      Rc<DxvkImage>& dst = even ? ssaoPong : ssaoPing;

      VkDescriptorSet ds = frameSets[SET_IDX_BLUR_0 + blurPass];

      const Rc<DxvkImageView>& srcView = even ? vPingSrv : vPongSrv;
      const Rc<DxvkImageView>& dstView = even ? vPongUav : vPingUav;

      VkDescriptorImageInfo srcInfo = { {}, srcView->handle(), VK_IMAGE_LAYOUT_GENERAL };
      VkDescriptorImageInfo dstInfo = { {}, dstView->handle(), VK_IMAGE_LAYOUT_GENERAL };

      imgWrite(ds, B_SRV_SSAO_PING_APPLY, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  srcInfo);
      imgWrite(ds, B_UAV_SSAO_PONG,       VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  dstInfo);
      bufWrite(ds, B_CB_CACAO, cbInfo);

      cmd->cmdBindPipeline(DxvkCmdBuffer::ExecBuffer,
        VK_PIPELINE_BIND_POINT_COMPUTE, pipelines[CACAO_PASS_EDGE_BLUR]);
      VkBindDescriptorSetsInfo bindInfo = { VK_STRUCTURE_TYPE_BIND_DESCRIPTOR_SETS_INFO };
      bindInfo.stageFlags          = VK_SHADER_STAGE_COMPUTE_BIT;
      bindInfo.layout            = pipeLayouts[CACAO_PASS_EDGE_BLUR];
      bindInfo.firstSet          = 0;
      bindInfo.descriptorSetCount = 1;
      bindInfo.pDescriptorSets   = &ds;
      cmd->cmdBindDescriptorSets(DxvkCmdBuffer::ExecBuffer, &bindInfo);
      cmd->cmdDispatch(DxvkCmdBuffer::ExecBuffer,
        dispatchSize(TG, bsiCopy.ssaoBufferWidth),
        dispatchSize(TG, bsiCopy.ssaoBufferHeight),
        4);

      if (blurPass < numBlurPasses - 1) computeBarrier();
    }
    computeBarrier();

    // ════════════════════════════════════════════════════════════════════════
    // PASS 5 — Apply: upsample SSAO to full-res, write AO to m_aoFinal (R channel)
    // The Apply shader (rgba16f output) writes AO to R, preserving G/B/A from aoFinal.
    // Compositing is done in the present blitter via binding=4 / c_composite_ao=true.
    // ════════════════════════════════════════════════════════════════════════
    {
      // After the blur loop the result sits in the buffer written by the
      // LAST blur pass: generate wrote ssaoPing, each pass swaps dst, so
      // after an even pass count the result is back in ssaoPing. Reading
      // the wrong buffer (a previous intermediate) fed stale AO into the
      // composite and oscillated through the Q3 temporal history.
      Rc<DxvkImage>& finalSsao = (numBlurPasses % 2 == 0) ? ssaoPing : ssaoPong;
      const Rc<DxvkImageView>& ssaoSrv = (numBlurPasses % 2 == 0) ? vPingSrv : vPongSrv;

      VkDescriptorSet ds = frameSets[SET_IDX_APPLY];

      VkDescriptorImageInfo ssaoSrvInfo = { {}, ssaoSrv->handle(), VK_IMAGE_LAYOUT_GENERAL };
      VkDescriptorImageInfo aoUavInfo   = { {}, aoUavView->handle(), VK_IMAGE_LAYOUT_GENERAL };

      imgWrite(ds, B_SRV_SSAO_PING_APPLY, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  ssaoSrvInfo);
      imgWrite(ds, B_UAV_OUTPUT,          VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  aoUavInfo);
      bufWrite(ds, B_CB_CACAO, cbInfo);

      cmd->cmdBindPipeline(DxvkCmdBuffer::ExecBuffer,
        VK_PIPELINE_BIND_POINT_COMPUTE, pipelines[CACAO_PASS_APPLY]);
      VkBindDescriptorSetsInfo bindInfo = { VK_STRUCTURE_TYPE_BIND_DESCRIPTOR_SETS_INFO };
      bindInfo.stageFlags          = VK_SHADER_STAGE_COMPUTE_BIT;
      bindInfo.layout            = pipeLayouts[CACAO_PASS_APPLY];
      bindInfo.firstSet          = 0;
      bindInfo.descriptorSetCount = 1;
      bindInfo.pDescriptorSets   = &ds;
      cmd->cmdBindDescriptorSets(DxvkCmdBuffer::ExecBuffer, &bindInfo);
      cmd->cmdDispatch(DxvkCmdBuffer::ExecBuffer,
        dispatchSize(TG, w), dispatchSize(TG, h), 1);
    }

    // ── AO barrier: compute storage write → fragment shader sampled read ─
    {
      VkImageMemoryBarrier2 b = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
      b.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
      b.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
      b.dstStageMask  = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
      b.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
      b.oldLayout     = VK_IMAGE_LAYOUT_GENERAL;
      b.newLayout     = VK_IMAGE_LAYOUT_GENERAL;
      b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      b.image         = aoFinal->handle();
      b.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
      VkDependencyInfo di = { VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
      di.imageMemoryBarrierCount = 1;
      di.pImageMemoryBarriers    = &b;
      cmd->cmdPipelineBarrier(DxvkCmdBuffer::ExecBuffer, &di);
    }

    // ── No depth layout restore needed (we read from GENERAL, no transition was made) ──

    // ── CPU readback: multi-stage probe ────────────────────────────────────
    // Buffer layout: [0..3]   deintD  layer0 mip0 center pixel (R32F, viewZ)
    //                [16..23] ssao    layer0 center pixel (R32G32F, blurred AO)
    //                [64..575] aoFinal center strip 64x1 (RGBA16F, 8 B/px)
    if (readbackBuf) {
      Rc<DxvkImage> finalSsao = (numBlurPasses % 2 == 0) ? ssaoPing : ssaoPong;

      auto transferReadBarrier = [&](const Rc<DxvkImage>& img, VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT) {
        VkImageMemoryBarrier2 rb = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
        rb.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT
                         | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        rb.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT
                         | VK_ACCESS_2_SHADER_READ_BIT;
        rb.dstStageMask  = VK_PIPELINE_STAGE_2_COPY_BIT;
        rb.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
        rb.oldLayout = rb.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        rb.srcQueueFamilyIndex = rb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        rb.image = img->handle();
        rb.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        VkDependencyInfo rdi = { VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        rdi.imageMemoryBarrierCount = 1;
        rdi.pImageMemoryBarriers    = &rb;
        cmd->cmdPipelineBarrier(DxvkCmdBuffer::ExecBuffer, &rdi);
      };

      transferReadBarrier(deintD);
      transferReadBarrier(finalSsao);
      transferReadBarrier(aoFinal);
      if (useSnapInput)
        transferReadBarrier(depthCopy, VK_IMAGE_ASPECT_DEPTH_BIT);

      VkDeviceSize bufBase = readbackBuf->getSliceInfo().offset;
      VkBuffer dstBuf      = readbackBuf->getSliceInfo().buffer;

      auto copyRegion = [&](VkDeviceSize offset, const Rc<DxvkImage>& img,
                            VkImageAspectFlags aspect, int32_t x, int32_t y,
                            uint32_t cw, uint32_t ch) {
        VkBufferImageCopy2 region = { VK_STRUCTURE_TYPE_BUFFER_IMAGE_COPY_2 };
        region.bufferOffset      = bufBase + offset;
        region.bufferRowLength   = 0;
        region.bufferImageHeight = 0;
        region.imageSubresource  = { aspect, 0, 0, 1 };
        region.imageOffset       = { x, y, 0 };
        region.imageExtent       = { cw, ch, 1 };
        VkCopyImageToBufferInfo2 copyInfo = { VK_STRUCTURE_TYPE_COPY_IMAGE_TO_BUFFER_INFO_2 };
        copyInfo.srcImage       = img->handle();
        copyInfo.srcImageLayout = VK_IMAGE_LAYOUT_GENERAL;
        copyInfo.dstBuffer      = dstBuf;
        copyInfo.regionCount    = 1;
        copyInfo.pRegions       = &region;
        cmd->cmdCopyImageToBuffer(DxvkCmdBuffer::ExecBuffer, &copyInfo);
      };

      // 1) deinterleaved depth, layer 0, mip 0, center pixel
      if (doReadback) {
        copyRegion(0,  deintD,     VK_IMAGE_ASPECT_COLOR_BIT,
                   int32_t(bsiCopy.deinterleavedDepthBufferWidth  / 2),
                   int32_t(bsiCopy.deinterleavedDepthBufferHeight / 2), 1, 1);
        // 2) blurred SSAO (final half-res buffer), layer 0, center pixel
        copyRegion(16, finalSsao,  VK_IMAGE_ASPECT_COLOR_BIT,
                   int32_t(bsiCopy.ssaoBufferWidth  / 2),
                   int32_t(bsiCopy.ssaoBufferHeight / 2), 1, 1);
        // 3) aoFinal strip 64x1 from the vertical center
        copyRegion(64, aoFinal,    VK_IMAGE_ASPECT_COLOR_BIT,
                   int32_t(w / 2) - 32, int32_t(h / 2), 64, 1);
        // 4) snapshot depth center pixel (raw depth-aspect value)
        if (useSnapInput)
          copyRegion(32, depthCopy,  VK_IMAGE_ASPECT_DEPTH_BIT,
                     int32_t(w / 2), int32_t(h / 2), 1, 1);
        // 5) world-diagnostics rows: deintD 16x1 and blurred SSAO 16x1 at the
        //    center height (uniformity detection: an all-equal row means the
        //    deinterleave/tap pipeline collapsed to a constant field), and a
        //    second aoFinal strip at quarter height (upper scene area).
        copyRegion(1024, deintD,    VK_IMAGE_ASPECT_COLOR_BIT,
                   int32_t(bsiCopy.deinterleavedDepthBufferWidth / 2) - 8,
                   int32_t(bsiCopy.deinterleavedDepthBufferHeight / 2), 16, 1);
        copyRegion(1152, finalSsao, VK_IMAGE_ASPECT_COLOR_BIT,
                   int32_t(bsiCopy.ssaoBufferWidth / 2) - 8,
                   int32_t(bsiCopy.ssaoBufferHeight / 2), 16, 1);
        copyRegion(1536, aoFinal,   VK_IMAGE_ASPECT_COLOR_BIT,
                   int32_t(w / 2) - 32, int32_t(h / 4), 64, 1);
      }

      // 4) Full-buffer dump (rotating A/B) when enabled in config.
      //    Regions must not overlap the strip area at [0..576].
      if (doDump) {
        // deinterleaved depth, layer 0, mip 0
        copyRegion(1u * 1024u * 1024u,  deintD,    VK_IMAGE_ASPECT_COLOR_BIT, 0, 0,
                   bsiCopy.deinterleavedDepthBufferWidth,
                   bsiCopy.deinterleavedDepthBufferHeight);
        // post-blur SSAO, layer 0
        copyRegion(4u * 1024u * 1024u,  finalSsao, VK_IMAGE_ASPECT_COLOR_BIT, 0, 0,
                   bsiCopy.ssaoBufferWidth, bsiCopy.ssaoBufferHeight);
        // aoFinal, full extent
        copyRegion(9u * 1024u * 1024u,  aoFinal,   VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, w, h);
      }

      cmd->track(readbackBuf, DxvkAccess::Write);
    }

    // ── Notify DXVK that depth is back in DEPTH_STENCIL_ATTACHMENT_OPTIMAL ──
    // Without this DXVK's m_rtLayouts.depth is stale and it emits wrong
    // barriers on the next draw, causing UI/logo flickering.
    cs->notifyExternalDepthLayout(VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

    // ── Track resources (keeps alive until GPU fence signals) ─────────────
    cmd->track(depthImage, DxvkAccess::Read);
    cmd->track(deintD,     DxvkAccess::Write);
    cmd->track(deintN,     DxvkAccess::Write);
    cmd->track(ssaoPing,   DxvkAccess::Write);
    cmd->track(ssaoPong,   DxvkAccess::Write);
    cmd->track(impMap,     DxvkAccess::Write);
    cmd->track(impMapP,    DxvkAccess::Write);
    cmd->track(aoFinal,    DxvkAccess::Write);
    cmd->track(loadCnt,    DxvkAccess::Write);
    cmd->track(cb,         DxvkAccess::Read);
  });
}


// ── Mid-frame composite (docs/DX9_VULKAN_WINE_PATHS.md §4.B) ─────────────────
//
// Bakes color * AO into the D3D9 back buffer at the world->UI boundary, before
// the game draws its interface. Replaces the present blitter composite when
// enabled: UI pixels are never darkened, and letterbox/scaling artifacts are
// impossible by construction (the composite happens at back-buffer resolution).
void D3D9CacaoRenderer::InitMidComposite(VkFormat format, VkExtent3D extent) {
  auto vkd = m_device->vkd();

  DestroyMidComposite();

  // Descriptor layout: set 1 bindings 0/1 = scene copy + AO result,
  // samplers come from the sampler heap (set 0), params via push constants.
  static const std::array<DxvkDescriptorSetLayoutBinding, 2> bindings = {{
    { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_FRAGMENT_BIT },
    { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_FRAGMENT_BIT },
  }};

  m_midLayout = m_device->createBuiltInPipelineLayout(
    DxvkPipelineLayoutFlag::UsesSamplerHeap,
    VK_SHADER_STAGE_FRAGMENT_BIT, sizeof(MidCompositeParams),
    bindings.size(), bindings.data());

  DxvkSamplerKey samplerInfo = { };
  samplerInfo.setFilter(VK_FILTER_LINEAR, VK_FILTER_LINEAR,
    VK_SAMPLER_MIPMAP_MODE_NEAREST);
  samplerInfo.setAddressModes(
    VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
    VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
    VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
  samplerInfo.setUsePixelCoordinates(false);
  m_midSampler = m_device->createSampler(samplerInfo);

  // Scene copies: one ring slot per frame slot, sized as the back buffer.
  for (uint32_t i = 0; i < kFrameLag; i++) {
    DxvkImageCreateInfo ici = { };
    ici.type        = VK_IMAGE_TYPE_2D;
    ici.format      = format;
    ici.sampleCount = VK_SAMPLE_COUNT_1_BIT;
    ici.extent      = extent;
    ici.numLayers   = 1;
    ici.mipLevels   = 1;
    ici.usage       = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    ici.stages      = VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    ici.access      = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
    ici.tiling      = VK_IMAGE_TILING_OPTIMAL;
    ici.layout      = VK_IMAGE_LAYOUT_GENERAL;
    m_sceneCopies[i] = m_device->createImage(ici, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    DxvkImageViewKey vkey = { };
    vkey.viewType   = VK_IMAGE_VIEW_TYPE_2D;
    vkey.format     = format;
    vkey.usage      = VK_IMAGE_USAGE_SAMPLED_BIT;
    vkey.aspects    = VK_IMAGE_ASPECT_COLOR_BIT;
    vkey.mipIndex   = 0u;
    vkey.mipCount   = 1u;
    vkey.layerIndex = 0u;
    vkey.layerCount = 1u;
    m_sceneSrvs[i] = m_sceneCopies[i]->createView(vkey);
  }

  util::DxvkBuiltInGraphicsState state = { };
  state.vs = util::DxvkBuiltInShaderStage(DXVK_CACAO_MID_VERT_SPV, nullptr);
  state.fs = util::DxvkBuiltInShaderStage(DXVK_CACAO_MID_FRAG_SPV, nullptr);
  state.colorFormats[0] = format;
  m_midPipeline = m_device->createBuiltInGraphicsPipeline(m_midLayout, state);

  m_midFormat = format;
  m_midExtent = extent;
  m_midInit   = true;
}


void D3D9CacaoRenderer::DestroyMidComposite() {
  auto vkd = m_device->vkd();
  if (m_midPipeline) {
    vkd->vkDestroyPipeline(vkd->device(), m_midPipeline, nullptr);
    m_midPipeline = VK_NULL_HANDLE;
  }
  for (uint32_t i = 0; i < kFrameLag; i++) {
    m_sceneSrvs[i]   = nullptr;
    m_sceneCopies[i] = nullptr;
  }
  m_bbRtv    = nullptr;
  m_bbImage  = nullptr;
  m_midSampler = nullptr;
  m_midLayout  = nullptr;
  m_midInit  = false;
}


void D3D9CacaoRenderer::CompositeMidFrame(
    D3D9DeviceEx*        device,
    const Rc<DxvkImage>& backBuffer,
    float                darkMask,
    float                contactT0,
    float                contactT1,
    float                contactStrength,
    bool                 debugView) {
  if (!m_initialized || backBuffer == nullptr)
    return;

  VkFormat  bbFormat = backBuffer->info().format;
  VkExtent3D bbExtent = backBuffer->info().extent;

  if (!m_midInit || m_midFormat != bbFormat
   || m_midExtent.width != bbExtent.width
   || m_midExtent.height != bbExtent.height
   || m_bbImage != backBuffer)
    InitMidComposite(bbFormat, bbExtent);

  if (!m_midInit)
    return;

  // RTV on the back buffer (cached per image instance)
  if (m_bbImage != backBuffer) {
    DxvkImageViewKey rkey = { };
    rkey.viewType   = VK_IMAGE_VIEW_TYPE_2D;
    rkey.format     = bbFormat;
    rkey.usage      = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    rkey.aspects    = VK_IMAGE_ASPECT_COLOR_BIT;
    rkey.mipIndex   = 0u;
    rkey.mipCount   = 1u;
    rkey.layerIndex = 0u;
    rkey.layerCount = 1u;
    m_bbRtv   = backBuffer->createView(rkey);
    m_bbImage = backBuffer;
  }

  uint32_t frame  = m_frameIndex;               // same slot Apply just wrote
  Rc<DxvkImage>     sceneCopy = m_sceneCopies[frame];
  Rc<DxvkImageView> sceneSrv  = m_sceneSrvs[frame];
  Rc<DxvkImageView> aoSrv     = m_aoFinalSrvs[frame];
  VkPipeline  pipeline  = m_midPipeline;
  const DxvkPipelineLayout* layout = m_midLayout;
  uint32_t samplerIdx = m_midSampler->getDescriptor().samplerIndex;

  MidCompositeParams params = { };
  params.srcExtent[0] = int32_t(bbExtent.width);
  params.srcExtent[1] = int32_t(bbExtent.height);
  params.samplerScene = samplerIdx;
  params.darkMask  = darkMask;
  params.contactT0 = contactT0;
  params.contactT1 = contactT1;
  params.contactStrength = contactStrength;
  params.debugView = debugView ? 1.0f : 0.0f;

  Rc<DxvkImage> bb = backBuffer;
  device->EmitCs([
    bb, bbRtv = m_bbRtv, sceneCopy, sceneSrv, aoSrv, pipeline, layout,
    params
  ] (DxvkContext* cs) mutable {
    Rc<DxvkCommandList> cmd = cs->beginExternalRendering();

    VkImageLayout tracked = bb->queryLayout(bb->getAvailableSubresources());

    auto bbBarrier = [&](VkImageLayout oldL, VkImageLayout newL,
                         VkPipelineStageFlags2 srcStage, VkAccessFlags2 srcAccess,
                         VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess) {
      VkImageMemoryBarrier2 b = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
      b.srcStageMask  = srcStage;
      b.srcAccessMask = srcAccess;
      b.dstStageMask  = dstStage;
      b.dstAccessMask = dstAccess;
      b.oldLayout     = oldL;
      b.newLayout     = newL;
      b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      b.image         = bb->handle();
      b.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
      VkDependencyInfo di = { VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
      di.imageMemoryBarrierCount = 1;
      di.pImageMemoryBarriers    = &b;
      cmd->cmdPipelineBarrier(DxvkCmdBuffer::ExecBuffer, &di);
    };

    // Tracked layout -> transfer source for the scene snapshot
    bbBarrier(tracked, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
      VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
      VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_SHADER_READ_BIT,
      VK_PIPELINE_STAGE_2_TRANSFER_BIT,
      VK_ACCESS_2_TRANSFER_READ_BIT);

    // Snapshot the back buffer (the pre-UI world image)
    VkImageCopy2 region = { VK_STRUCTURE_TYPE_IMAGE_COPY_2 };
    region.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    region.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    region.extent         = sceneCopy->info().extent;
    VkCopyImageInfo2 copyInfo = { VK_STRUCTURE_TYPE_COPY_IMAGE_INFO_2 };
    copyInfo.srcImage       = bb->handle();
    copyInfo.srcImageLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    copyInfo.dstImage       = sceneCopy->handle();
    copyInfo.dstImageLayout = VK_IMAGE_LAYOUT_GENERAL;
    copyInfo.regionCount    = 1;
    copyInfo.pRegions       = &region;
    cmd->cmdCopyImage(DxvkCmdBuffer::ExecBuffer, &copyInfo);

    // Transfer source -> color attachment for the composite draw
    bbBarrier(VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      VK_PIPELINE_STAGE_2_TRANSFER_BIT,
      VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT,
      VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
      VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT);

    // Scene copy: transfer write -> shader read
    {
      VkImageMemoryBarrier2 sc = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
      sc.srcStageMask  = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
      sc.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
      sc.dstStageMask  = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
      sc.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
      sc.oldLayout     = VK_IMAGE_LAYOUT_GENERAL;
      sc.newLayout     = VK_IMAGE_LAYOUT_GENERAL;
      sc.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      sc.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      sc.image         = sceneCopy->handle();
      sc.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
      VkDependencyInfo di = { VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
      di.imageMemoryBarrierCount = 1;
      di.pImageMemoryBarriers    = &sc;
      cmd->cmdPipelineBarrier(DxvkCmdBuffer::ExecBuffer, &di);
    }

    VkRenderingAttachmentInfo att = { VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
    att.imageView   = bbRtv->handle();
    att.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    att.loadOp      = VK_ATTACHMENT_LOAD_OP_LOAD;
    att.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;

    VkRenderingInfo renderInfo = { VK_STRUCTURE_TYPE_RENDERING_INFO };
    renderInfo.renderArea.offset = { 0u, 0u };
    renderInfo.renderArea.extent = { sceneCopy->info().extent.width,
                                     sceneCopy->info().extent.height };
    renderInfo.layerCount = 1;
    renderInfo.colorAttachmentCount = 1;
    renderInfo.pColorAttachments = &att;

    cmd->cmdBeginRendering(DxvkCmdBuffer::ExecBuffer, &renderInfo);

    VkViewport viewport = { };
    viewport.x = 0.0f; viewport.y = 0.0f;
    viewport.width  = float(sceneCopy->info().extent.width);
    viewport.height = float(sceneCopy->info().extent.height);
    viewport.minDepth = 0.0f; viewport.maxDepth = 1.0f;
    cmd->cmdSetViewport(1, &viewport);

    VkRect2D scissor = { };
    scissor.offset = { 0, 0 };
    scissor.extent = { sceneCopy->info().extent.width,
                       sceneCopy->info().extent.height };
    cmd->cmdSetScissor(1, &scissor);

    cmd->cmdBindPipeline(DxvkCmdBuffer::ExecBuffer,
      VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

    std::array<DxvkDescriptorWrite, 2> writes = { };
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    writes[0].descriptor     = sceneSrv->getDescriptor();
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    writes[1].descriptor     = aoSrv->getDescriptor();

    cmd->bindResources(DxvkCmdBuffer::ExecBuffer,
      layout, writes.size(), writes.data(), sizeof(params), &params);

    cmd->cmdDraw(3, 1, 0, 0);

    cmd->cmdEndRendering(DxvkCmdBuffer::ExecBuffer);

    // Restore the tracked layout on the back buffer
    bbBarrier(VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, tracked,
      VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
      VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT,
      VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
      VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);

    cmd->track(bb,        DxvkAccess::Write);
    cmd->track(sceneCopy, DxvkAccess::Write);
    cmd->track(sceneSrv->image(), DxvkAccess::Read);
    cmd->track(aoSrv->image(),    DxvkAccess::Read);
  });
}

} // namespace dxvk
