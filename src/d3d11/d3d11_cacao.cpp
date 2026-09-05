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

#include "d3d11_cacao.h"
#include "d3d11_device.h"
#include "../d3d9/d3d9_cacao_log.h"
#include "../dxvk/hud/dxvk_hud.h"

#include "../dxvk/dxvk_device.h"
#include "../dxvk/dxvk_context.h"
#include "../dxvk/dxvk_cmdlist.h"
#include "../dxvk/dxvk_image.h"
#include "../dxvk/dxvk_buffer.h"

#include <cmath>
#include <cstring>
#include <algorithm>

// ── Compiled SPIR-V headers (compiled from FidelityFX SDK GLSL) ───────────────
#include "../d3d9/cacao_spirv/ffx_cacao_prepare_native_depths_and_mips_pass.h"
#include "../d3d9/cacao_spirv/ffx_cacao_prepare_native_normals_pass.h"
#include "../d3d9/cacao_spirv/ffx_cacao_generate_q3_pass.h"
#include "../d3d9/cacao_spirv/ffx_cacao_generate_importance_map_pass.h"
#include "../d3d9/cacao_spirv/ffx_cacao_generate_importance_map_a_pass.h"
#include "../d3d9/cacao_spirv/ffx_cacao_generate_importance_map_b_pass.h"
#include "../d3d9/cacao_spirv/ffx_cacao_edge_sensitive_blur_8_pass.h"
#include "../d3d9/cacao_spirv/ffx_cacao_apply_pass.h"
#include "../d3d9/cacao_spirv/ffx_cacao_clear_load_counter_pass.h"

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
CacaoRenderer::CacaoRenderer(DxvkDevice* device, CacaoQuality quality)
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

CacaoRenderer::~CacaoRenderer() {
  DestroyResources();

  auto vkd = m_device->vkd();
  VkDevice dev = vkd->device();

  if (m_descPool)
    vkd->vkDestroyDescriptorPool(dev, m_descPool, nullptr);

  for (auto& pl : m_pipelines)
    if (pl) vkd->vkDestroyPipeline(dev, pl, nullptr);
  for (auto& pl : m_pipelineLayouts)
    if (pl) vkd->vkDestroyPipelineLayout(dev, pl, nullptr);
  for (auto& sl : m_descSetLayouts)
    if (sl) vkd->vkDestroyDescriptorSetLayout(dev, sl, nullptr);
  for (auto& s : m_descSetLayouts)
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
void CacaoRenderer::InitPipelines() {
  auto vkd = m_device->vkd();
  VkDevice dev = vkd->device();

  // ── Samplers ─────────────────────────────────────────────────────────────
  m_descSetLayouts[0] = makeSampler(vkd, VK_FILTER_NEAREST, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
  m_descSetLayouts[1] = makeSampler(vkd, VK_FILTER_NEAREST, VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT);
  m_descSetLayouts[2] = makeSampler(vkd, VK_FILTER_LINEAR,  VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, 1000.0f);
  m_descSetLayouts[3] = makeSampler(vkd, VK_FILTER_LINEAR,  VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
  m_descSetLayouts[4] = makeSampler(vkd, VK_FILTER_NEAREST, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);

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
    return makeSetLayout(vkd, passDefs, m_descSetLayouts.data(), B_SAMPLER_POINT_CLAMP, 5);
  };

  // PASS_PREPARE_DEPTHS
  m_descSetLayouts[CACAO_PASS_PREPARE_DEPTHS] = buildLayout({
    { B_SRV_DEPTH_IN,     VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT },
    { B_UAV_DEPTH_MIPS,   VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  4, VK_SHADER_STAGE_COMPUTE_BIT },
    { B_UAV_DEINT_DEPTHS, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT },
  });

  // PASS_PREPARE_NORMALS
  m_descSetLayouts[CACAO_PASS_PREPARE_NORMALS] = buildLayout({
    { B_SRV_DEPTH_IN,       VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { B_UAV_DEINT_NORMALS,  VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
  });

  // PASS_GENERATE_Q3
  m_descSetLayouts[CACAO_PASS_GENERATE_Q3] = buildLayout({
    { B_SRV_LOAD_COUNTER,          VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT },
    { B_SRV_DEINTERLEAVED_DEPTHS,  VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT },
    { B_SRV_DEINTERLEAVED_NORMALS, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT },
    { B_SRV_SSAO_PING,             VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT },
    { B_SRV_IMPORTANCE_MAP,        VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT },
    { B_UAV_SSAO_PING,             VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT },
  });

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
    m_descSetLayouts[CACAO_PASS_IMPORTANCE_MAP]   = buildLayout(impDefs);
    m_descSetLayouts[CACAO_PASS_IMPORTANCE_MAP_A] = buildLayout(impDefs);  // separate handle
    m_descSetLayouts[CACAO_PASS_IMPORTANCE_MAP_B] = buildLayout(impDefs);  // separate handle
  }

  // PASS_EDGE_BLUR: ping → pong
  m_descSetLayouts[CACAO_PASS_EDGE_BLUR] = buildLayout({
    { B_SRV_SSAO_PING_APPLY,  VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT },
    { B_UAV_SSAO_PONG,        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT },
  });

  // PASS_APPLY: ssao → aoFinal
  m_descSetLayouts[CACAO_PASS_APPLY] = buildLayout({
    { B_SRV_SSAO_PING_APPLY,  VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT },
    { B_UAV_OUTPUT,           VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT },
  });

  // ── Pipeline layouts (one per pass) ───────────────────────────────────────
  for (uint32_t i = 0; i < CACAO_PASS_COUNT; i++)
    m_pipelineLayouts[i] = makePipelineLayout(vkd, m_descSetLayouts[i]);

  // ── Compute pipelines ──────────────────────────────────────────────────────
#define MAKE_PIPELINE(pass, spv) \
  m_pipelines[pass] = makeComputePipeline(vkd, m_pipelineLayouts[pass], \
    spv, sizeof(spv))

  MAKE_PIPELINE(CACAO_PASS_PREPARE_DEPTHS,    FFX_CACAO_PREPARE_NATIVE_DEPTHS_AND_MIPS_PASS_SPV);
  MAKE_PIPELINE(CACAO_PASS_PREPARE_NORMALS,   FFX_CACAO_PREPARE_NATIVE_NORMALS_PASS_SPV);
  MAKE_PIPELINE(CACAO_PASS_GENERATE_Q3,       FFX_CACAO_GENERATE_Q3_PASS_SPV);
  MAKE_PIPELINE(CACAO_PASS_IMPORTANCE_MAP,    FFX_CACAO_GENERATE_IMPORTANCE_MAP_PASS_SPV);
  MAKE_PIPELINE(CACAO_PASS_IMPORTANCE_MAP_A,  FFX_CACAO_GENERATE_IMPORTANCE_MAP_A_PASS_SPV);
  MAKE_PIPELINE(CACAO_PASS_IMPORTANCE_MAP_B,  FFX_CACAO_GENERATE_IMPORTANCE_MAP_B_PASS_SPV);
  MAKE_PIPELINE(CACAO_PASS_EDGE_BLUR,         FFX_CACAO_EDGE_SENSITIVE_BLUR_8_PASS_SPV);
  MAKE_PIPELINE(CACAO_PASS_APPLY,             FFX_CACAO_APPLY_PASS_SPV);

#undef MAKE_PIPELINE

  // ── Pre-allocate descriptor sets (double-buffered) ────────────────────────
  // Layout sequence per frame: DEPTHS, NORMALS, Q3, BLUR×4, APPLY
  VkDescriptorSetLayout frameLayouts[kSetsPerFrame] = {
    m_descSetLayouts[CACAO_PASS_PREPARE_DEPTHS],
    m_descSetLayouts[CACAO_PASS_PREPARE_NORMALS],
    m_descSetLayouts[CACAO_PASS_GENERATE_Q3],
    m_descSetLayouts[CACAO_PASS_EDGE_BLUR],
    m_descSetLayouts[CACAO_PASS_EDGE_BLUR],
    m_descSetLayouts[CACAO_PASS_EDGE_BLUR],
    m_descSetLayouts[CACAO_PASS_EDGE_BLUR],
    m_descSetLayouts[CACAO_PASS_APPLY],
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
void CacaoRenderer::InitResources(uint32_t w, uint32_t h) {
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
    VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);

  m_deinterleavedNormals = CreateIntermediateImage(hw, hh,
    VK_FORMAT_R32G32B32A32_SFLOAT, 4, 1,
    VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);

  m_ssaoPing = CreateIntermediateImage(hw, hh,
    VK_FORMAT_R32G32_SFLOAT, 4, 1,
    VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);

  m_ssaoPong = CreateIntermediateImage(hw, hh,
    VK_FORMAT_R32G32_SFLOAT, 4, 1,
    VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);

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
      VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
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

  // Readback buffer: 8 bytes = one RGBA16F pixel (4 × float16 = 8 bytes)
  DxvkBufferCreateInfo rbci = {};
  rbci.size   = 8u;
  rbci.usage  = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  rbci.stages = VK_PIPELINE_STAGE_TRANSFER_BIT;
  rbci.access = VK_ACCESS_TRANSFER_WRITE_BIT;
  m_readbackBuffer  = m_device->createBuffer(rbci,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  m_readbackPending = false;
}

// ── DestroyResources ───────────────────────────────────────────────────────────
void CacaoRenderer::DestroyResources() {
  m_deinterleavedDepths  = nullptr;
  m_deinterleavedNormals = nullptr;
  m_ssaoPing             = nullptr;
  m_ssaoPong             = nullptr;
  m_importanceMap        = nullptr;
  m_importanceMapPong    = nullptr;
  m_aoFinalSrvs[0]     = nullptr;
  m_aoFinalSrvs[1]     = nullptr;
  m_aoFinals[0]        = nullptr;
  m_aoFinals[1]        = nullptr;
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
Rc<DxvkImage> CacaoRenderer::CreateIntermediateImage(
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
void CacaoRenderer::UpdateConstants(
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
  c->EffectRadius                       = worldRadius;
  c->EffectShadowStrength               = m_strength * 2.0f;  // user 0-2 → internal 0-4
  c->EffectShadowPow                    = 1.5f;
  c->EffectShadowClamp                  = m_shadowClamp;
  c->EffectFadeOutMul                   = -1.0f / (worldRadius * 20.0f);  // fade over 20x radius
  c->EffectFadeOutAdd                   = 1.0f;
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
void CacaoRenderer::FillAoBuffer(D3D9DeviceEx* device, float value) {
  if (!m_initialized || !m_enabled || !m_aoFinals[m_frameIndex])
    return;

  auto vkd = m_device->vkd();
  Rc<DxvkImage> aoFinal = m_aoFinals[m_frameIndex];

  ctx->EmitCs([
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

// ── Apply ──────────────────────────────────────────────────────────────────────
void CacaoRenderer::Apply(
    DxvkContext*   ctx,
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

  // ── CPU readback: log AO pixel value from previous frame's GPU copy ────
  if (m_readbackPending && m_readbackBuffer) {
    m_readbackPending = false;
    // RGBA16F: 4 × uint16_t; R channel = AO value
    const uint16_t* px = reinterpret_cast<const uint16_t*>(m_readbackBuffer->mapPtr(0));
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
    float r = half2float(px[0]);
    float g = half2float(px[1]);
    float b = half2float(px[2]);
    float a = half2float(px[3]);
    CacaoLogger::get().log(str::format(
      "[READBACK] aoFinal center pixel: R=", r, " G=", g, " B=", b, " A=", a,
      " (AO=R, expect 0.0..1.0; 1.0=no occlusion, 0.0=full shadow)"));
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

  // Decide whether to issue a readback this frame
  m_readbackFrameCount++;
  bool doReadback = (m_readbackFrameCount % kReadbackInterval == 0) && m_readbackBuffer;
  Rc<DxvkBuffer> readbackBuf = doReadback ? m_readbackBuffer : nullptr;
  if (doReadback)
    m_readbackPending = true;

  ctx->EmitCs([
    vkd, depthImage = std::move(depthImage), depthCopy,
    deintD, deintN, ssaoPing, ssaoPong, impMap, impMapP, aoFinal, loadCnt, cb,
    pipelines, pipeLayouts,
    frameSets /* copied by value (array of VkDescriptorSet handles) */,
    w, h, bsiCopy, quality, debugFill,
    readbackBuf
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

    const VkFormat fmtDeintD = VK_FORMAT_R32_SFLOAT;
    const VkFormat fmtDeintN = VK_FORMAT_R32G32B32A32_SFLOAT;
    const VkFormat fmtSsao   = VK_FORMAT_R32G32_SFLOAT;
    const VkFormat fmtLoad   = VK_FORMAT_R32_UINT;
    const VkFormat fmtAo     = VK_FORMAT_R16G16B16A16_SFLOAT;

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

    // Use original depth image format for shader sampling
    const VkFormat fmtDepthSrv = depthImage->info().format;

    // ════════════════════════════════════════════════════════════════════════
    // PASS 1 — Prepare native depths and mips
    // ════════════════════════════════════════════════════════════════════════
    {
      VkDescriptorSet ds = frameSets[SET_IDX_DEPTHS];

      // depth SRV from original depth image (DEPTH aspect, GENERAL layout)
      auto depthSrv = makeView(depthImage, fmtDepthSrv, VK_IMAGE_VIEW_TYPE_2D,
        VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_LAYOUT_GENERAL,
        0, 1, 0, 1, VK_IMAGE_ASPECT_DEPTH_BIT);
      VkDescriptorImageInfo depthSrvInfo = { {}, depthSrv->handle(), VK_IMAGE_LAYOUT_GENERAL };
      imgWrite(ds, B_SRV_DEPTH_IN, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, depthSrvInfo);

      // deinterleaved depths array UAV (all 4 layers, mip 0)
      auto deintDUav = makeView(deintD, fmtDeintD, VK_IMAGE_VIEW_TYPE_2D_ARRAY,
        VK_IMAGE_USAGE_STORAGE_BIT, VK_IMAGE_LAYOUT_GENERAL, 0, 4);
      VkDescriptorImageInfo deintDUavInfo = { {}, deintDUav->handle(), VK_IMAGE_LAYOUT_GENERAL };
      imgWrite(ds, B_UAV_DEINT_DEPTHS, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, deintDUavInfo);

      // depth mips UAV: 4 array elements, one per mip level
      // (descriptor array binding with count=4, each element = one mip view)
      std::array<Rc<DxvkImageView>, 4> mipViews;
      std::array<VkDescriptorImageInfo, 4> mipInfos = {};
      for (uint32_t m = 0; m < 4; m++) {
        mipViews[m] = makeView(deintD, fmtDeintD, VK_IMAGE_VIEW_TYPE_2D_ARRAY,
          VK_IMAGE_USAGE_STORAGE_BIT, VK_IMAGE_LAYOUT_GENERAL, 0, 4, m, 1);
        mipInfos[m] = { {}, mipViews[m]->handle(), VK_IMAGE_LAYOUT_GENERAL };
      }
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
      cmd->cmdBindDescriptorSets(DxvkCmdBuffer::ExecBuffer,
        VK_PIPELINE_BIND_POINT_COMPUTE, pipeLayouts[CACAO_PASS_PREPARE_DEPTHS], 0, 1, &ds);
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
      auto depthSrv = makeView(depthImage, fmtDepthSrv, VK_IMAGE_VIEW_TYPE_2D,
        VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_LAYOUT_GENERAL,
        0, 1, 0, 1, VK_IMAGE_ASPECT_DEPTH_BIT);
      VkDescriptorImageInfo depthSrvInfo = { {}, depthSrv->handle(), VK_IMAGE_LAYOUT_GENERAL };
      imgWrite(ds, B_SRV_DEPTH_IN, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, depthSrvInfo);

      auto deintNUav = makeView(deintN, fmtDeintN, VK_IMAGE_VIEW_TYPE_2D_ARRAY,
        VK_IMAGE_USAGE_STORAGE_BIT, VK_IMAGE_LAYOUT_GENERAL, 0, 4);
      VkDescriptorImageInfo deintNUavInfo = { {}, deintNUav->handle(), VK_IMAGE_LAYOUT_GENERAL };
      imgWrite(ds, B_UAV_DEINT_NORMALS, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, deintNUavInfo);

      bufWrite(ds, B_CB_CACAO, cbInfo);

      cmd->cmdBindPipeline(DxvkCmdBuffer::ExecBuffer,
        VK_PIPELINE_BIND_POINT_COMPUTE, pipelines[CACAO_PASS_PREPARE_NORMALS]);
      cmd->cmdBindDescriptorSets(DxvkCmdBuffer::ExecBuffer,
        VK_PIPELINE_BIND_POINT_COMPUTE, pipeLayouts[CACAO_PASS_PREPARE_NORMALS], 0, 1, &ds);
      cmd->cmdDispatch(DxvkCmdBuffer::ExecBuffer,
        dispatchSize(TG, bsiCopy.ssaoBufferWidth),
        dispatchSize(TG, bsiCopy.ssaoBufferHeight),
        1);
    }
    computeBarrier();

    // ════════════════════════════════════════════════════════════════════════
    // PASS 3 — Generate Q3 AO (z=4: all 4 deinterleaved layers at once)
    //          Reads: deintD (SRV), deintN (SRV), ssaoPong (SRV, unused in non-adaptive)
    //          Writes: ssaoPing (UAV)
    // ════════════════════════════════════════════════════════════════════════
    {
      VkDescriptorSet ds = frameSets[SET_IDX_Q3];

      auto deintDSrv  = makeView(deintD,  fmtDeintD, VK_IMAGE_VIEW_TYPE_2D_ARRAY,
        VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_LAYOUT_GENERAL, 0, 4);
      auto deintNSrv  = makeView(deintN,  fmtDeintN, VK_IMAGE_VIEW_TYPE_2D_ARRAY,
        VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_LAYOUT_GENERAL, 0, 4);
      auto pingUav    = makeView(ssaoPing, fmtSsao, VK_IMAGE_VIEW_TYPE_2D_ARRAY,
        VK_IMAGE_USAGE_STORAGE_BIT, VK_IMAGE_LAYOUT_GENERAL, 0, 4);
      auto pongSrv    = makeView(ssaoPong, fmtSsao, VK_IMAGE_VIEW_TYPE_2D_ARRAY,
        VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_LAYOUT_GENERAL, 0, 4);
      auto impSrv     = makeView(impMap,  VK_FORMAT_R32_SFLOAT, VK_IMAGE_VIEW_TYPE_2D,
        VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_LAYOUT_GENERAL, 0, 1);
      auto loadSrv    = makeView(loadCnt, fmtLoad, VK_IMAGE_VIEW_TYPE_1D,
        VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_LAYOUT_GENERAL, 0, 1);

      VkDescriptorImageInfo deintDInfo  = { {}, deintDSrv->handle(),  VK_IMAGE_LAYOUT_GENERAL };
      VkDescriptorImageInfo deintNInfo  = { {}, deintNSrv->handle(),  VK_IMAGE_LAYOUT_GENERAL };
      VkDescriptorImageInfo pingUavInfo = { {}, pingUav->handle(),    VK_IMAGE_LAYOUT_GENERAL };
      VkDescriptorImageInfo pongSrvInfo = { {}, pongSrv->handle(),    VK_IMAGE_LAYOUT_GENERAL };
      VkDescriptorImageInfo impInfo     = { {}, impSrv->handle(),     VK_IMAGE_LAYOUT_GENERAL };
      VkDescriptorImageInfo loadInfo    = { {}, loadSrv->handle(),    VK_IMAGE_LAYOUT_GENERAL };

      imgWrite(ds, B_SRV_LOAD_COUNTER,          VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  loadInfo);
      imgWrite(ds, B_SRV_DEINTERLEAVED_DEPTHS,  VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  deintDInfo);
      imgWrite(ds, B_SRV_DEINTERLEAVED_NORMALS, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  deintNInfo);
      imgWrite(ds, B_SRV_SSAO_PING,             VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  pongSrvInfo); // read pong
      imgWrite(ds, B_SRV_IMPORTANCE_MAP,        VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  impInfo);
      imgWrite(ds, B_UAV_SSAO_PING,             VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  pingUavInfo); // write ping
      bufWrite(ds, B_CB_CACAO, cbInfo);

      cmd->cmdBindPipeline(DxvkCmdBuffer::ExecBuffer,
        VK_PIPELINE_BIND_POINT_COMPUTE, pipelines[CACAO_PASS_GENERATE_Q3]);
      cmd->cmdBindDescriptorSets(DxvkCmdBuffer::ExecBuffer,
        VK_PIPELINE_BIND_POINT_COMPUTE, pipeLayouts[CACAO_PASS_GENERATE_Q3], 0, 1, &ds);
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

      auto srcView = makeView(src, fmtSsao, VK_IMAGE_VIEW_TYPE_2D_ARRAY,
        VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_LAYOUT_GENERAL, 0, 4);
      auto dstView = makeView(dst, fmtSsao, VK_IMAGE_VIEW_TYPE_2D_ARRAY,
        VK_IMAGE_USAGE_STORAGE_BIT, VK_IMAGE_LAYOUT_GENERAL, 0, 4);

      VkDescriptorImageInfo srcInfo = { {}, srcView->handle(), VK_IMAGE_LAYOUT_GENERAL };
      VkDescriptorImageInfo dstInfo = { {}, dstView->handle(), VK_IMAGE_LAYOUT_GENERAL };

      imgWrite(ds, B_SRV_SSAO_PING_APPLY, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  srcInfo);
      imgWrite(ds, B_UAV_SSAO_PONG,       VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  dstInfo);
      bufWrite(ds, B_CB_CACAO, cbInfo);

      cmd->cmdBindPipeline(DxvkCmdBuffer::ExecBuffer,
        VK_PIPELINE_BIND_POINT_COMPUTE, pipelines[CACAO_PASS_EDGE_BLUR]);
      cmd->cmdBindDescriptorSets(DxvkCmdBuffer::ExecBuffer,
        VK_PIPELINE_BIND_POINT_COMPUTE, pipeLayouts[CACAO_PASS_EDGE_BLUR], 0, 1, &ds);
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
      // After blur, finalSsao is in ssaoPong (if numBlurPasses even) or ssaoPing (if odd)
      Rc<DxvkImage>& finalSsao = (numBlurPasses % 2 == 0) ? ssaoPong : ssaoPing;

      VkDescriptorSet ds = frameSets[SET_IDX_APPLY];

      auto ssaoSrv = makeView(finalSsao, fmtSsao, VK_IMAGE_VIEW_TYPE_2D_ARRAY,
        VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_LAYOUT_GENERAL, 0, 4);
      auto aoUav   = makeView(aoFinal, fmtAo, VK_IMAGE_VIEW_TYPE_2D,
        VK_IMAGE_USAGE_STORAGE_BIT, VK_IMAGE_LAYOUT_GENERAL, 0, 1);

      VkDescriptorImageInfo ssaoSrvInfo = { {}, ssaoSrv->handle(), VK_IMAGE_LAYOUT_GENERAL };
      VkDescriptorImageInfo aoUavInfo   = { {}, aoUav->handle(),   VK_IMAGE_LAYOUT_GENERAL };

      imgWrite(ds, B_SRV_SSAO_PING_APPLY, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  ssaoSrvInfo);
      imgWrite(ds, B_UAV_OUTPUT,          VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  aoUavInfo);
      bufWrite(ds, B_CB_CACAO, cbInfo);

      cmd->cmdBindPipeline(DxvkCmdBuffer::ExecBuffer,
        VK_PIPELINE_BIND_POINT_COMPUTE, pipelines[CACAO_PASS_APPLY]);
      cmd->cmdBindDescriptorSets(DxvkCmdBuffer::ExecBuffer,
        VK_PIPELINE_BIND_POINT_COMPUTE, pipeLayouts[CACAO_PASS_APPLY], 0, 1, &ds);
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

    // ── CPU readback: copy center pixel of aoFinal to host-visible buffer ───
    if (readbackBuf) {
      // Barrier: fragment-read → transfer-read
      VkImageMemoryBarrier2 rb = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
      rb.srcStageMask  = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
      rb.srcAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
      rb.dstStageMask  = VK_PIPELINE_STAGE_2_COPY_BIT;
      rb.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
      rb.oldLayout = rb.newLayout = VK_IMAGE_LAYOUT_GENERAL;
      rb.srcQueueFamilyIndex = rb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      rb.image = aoFinal->handle();
      rb.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
      VkDependencyInfo rdi = { VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
      rdi.imageMemoryBarrierCount = 1;
      rdi.pImageMemoryBarriers    = &rb;
      cmd->cmdPipelineBarrier(DxvkCmdBuffer::ExecBuffer, &rdi);

      // Copy 1 pixel from center of aoFinal
      VkBufferImageCopy2 region = { VK_STRUCTURE_TYPE_BUFFER_IMAGE_COPY_2 };
      region.bufferOffset      = readbackBuf->getSliceInfo().offset;
      region.bufferRowLength   = 0;
      region.bufferImageHeight = 0;
      region.imageSubresource  = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
      region.imageOffset       = { int32_t(w / 2), int32_t(h / 2), 0 };
      region.imageExtent       = { 1, 1, 1 };
      VkCopyImageToBufferInfo2 copyInfo = { VK_STRUCTURE_TYPE_COPY_IMAGE_TO_BUFFER_INFO_2 };
      copyInfo.srcImage       = aoFinal->handle();
      copyInfo.srcImageLayout = VK_IMAGE_LAYOUT_GENERAL;
      copyInfo.dstBuffer      = readbackBuf->getSliceInfo().buffer;
      copyInfo.regionCount    = 1;
      copyInfo.pRegions       = &region;
      cmd->cmdCopyImageToBuffer(DxvkCmdBuffer::ExecBuffer, &copyInfo);

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

} // namespace dxvk
