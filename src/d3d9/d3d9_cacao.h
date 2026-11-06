#pragma once

#include "../dxvk/dxvk_device.h"
#include "../dxvk/dxvk_buffer.h"
#include "../dxvk/dxvk_image.h"

#include <array>
#include <chrono>

namespace dxvk {

  // Forward-declare D3D9DeviceEx to avoid circular include.
  // d3d9_cacao.cpp includes d3d9_device.h for the full definition.
  class D3D9DeviceEx;

  // CACAO quality levels (match FidelityFX quality settings)
  enum class CacaoQuality : uint32_t {
    Lowest  = 0,  // Q0
    Low     = 1,  // Q1
    Medium  = 2,  // Q2
    High    = 3,  // Q3 (recommended for RDNA2)
    Highest = 4,  // Q3 + adaptive (not yet implemented)
  };

  // CACAO constants UBO - exactly mirrors FfxCacaoConstants from SDK host header
  // Total: 172 dwords (688 bytes), matching FFX_CACAO_CONSTANT_BUFFER_1_SIZE
  struct CacaoConstants {
    float  DepthUnpackConsts[2];
    float  CameraTanHalfFOV[2];
    float  NDCToViewMul[2];
    float  NDCToViewAdd[2];
    float  DepthBufferUVToViewMul[2];
    float  DepthBufferUVToViewAdd[2];
    float  EffectRadius;
    float  EffectShadowStrength;
    float  EffectShadowPow;
    float  EffectShadowClamp;
    float  EffectFadeOutMul;
    float  EffectFadeOutAdd;
    float  EffectHorizonAngleThreshold;
    float  EffectSamplingRadiusNearLimitRec;
    float  DepthPrecisionOffsetMod;
    float  NegRecEffectRadius;
    float  LoadCounterAvgDiv;
    float  AdaptiveSampleCountLimit;
    float  InvSharpness;
    int    BlurNumPasses;
    float  BilateralSigmaSquared;
    float  BilateralSimilarityDistanceSigma;
    // PatternRotScaleMatrices: [4 passes][5 patterns][4 floats per vec4]
    float  PatternRotScaleMatrices[4][5][4];
    float  NormalsUnpackMul;
    float  NormalsUnpackAdd;
    float  DetailAOStrength;
    float  Dummy0;
    float  SSAOBufferDimensions[2];
    float  SSAOBufferInverseDimensions[2];
    float  DepthBufferDimensions[2];
    float  DepthBufferInverseDimensions[2];
    int    DepthBufferOffset[2];
    int    Pad[2];
    // PerPassFullResUVOffset: 4 passes * vec4 = 16 floats
    float  PerPassFullResUVOffset[4 * 4];
    float  InputOutputBufferDimensions[2];
    float  InputOutputBufferInverseDimensions[2];
    float  ImportanceMapDimensions[2];
    float  ImportanceMapInverseDimensions[2];
    float  DeinterleavedDepthBufferDimensions[2];
    float  DeinterleavedDepthBufferInverseDimensions[2];
    float  DeinterleavedDepthBufferOffset[2];
    float  DeinterleavedDepthBufferNormalisedOffset[2];
    float  NormalsWorldToViewspaceMatrix[4][4];
  };

  // Push constants for the mid-frame composite fragment shader.
  // Must match the push_block in dxvk_cacao_mid_frag.frag exactly.
  struct MidCompositeParams {
    int32_t  srcExtent[2];       // back buffer extent (pixels)
    uint32_t samplerScene;       // sampler heap index (linear clamp)
    float    darkMask;           // baked-lighting compensation
    float    contactT0;          // contact mode thresholds
    float    contactT1;
    float    contactStrength;
    float    debugView;          // 1.0 = render raw AO grayscale
  };

  // Pass indices for pipeline array
  enum CacaoPass : uint32_t {
    CACAO_PASS_PREPARE_DEPTHS       = 0,
    CACAO_PASS_PREPARE_NORMALS      = 1,
    CACAO_PASS_GENERATE_Q3          = 2,
    CACAO_PASS_IMPORTANCE_MAP       = 3,
    CACAO_PASS_IMPORTANCE_MAP_A     = 4,
    CACAO_PASS_IMPORTANCE_MAP_B     = 5,
    CACAO_PASS_EDGE_BLUR            = 6,
    CACAO_PASS_APPLY                = 7,
    CACAO_PASS_GENERATE_Q0          = 8,
    CACAO_PASS_GENERATE_Q1          = 9,
    CACAO_PASS_GENERATE_Q2          = 10,
    CACAO_PASS_COUNT                = 11,
  };

  class D3D9CacaoRenderer {
  public:
    D3D9CacaoRenderer(DxvkDevice* device, CacaoQuality quality = CacaoQuality::High);
    ~D3D9CacaoRenderer();

    // Apply CACAO AO to the scene using the current depth buffer.
    // projMatrix: row-major D3D9 projection matrix (from D3DTS_PROJECTION).
    // depthImage: the game's depth stencil VkImage.
    // AO result is written to m_aoFinal (R16G16B16A16_SFLOAT, AO in R channel).
    // Composite step (AO × scene) is a future TODO.
    void Apply(
      D3D9DeviceEx*       device,
      Rc<DxvkImage>       depthImage,
      const float*        projMatrix4x4,  // 16 floats, row-major
      uint32_t            width,
      uint32_t            height,
      bool                debugFill = false); // if true: skip CACAO, fill AO with 0.5 to test composite

    bool IsInitialized() const { return m_initialized; }
    bool IsEnabled()     const { return m_enabled; }
    void SetEnabled(bool enabled) { m_enabled = enabled; }

    // Apply user config values (call after construction, before first Apply()).
    void SetConfig(float radius, float strength, float shadowClamp, CacaoQuality quality) {
      m_radius      = radius;
      m_strength    = strength;
      m_shadowClamp = shadowClamp;
      m_quality     = quality;
    }

    // Returns the AO result view for composite into the present blit.
    // Returns the view for the current frame index (updated by Apply/FillAoBuffer).
    Rc<DxvkImageView> GetAoView() const { return m_aoFinalSrvs[m_frameIndex]; }

    // Fill AO buffer with a constant value (used for UI frames to avoid flickering)
    void FillAoBuffer(D3D9DeviceEx* device, float value);

    // Advance frame index for double-buffering (call before Apply/FillAoBuffer)
    void AdvanceFrameIndex() { m_frameIndex = (m_frameIndex + 1u) % kFrameLag; }

    void SetDumpEnabled(bool enabled) { m_dumpEnabled = enabled; }

    // Mid-frame composite: bake color * AO into the back buffer right at the
    // world->UI boundary, before the game draws its interface. When enabled,
    // the present blitter composite is bypassed entirely (UI stays untouched,
    // letterbox/scaling artifacts are impossible by construction).
    void CompositeMidFrame(
      D3D9DeviceEx*        device,
      const Rc<DxvkImage>& backBuffer,
      float                darkMask,
      float                contactT0,
      float                contactT1,
      float                contactStrength,
      bool                 debugView);

    void SetMidComposite(bool enabled) { m_midComposite = enabled; }
    bool MidCompositeEnabled() const { return m_midComposite; }

  private:
    void InitPipelines();
    void InitResources(uint32_t width, uint32_t height);
    void DestroyResources();

    void UpdateConstants(
      CacaoConstants*  consts,
      const float*     proj,
      uint32_t         width,
      uint32_t         height);

    Rc<DxvkImage> CreateIntermediateImage(
      uint32_t width, uint32_t height,
      VkFormat format,
      uint32_t layers    = 1,
      uint32_t mipLevels = 1,
      VkImageUsageFlags extraUsage = 0);

    DxvkDevice*   m_device;
    CacaoQuality  m_quality;
    bool          m_initialized = false;
    bool          m_enabled     = true;

    // User-configurable effect parameters
    float         m_radius      = 1.2f;
    float         m_strength    = 1.0f;
    float         m_shadowClamp = 0.98f;

    // Diagnostics: world radius scale (m_radius * depthLinearizeMul, set in
    // UpdateConstants) and the sample count of the last chain input depth.
    float         m_worldRadiusScale    = 1.0f;
    uint32_t      m_lastDepthSampleCount = 1u;

    // ── Mid-frame composite state ──────────────────────────────────────────
    bool                 m_midComposite = false;
    bool                 m_midInit      = false;
    VkFormat             m_midFormat    = VK_FORMAT_UNDEFINED;
    VkExtent3D           m_midExtent    = { };
    Rc<DxvkImage>        m_bbImage;                 // tracked for RTV reuse
    Rc<DxvkImageView>    m_bbRtv;
    std::array<Rc<DxvkImage>,     4> m_sceneCopies = { };
    std::array<Rc<DxvkImageView>, 4> m_sceneSrvs   = { };
    const DxvkPipelineLayout* m_midLayout  = nullptr;
    VkPipeline           m_midPipeline = VK_NULL_HANDLE;
    Rc<DxvkSampler>      m_midSampler;

    void InitMidComposite(VkFormat format, VkExtent3D extent);
    void DestroyMidComposite();

    uint32_t      m_width  = 0;
    uint32_t      m_height = 0;

    // ── Raw Vulkan pipeline objects ────────────────────────────────────────
    std::array<VkPipeline,              CACAO_PASS_COUNT> m_pipelines       = {};
    std::array<VkDescriptorSetLayout,   CACAO_PASS_COUNT> m_setLayouts      = {};
    std::array<VkPipelineLayout,        CACAO_PASS_COUNT> m_pipelineLayouts = {};
    std::array<VkSampler, 5>                              m_samplers        = {};

    // ── Descriptor pool and pre-allocated sets ────────────────────────────
    // Double-buffered: 2 frames × kSetsPerFrame sets = kTotalSets sets.
    // Alternating each frame avoids updating sets while GPU executes them.
    // 4 = swapchain image count: descriptor sets and AO buffers are updated
    // on the host at record time; a larger lag guarantees the GPU has finished
    // using slot N-kTotalSets before it is reused, preventing flicker under
    // heavy scenes where the CS queue runs multiple frames behind.
    static constexpr uint32_t kFrameLag    = 4u;
    static constexpr uint32_t kSetsPerFrame = 8u;  // DEPTHS, NORMALS, Q3, BLUR×4, APPLY
    static constexpr uint32_t kTotalSets   = kFrameLag * kSetsPerFrame;

    VkDescriptorPool                                             m_descPool   = VK_NULL_HANDLE;
    std::array<VkDescriptorSet, kTotalSets>                      m_descSets   = {};
    uint32_t                                                     m_frameIndex = 0u;

    // ── Intermediate render targets ────────────────────────────────────────
    Rc<DxvkImage>     m_deinterleavedDepths;   // 4-layer array, half-res, R32F, 4 mips
    Rc<DxvkImage>     m_deinterleavedNormals;  // 4-layer array, half-res, RGBA32F
    Rc<DxvkImage>     m_ssaoPing;             // 4-layer, half-res, RG32F
    Rc<DxvkImage>     m_ssaoPong;             // 4-layer, half-res, RG32F
    Rc<DxvkImage>     m_importanceMap;        // single-layer, quarter-res, R32F
    Rc<DxvkImage>     m_importanceMapPong;    // single-layer, quarter-res, R32F
    Rc<DxvkImage>     m_aoFinals[kFrameLag];  // full-res, R16G16B16A16_SFLOAT (AO in R), one per frame
    Rc<DxvkImageView> m_aoFinalSrvs[kFrameLag]; // SAMPLED_IMAGE views for composite
    Rc<DxvkImage>     m_loadCounter;          // 1×1, R32_UINT (for adaptive load counter)

    // ── Depth copy target ────────────────────────────────────────────────
    // DXVK D3D9 depth surfaces have unreliable layout tracking.
    // We copy depth to our own image with proper layout for compute sampling.
    Rc<DxvkImage>     m_depthCopy;

    // ── Cached image views (static per resolution) ───────────────────────
    // Recreating ~15 VkImageViews every Apply was measurable overhead on
    // the present thread; these only depend on our own intermediates.
    Rc<DxvkImageView> m_vDeintDUav, m_vDeintDSrv, m_vDeintNUav, m_vDeintNSrv;
    std::array<Rc<DxvkImageView>, 4> m_vDeintDMips;
    Rc<DxvkImageView> m_vPingSrv, m_vPingUav, m_vPongSrv, m_vPongUav;
    Rc<DxvkImageView> m_vImpSrv, m_vLoadSrv;
    Rc<DxvkImageView> m_vAoUavs[kFrameLag];

    // ── Constants UBO (double-buffered to avoid race conditions) ───────────
    Rc<DxvkBuffer> m_constantBuffers[kFrameLag];  // One buffer per frame in flight

    // ── CPU readback diagnostics ───────────────────────────────────────────
    // Every kReadbackInterval frames we copy 1 pixel from m_aoFinal into
    // m_readbackBuffer (HOST_VISIBLE) and log the R-channel value.
    // This confirms whether the Apply shader actually writes non-trivial AO.
    static constexpr uint32_t kReadbackInterval = 60u; // log once per ~1 sec
    uint32_t          m_readbackFrameCount = 0u;
    Rc<DxvkBuffer>    m_readbackBuffer;   // 8 bytes: one RGBA16F pixel
    bool              m_readbackPending   = false; // GPU copy issued, not yet read
    // Consecutive per-apply readback probes after each full dump: catches
    // frame-to-frame AO state alternation (flicker) in the log.
    uint32_t          m_probeBurst        = 0;

    // ── Full-buffer dump (cacaodump_A/B rotating pair) ──────────────────
    // Every readback frame with cacaoDump enabled copies the full AO result,
    // deinterleaved depth layer 0 and pre-blur SSAO layer 0 into a staging
    // buffer; the host then writes PGM images into the game CWD. Slot A/B
    // alternate, so the two files always hold the last two dumped frames
    // (works as an instant flicker diff pair and captures whatever scene
    // the user was in when they quit the game).
    bool              m_dumpEnabled       = false;
    uint32_t          m_dumpSlot          = 0;      // 0=A, 1=B
    bool              m_dumpWritePending  = false;
    uint32_t          m_dumpWriteSlot     = 0;
    uint32_t          m_dumpWriteAt       = 0;
    uint32_t          m_applyCount        = 0;
    // Host-side PGM writing is slow and runs on the present thread;
    // issue full-buffer dumps at most once per this interval.
    static constexpr std::chrono::milliseconds kDumpMinInterval { 5000 };
    std::chrono::steady_clock::time_point m_dumpLastIssue {};

    void              WriteDumpFiles(uint32_t slot);
  };

}
