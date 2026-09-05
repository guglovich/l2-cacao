#pragma once

#include "../dxvk/dxvk_device.h"
#include "../dxvk/dxvk_context.h"
#include "../util/util_matrix.h"

namespace dxvk {

  // Reuse CacaoQuality from d3d9_cacao.h to avoid duplication
  enum CacaoQuality : uint32_t {
    Lowest = 0,
    Low = 1,
    Medium = 2,
    High = 3,
    Highest = 4,
  };

  // CACAO pass enum
  enum CacaoPass : uint32_t {
    CACAO_PASS_PREPARE_DEPTHS = 0,
    CACAO_PASS_PREPARE_NORMALS = 1,
    CACAO_PASS_GENERATE_Q3 = 2,
    CACAO_PASS_IMPORTANCE_MAP = 3,
    CACAO_PASS_IMPORTANCE_MAP_A = 4,
    CACAO_PASS_IMPORTANCE_MAP_B = 5,
    CACAO_PASS_EDGE_BLUR = 6,
    CACAO_PASS_APPLY = 7,
    CACAO_PASS_COUNT = 8,
  };

  static constexpr uint32_t kTotalSets = CACAO_PASS_COUNT;

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
    float  PerPassFullResUVOffset[4 * 4];
    float  InputOutputBufferDimensions[2];
    float  InputOutputBufferInverseDimensions[2];
    float  ImportanceMapDimensions[2];
    float  ImportanceMapInverseDimensions[2];
    float  DeinterleavedDepthBufferDimensions[2];
    float  DeinterleavedDepthBufferInverseDimensions[2];
    float  DeinterleavedDepthBufferOffset[2];
    float  DeinterleavedDepthBufferNormalisedOffset[2];
  };

  /**
   * \brief Universal CACAO renderer (API-independent)
   * 
   * Works with DxvkDevice and DxvkContext directly.
   * Adapted from D3D9 CACAO for use in D3D11 layer.
   */
  class CacaoRenderer {
  public:
    CacaoRenderer();
    CacaoRenderer(DxvkDevice* device, CacaoQuality quality);
    ~CacaoRenderer();

    bool IsEnabled() const { return m_enabled && m_initialized; }
    bool IsInitialized() const { return m_initialized; }

    void SetEnabled(bool enabled) { m_enabled = enabled; }

    void Apply(
            DxvkContext*   ctx,
            Rc<DxvkImage>  depthImage,
      const float*         projMatrix4x4,
            uint32_t       width,
            uint32_t       height,
            bool           debugFill = false);

    void FillAoBuffer(DxvkContext* ctx, float value);

    Rc<DxvkImageView> GetAoView() const { return m_aoFinalSrvs[m_frameIndex]; }

    void AdvanceFrameIndex() { m_frameIndex = (m_frameIndex + 1u) % kFrameLag; }

  private:
    void InitPipelines();
    void InitResources(uint32_t width, uint32_t height);
    void DestroyResources();
    void UpdateConstants(CacaoConstants* c, const float* proj, uint32_t w, uint32_t h);

    DxvkDevice* m_device;
    CacaoQuality m_quality;
    bool m_enabled = false;
    bool m_initialized = false;

    static constexpr uint32_t kFrameLag = 2;
    static constexpr uint32_t kSetsPerFrame = 8;
    static constexpr uint32_t kReadbackInterval = 60u;

    uint32_t m_frameIndex = 0;
    uint32_t m_width = 0, m_height = 0;

    Rc<DxvkImage> m_deinterleavedDepths;
    Rc<DxvkImage> m_deinterleavedNormals;
    Rc<DxvkImage> m_ssaoPing, m_ssaoPong;
    Rc<DxvkImage> m_importanceMap, m_importanceMapPong;
    Rc<DxvkImage> m_aoFinals[kFrameLag];
    Rc<DxvkImageView> m_aoFinalSrvs[kFrameLag];
    Rc<DxvkImage> m_loadCounter;
    Rc<DxvkImage> m_depthCopy;

    VkDescriptorPool m_descPool = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_descSetLayouts[8] = {};
    VkDescriptorSet m_descSets[2 * 8] = {};

    VkPipeline m_pipelines[9] = {};
    VkPipelineLayout m_pipelineLayouts[8] = {};

    Rc<DxvkBuffer> m_constantBuffers[kFrameLag];

    uint32_t m_readbackFrameCount = 0u;
    Rc<DxvkBuffer> m_readbackBuffer;
    bool m_readbackPending = false;
  };

}
