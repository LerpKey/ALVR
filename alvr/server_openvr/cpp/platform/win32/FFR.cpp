#include "FFR.h"

#include "alvr_server/Settings.h"
#include "alvr_server/Utils.h"
#include "alvr_server/bindings.h"

using Microsoft::WRL::ComPtr;
using namespace d3d_render_utils;

namespace {

struct FoveationVars {
    // Match HLSL cbuffer layout exactly - using packed vectors
    uint32_t targetResolution[2];      // uint2 targetResolution
    uint32_t optimizedResolution[2];   // uint2 optimizedResolution
    float eyeSizeRatio[2];             // float2 eyeSizeRatio
    float centerSize[2];               // float2 centerSize
    float centerShift[2];              // float2 centerShift (contains FFR + DFR combined)
    float edgeRatio[2];                // float2 edgeRatio
    // NOTE: Using centerShift to combine FFR base + DFR eye shift to maintain compatibility
};

FoveationVars CalculateFoveationVars() {
    float targetEyeWidth = (float)Settings::Instance().m_renderWidth / 2;
    float targetEyeHeight = (float)Settings::Instance().m_renderHeight;

    float centerSizeX = (float)Settings::Instance().m_foveationCenterSizeX;
    float centerSizeY = (float)Settings::Instance().m_foveationCenterSizeY;
    float centerShiftX = (float)Settings::Instance().m_foveationCenterShiftX;
    float centerShiftY = (float)Settings::Instance().m_foveationCenterShiftY;
    float edgeRatioX = (float)Settings::Instance().m_foveationEdgeRatioX;
    float edgeRatioY = (float)Settings::Instance().m_foveationEdgeRatioY;

    float edgeSizeX = targetEyeWidth - centerSizeX * targetEyeWidth;
    float edgeSizeY = targetEyeHeight - centerSizeY * targetEyeHeight;

    float centerSizeXAligned
        = 1. - ceil(edgeSizeX / (edgeRatioX * 2.)) * (edgeRatioX * 2.) / targetEyeWidth;
    float centerSizeYAligned
        = 1. - ceil(edgeSizeY / (edgeRatioY * 2.)) * (edgeRatioY * 2.) / targetEyeHeight;

    float edgeSizeXAligned = targetEyeWidth - centerSizeXAligned * targetEyeWidth;
    float edgeSizeYAligned = targetEyeHeight - centerSizeYAligned * targetEyeHeight;

    float centerShiftXAligned = ceil(centerShiftX * edgeSizeXAligned / (edgeRatioX * 2.))
        * (edgeRatioX * 2.) / edgeSizeXAligned;
    float centerShiftYAligned = ceil(centerShiftY * edgeSizeYAligned / (edgeRatioY * 2.))
        * (edgeRatioY * 2.) / edgeSizeYAligned;

    float foveationScaleX = (centerSizeXAligned + (1. - centerSizeXAligned) / edgeRatioX);
    float foveationScaleY = (centerSizeYAligned + (1. - centerSizeYAligned) / edgeRatioY);

    float optimizedEyeWidth = foveationScaleX * targetEyeWidth;
    float optimizedEyeHeight = foveationScaleY * targetEyeHeight;

    // round the frame dimensions to a number of pixel multiple of 32 for the encoder
    auto optimizedEyeWidthAligned = (uint32_t)ceil(optimizedEyeWidth / 32.f) * 32;
    auto optimizedEyeHeightAligned = (uint32_t)ceil(optimizedEyeHeight / 32.f) * 32;

    float eyeWidthRatioAligned = optimizedEyeWidth / optimizedEyeWidthAligned;
    float eyeHeightRatioAligned = optimizedEyeHeight / optimizedEyeHeightAligned;

    // Return struct with correct HLSL cbuffer layout
    FoveationVars vars{};
    vars.targetResolution[0] = (uint32_t)targetEyeWidth;
    vars.targetResolution[1] = (uint32_t)targetEyeHeight;
    vars.optimizedResolution[0] = optimizedEyeWidthAligned;
    vars.optimizedResolution[1] = optimizedEyeHeightAligned;
    vars.eyeSizeRatio[0] = eyeWidthRatioAligned;
    vars.eyeSizeRatio[1] = eyeHeightRatioAligned;
    vars.centerSize[0] = centerSizeXAligned;
    vars.centerSize[1] = centerSizeYAligned;
    // NOTE: centerShift field now carries eyeShift data (dynamic eye tracking)
    // Static FFR center (0.4, 0.1) is hardcoded in HLSL shader
    vars.centerShift[0] = 0.0f;  // Will be set to eyeShift data in UpdateFoveationParams
    vars.centerShift[1] = 0.0f;
    vars.edgeRatio[0] = edgeRatioX;
    vars.edgeRatio[1] = edgeRatioY;
    // NOTE: centerShift field now exclusively carries dynamic eye tracking data

    // Ensure buffer size matches original HLSL expectations (48 bytes = 6 * float2)
    static_assert(sizeof(FoveationVars) == 48, "FoveationVars size must match original HLSL cbuffer layout");

    return vars;
}
}

void FFR::GetOptimizedResolution(uint32_t* width, uint32_t* height) {
    auto fovVars = CalculateFoveationVars();
    *width = fovVars.optimizedResolution[0] * 2;  // optimizedEyeWidth * 2
    *height = fovVars.optimizedResolution[1];     // optimizedEyeHeight
}

FFR::FFR(ID3D11Device* device)
    : mDevice(device) { 
    // Get device context for buffer updates
    mDevice->GetImmediateContext(mContext.GetAddressOf());
}

void FFR::Initialize(ID3D11Texture2D* compositionTexture) {
    auto fovVars = CalculateFoveationVars();
    
    // Create dynamic buffer for DFR updates
    mFoveatedRenderingBuffer = CreateBuffer(mDevice.Get(), fovVars, D3D11_USAGE_DYNAMIC);

    std::vector<uint8_t> quadShaderCSO(
        QUAD_SHADER_CSO_PTR, QUAD_SHADER_CSO_PTR + QUAD_SHADER_CSO_LEN
    );
    mQuadVertexShader = CreateVertexShader(mDevice.Get(), quadShaderCSO);

    mOptimizedTexture = CreateTexture(
        mDevice.Get(),
        fovVars.optimizedResolution[0] * 2,  // optimizedEyeWidth * 2
        fovVars.optimizedResolution[1],      // optimizedEyeHeight
        Settings::Instance().m_enableHdr ? DXGI_FORMAT_R16G16B16A16_FLOAT
                                         : DXGI_FORMAT_R8G8B8A8_UNORM_SRGB
    );

    if (Settings::Instance().m_enableFoveatedEncoding) {
        std::vector<uint8_t> compressAxisAlignedShaderCSO(
            COMPRESS_AXIS_ALIGNED_CSO_PTR,
            COMPRESS_AXIS_ALIGNED_CSO_PTR + COMPRESS_AXIS_ALIGNED_CSO_LEN
        );
        auto compressAxisAlignedPipeline = RenderPipeline(mDevice.Get());
        compressAxisAlignedPipeline.Initialize(
            { compositionTexture },
            mQuadVertexShader.Get(),
            compressAxisAlignedShaderCSO,
            mOptimizedTexture.Get(),
            mFoveatedRenderingBuffer.Get()
        );

        mPipelines.push_back(compressAxisAlignedPipeline);
    } else {
        mOptimizedTexture = compositionTexture;
    }
}

void FFR::Render() {
    for (auto& p : mPipelines) {
        p.Render();
    }
}

// DFR: Update foveation parameters dynamically each frame
void FFR::UpdateFoveationParams() {
    // Legacy function - calls new timestamp-aware version with 0 timestamp
    UpdateFoveationParams(0);
}

// 🎯 NEW: Frame-Perfect timestamp binding implementation
void FFR::UpdateFoveationParams(uint64_t target_timestamp_ns) {
    if (!mFoveatedRenderingBuffer) {
        return; // Not initialized yet
    }

    // Calculate new foveation variables with DFR shift
    auto fovVars = CalculateFoveationVars();

    // 🎯 使用统一时间戳生成器：SteamVR的targetTimestampNs确保Frame-Perfect绑定
    DFRShiftParams dfrShift = {0.0f, 0.0f, false}; // Default fallback
    try {
        if (target_timestamp_ns != 0) {
            // 🎯 关键：使用SteamVR的精确时间戳获取shift数据
            dfrShift = get_eye_tracked_ffr_shift_with_timestamp(target_timestamp_ns);
        } else {
            // Fallback for legacy calls (should not happen in unified timestamp mode)
            Error("🎯 TIMESTAMP_ERROR: target_timestamp_ns is 0, using fallback\n");
            dfrShift = get_eye_tracked_ffr_shift();
        }

        if (dfrShift.is_eye_tracked) {
            // centerShift now carries pure eyeShift data (static FFR center hardcoded in shader)
            fovVars.centerShift[0] = dfrShift.shift_x;
            fovVars.centerShift[1] = dfrShift.shift_y;
        } else {
            // No eye tracking: set eyeShift to zero (use static FFR only)
            fovVars.centerShift[0] = 0.0f;
            fovVars.centerShift[1] = 0.0f;
        }
    } catch (...) {
        // Rust function failed: set eyeShift to zero (use static FFR only)
        Error("🎯 RUST_ERROR: get_eye_tracked_ffr_shift_with_timestamp failed\n");
        fovVars.centerShift[0] = 0.0f;
        fovVars.centerShift[1] = 0.0f;
    }

    // Update the constant buffer with corrected Map/Unmap usage
    D3D11_MAPPED_SUBRESOURCE mappedResource;
    HRESULT hr = mContext->Map(
        mFoveatedRenderingBuffer.Get(),
        0,
        D3D11_MAP_WRITE_DISCARD,
        0,
        &mappedResource
    );

    if (SUCCEEDED(hr)) {
        memcpy(mappedResource.pData, &fovVars, sizeof(FoveationVars));
        mContext->Unmap(mFoveatedRenderingBuffer.Get(), 0);

        // Debug output only on successful update - using already captured dfrShift data
        static int successCount = 0;
        if (++successCount % 600 == 0) { // Log every 10 seconds at 60fps
            if (dfrShift.is_eye_tracked) {
                Info("🎯 DFR Unified: Buffer updated %d times, timestamp=%llu, eyeShift (%.3f, %.3f)\n",
                     successCount, target_timestamp_ns, fovVars.centerShift[0], fovVars.centerShift[1]);
            } else {
                Info("🎯 FFR Unified: Buffer updated %d times, timestamp=%llu, eyeShift (0.0, 0.0)\n",
                     successCount, target_timestamp_ns);
            }
        }
    } else {
        // Log error details for debugging
        Error("🎯 D3D_ERROR: Map failed with HRESULT=0x%08X, buffer=%p, context=%p\n",
              hr, mFoveatedRenderingBuffer.Get(), mContext.Get());
    }
}

ID3D11Texture2D* FFR::GetOutputTexture() { return mOptimizedTexture.Get(); }
