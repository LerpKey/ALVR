#include "FFR.h"

#include "alvr_server/Settings.h"
#include "alvr_server/Utils.h"
#include "alvr_server/bindings.h"

using Microsoft::WRL::ComPtr;
using namespace d3d_render_utils;

namespace {

struct alignas(16) FoveationVars {
    // Match HLSL cbuffer layout exactly - using packed vectors
    uint32_t targetResolution[2];      // uint2 targetResolution
    uint32_t optimizedResolution[2];   // uint2 optimizedResolution
    float eyeSizeRatio[2];             // float2 eyeSizeRatio
    float centerSize[2];               // float2 centerSize
    float baseCenterL[2];              // float2 baseCenterL (static FFR center left, aligned)
    float baseCenterR[2];              // float2 baseCenterR (static FFR center right, aligned)
    float eyeShiftL[2];                // float2 eyeShiftL
    float eyeShiftR[2];                // float2 eyeShiftR
    float edgeRatio[2];                // float2 edgeRatio
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
    vars.baseCenterL[0] = centerShiftXAligned;
    vars.baseCenterL[1] = centerShiftYAligned;
    vars.baseCenterR[0] = centerShiftXAligned;
    vars.baseCenterR[1] = centerShiftYAligned;
    // eyeShift will be filled in UpdateFoveationParams
    vars.eyeShiftL[0] = 0.0f;
    vars.eyeShiftL[1] = 0.0f;
    vars.eyeShiftR[0] = 0.0f;
    vars.eyeShiftR[1] = 0.0f;
    vars.edgeRatio[0] = edgeRatioX;
    vars.edgeRatio[1] = edgeRatioY;

    // Ensure buffer size matches HLSL expectations (80 bytes = 5 * 16-byte rows)
    static_assert(sizeof(FoveationVars) == 80, "FoveationVars size must match HLSL cbuffer layout");

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
    DFRShiftParams dfrShift = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, false}; // Default fallback
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
            fovVars.eyeShiftL[0] = dfrShift.left_shift_x;
            fovVars.eyeShiftL[1] = dfrShift.left_shift_y;
            fovVars.eyeShiftR[0] = dfrShift.right_shift_x;
            fovVars.eyeShiftR[1] = dfrShift.right_shift_y;
        } else {
            fovVars.eyeShiftL[0] = fovVars.baseCenterL[0];
            fovVars.eyeShiftL[1] = fovVars.baseCenterL[1];
            fovVars.eyeShiftR[0] = fovVars.baseCenterR[0];
            fovVars.eyeShiftR[1] = fovVars.baseCenterR[1];
        }
    } catch (...) {
        // Rust function failed: set eyeShift to zero (use static FFR only)
        Error("🎯 RUST_ERROR: get_eye_tracked_ffr_shift_with_timestamp failed\n");
        fovVars.eyeShiftL[0] = fovVars.baseCenterL[0];
        fovVars.eyeShiftL[1] = fovVars.baseCenterL[1];
        fovVars.eyeShiftR[0] = fovVars.baseCenterR[0];
        fovVars.eyeShiftR[1] = fovVars.baseCenterR[1];
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
                     successCount, target_timestamp_ns, fovVars.eyeShiftL[0], fovVars.eyeShiftL[1]);
            } else {
                Info("🎯 FFR Unified: Buffer updated %d times, timestamp=%llu, eyeShift (0.0, 0.0)\n",
                     successCount, target_timestamp_ns);
            }
        }

        // Detailed bounds diagnostics (throttled)
        static int boundsDebugCount = 0;
        if (++boundsDebugCount % 120 == 0) { // ~2s at 60fps
            auto logBounds = [&](const char* eyeLabel, float eyeShiftX, float eyeShiftY) {
                // Match shader math for bounds
                float finalShiftX = 0.4f + eyeShiftX;
                float finalShiftY = 0.1f - eyeShiftY;

                float c0x = (1.f - fovVars.centerSize[0]) * 0.5f;
                float c0y = (1.f - fovVars.centerSize[1]) * 0.5f;
                float c1x = (fovVars.edgeRatio[0] - 1.f) * c0x * (finalShiftX + 1.f) / fovVars.edgeRatio[0];
                float c1y = (fovVars.edgeRatio[1] - 1.f) * c0y * (finalShiftY + 1.f) / fovVars.edgeRatio[1];
                float c2x = (fovVars.edgeRatio[0] - 1.f) * fovVars.centerSize[0] + 1.f;
                float c2y = (fovVars.edgeRatio[1] - 1.f) * fovVars.centerSize[1] + 1.f;
                float loBoundX = c0x * (finalShiftX + 1.f) / c2x;
                float loBoundY = c0y * (finalShiftY + 1.f) / c2y;
                float hiBoundX = c0x * (finalShiftX - 1.f) / c2x + 1.f;
                float hiBoundY = c0y * (finalShiftY - 1.f) / c2y + 1.f;

                Info(
                    "🎯 FFR BOUNDS [%s] centerSize=(%.3f,%.3f) edgeRatio=(%.3f,%.3f) finalShift=(%.3f,%.3f) lo=(%.3f,%.3f) hi=(%.3f,%.3f) eyeSizeRatio=(%.3f,%.3f)",
                    eyeLabel,
                    fovVars.centerSize[0], fovVars.centerSize[1],
                    fovVars.edgeRatio[0], fovVars.edgeRatio[1],
                    finalShiftX, finalShiftY,
                    loBoundX, loBoundY,
                    hiBoundX, hiBoundY,
                    fovVars.eyeSizeRatio[0], fovVars.eyeSizeRatio[1]
                );
            };

            // Left eye (no X flip, Y invert)
            logBounds("L", fovVars.eyeShiftL[0], fovVars.eyeShiftL[1]);
            // Right eye (X and Y invert)
            logBounds("R", -fovVars.eyeShiftR[0], -fovVars.eyeShiftR[1]);
        }
    } else {
        // Log error details for debugging
        Error("🎯 D3D_ERROR: Map failed with HRESULT=0x%08X, buffer=%p, context=%p\n",
              hr, mFoveatedRenderingBuffer.Get(), mContext.Get());
    }
}

ID3D11Texture2D* FFR::GetOutputTexture() { return mOptimizedTexture.Get(); }
