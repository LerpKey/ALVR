#pragma once

#include "d3d-render-utils/RenderPipeline.h"

// DFR (Dynamic Foveated Rendering) support - Eye tracking integration
extern "C" {
    struct DFRShiftParams {
        float shift_x, shift_y;
        float left_shift_x, left_shift_y;
        float right_shift_x, right_shift_y;
        bool is_eye_tracked;
    };
    DFRShiftParams get_eye_tracked_ffr_shift();
    // 🎯 NEW: Frame-Perfect timestamp binding function
    DFRShiftParams get_eye_tracked_ffr_shift_with_timestamp(uint64_t target_timestamp_ns);
}

class FFR {
public:
    FFR(ID3D11Device* device);
    void Initialize(ID3D11Texture2D* compositionTexture);
    void Render();
    void GetOptimizedResolution(uint32_t* width, uint32_t* height);
    ID3D11Texture2D* GetOutputTexture();
    
    // DFR support - Update foveation parameters dynamically
    void UpdateFoveationParams();
    // 🎯 NEW: Frame-Perfect timestamp binding
    void UpdateFoveationParams(uint64_t target_timestamp_ns);

private:
    Microsoft::WRL::ComPtr<ID3D11Device> mDevice;
    Microsoft::WRL::ComPtr<ID3D11DeviceContext> mContext;
    Microsoft::WRL::ComPtr<ID3D11Texture2D> mOptimizedTexture;
    Microsoft::WRL::ComPtr<ID3D11VertexShader> mQuadVertexShader;
    Microsoft::WRL::ComPtr<ID3D11Buffer> mFoveatedRenderingBuffer;

    std::vector<d3d_render_utils::RenderPipeline> mPipelines;
};
