// Compress to rectangular slices

#include "FoveatedRendering.hlsli"



Texture2D<float4> compositionTexture;

SamplerState trilinearSampler {
	Filter = MIN_MAG_MIP_LINEAR;
	//AddressU = Wrap;
	//AddressV = Wrap;
};

float4 main(float2 uv : TEXCOORD0) : SV_Target {
	bool isRightEye = uv.x > 0.5;
	float2 eyeUV = TextureToEyeUV(uv, isRightEye) / eyeSizeRatio;


	// ==============================================================================
	// DFR 1.1: Per-eye dynamic center (NOT staticFFRCenter + eyeShift)
	// ==============================================================================
	// Key insight: For DFR (Dynamic Foveated Rendering), eyeShift IS the center.
	// staticFFRCenter (baseCenterL/R) is an experience-based static value for FFR mode,
	// which is NOT accurate for eye-tracking based DFR.
	//
	// When eye tracking is active: eyeShift contains the precise gaze-derived center
	// When eye tracking fails: eyeShift = (0,0), center at screen middle (safe default)
	// ==============================================================================

	// Select per-eye shift data
	float2 eyeShift = isRightEye ? eyeShiftR : eyeShiftL;


	// HLSL texture coords: Y down, Eye tracking: Y up -> need Y inversion

	//float2 finalShift = staticFFRCenter;
	float2 finalShift = float2(0., 0.);// Initialize 不缺定可以这样吗，这里应该分别考虑FFR和DFR两种情况
	if (isRightEye) {
		// Right eye: invert X (texture is horizontally flipped), invert Y (coord conversion)
		finalShift.x = -eyeShift.x;
		finalShift.y = -eyeShift.y;
	} else {
		// Left eye: X as-is, invert Y (coord conversion)
		finalShift.x = eyeShift.x;
		finalShift.y = -eyeShift.y;
	}

	float2 c0 = (1. - centerSize) / 2.;
	float2 c1 = (edgeRatio - 1.) * c0 * (finalShift + 1.) / edgeRatio;
	float2 c2 = (edgeRatio - 1.) * centerSize + 1.;

	float2 loBound = c0 * (finalShift + 1.) / c2;
	float2 hiBound = c0 * (finalShift - 1.) / c2 + 1.;
	float2 underBound = float2(eyeUV.x < loBound.x, eyeUV.y < loBound.y);
	float2 inBound = float2(loBound.x < eyeUV.x && eyeUV.x < hiBound.x,
							loBound.y < eyeUV.y && eyeUV.y < hiBound.y);
	float2 overBound = float2(eyeUV.x > hiBound.x, eyeUV.y > hiBound.y);

	float2 center = eyeUV * c2 / edgeRatio + c1;
	float2 d2 = eyeUV * c2;
	float2 d3 = (eyeUV - 1.) * c2 + 1.;
	float2 g1 = eyeUV / loBound;
	float2 g2 = (1. - eyeUV) / (1. - hiBound);

	float2 leftEdge = g1 * center + (1. - g1) * d2;
	float2 rightEdge = g2 * center + (1. - g2) * d3;

	float2 compressedUV = underBound * leftEdge + inBound * center + overBound * rightEdge;

	float4 finalColor = compositionTexture.Sample(trilinearSampler, EyeToTextureUV(compressedUV, isRightEye));

	return finalColor;
}
