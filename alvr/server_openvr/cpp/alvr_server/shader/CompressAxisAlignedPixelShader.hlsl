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


	// DFR v3: Separate static FFR center and dynamic eye shift
	// Static FFR center for dual-eye convergence (hardcoded for compatibility)
	const float2 staticFFRCenter = float2(0.4, 0.1);

	// centerShift now carries dynamic eyeShift data
	float2 eyeShift = centerShift;

	// Calculate final shift: static FFR + dynamic eye tracking with coordinate system conversion
	// HLSL texture coords: Y down, Eye tracking: Y up -> need Y inversion
	// Left eye: staticFFRCenter + eyeShift (Y inverted)
	// Right eye: staticFFRCenter + eyeShift (X and Y inverted)
	float2 finalShift = staticFFRCenter;
	if (isRightEye) {
		finalShift.x += -eyeShift.x;  // Right eye: invert X shift due to horizontal flip
		finalShift.y += -eyeShift.y;  // Right eye: invert Y shift (HLSL Y-down vs eye-tracking Y-up)
	} else {
		finalShift.x += eyeShift.x;   // Left eye: X shift as-is
		finalShift.y += -eyeShift.y;  // Left eye: invert Y shift (HLSL Y-down vs eye-tracking Y-up)
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
