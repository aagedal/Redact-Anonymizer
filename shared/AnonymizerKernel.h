/*
** Redact Anonymizer - Metal compute kernels.
**
** The MSL source is embedded as a string and compiled at runtime with
** newLibraryWithSource (same approach as Adobe's SDK_ProcAmp GPU sample).
**
** The noise / sampling functions mirror the CPU implementations in
** AnonymizerAlgo.h - keep the two in sync so software and GPU renders
** are interchangeable.
**
** Frames are 4-channel buffers, float4 (32f) or half4 (16f) per pixel,
** selected by params.is16f at runtime. Channel order only matters for
** desaturation (params.channelOrder, ANON_ORDER_* in AnonymizerAlgo.h).
**
** The host side of struct AnonParams is AnonParamsHost below - every host
** uses that one definition, so the layouts cannot drift apart.
*/

#ifndef ANONYMIZER_KERNEL_H
#define ANONYMIZER_KERNEL_H

static const char* const kAnonymizerMetalString = R"MSLSRC(
#include <metal_stdlib>
using namespace metal;

struct AnonParams
{
	int srcPitch;
	int dstPitch;
	int is16f;
	int width;
	int height;
	int blurRadius;
	int blurDir;
	int mosaicShape;  // 0 square, 1 triangle, 2 hexagon
	float distortAmount;
	float distortScale;
	float blurSigma;
	float mosaicSize;
	uint seed;
	float distortBiasDx;
	float distortBiasDy;
	float desaturate;   // 0..1, applied in the final (mosaic) pass
	int channelOrder;   // 0 RGBA, 1 BGRA, 2 ARGB
	float cellNoise;    // 0..1 per-cell brightness offset (mosaic pass)
	float colorNoise;   // 0..1 per-cell, luma-neutral per-channel offset (mosaic pass)
	uint seed2;         // Smooth Jitter: second pattern, weighted by warpMix2
	float warpMix;      // weight of seed's pattern (1 without Smooth Jitter)
	float warpMix2;
	uint cellSeed;      // still seed for cell/colour noise and frost grain
	float frostAmount;  // 0..1 lift toward white (frost pass)
	float frostGrainSize;
	int blurAlpha;      // blur alpha too (premultiplied fill only)
	float frostColor0;  // frost target colour, buffer channel order
	float frostColor1;
	float frostColor2;
	float frostColor3;
};

inline float4 LoadPix(device const uchar* buf, int pitch, int x, int y, int is16f)
{
	int idx = y * pitch + x;
	if (is16f)
		return float4(((device const half4*)buf)[idx]);
	return ((device const float4*)buf)[idx];
}

inline void StorePix(device uchar* buf, int pitch, int x, int y, int is16f, float4 v)
{
	int idx = y * pitch + x;
	if (is16f)
		((device half4*)buf)[idx] = half4(v);
	else
		((device float4*)buf)[idx] = v;
}

inline float4 SampleBilinear(device const uchar* buf, int pitch, int w, int h, int is16f, float2 p)
{
	float x = clamp(p.x, 0.0f, float(w - 1));
	float y = clamp(p.y, 0.0f, float(h - 1));
	int x0 = int(floor(x));
	int y0 = int(floor(y));
	int x1 = min(x0 + 1, w - 1);
	int y1 = min(y0 + 1, h - 1);
	float tx = x - float(x0);
	float ty = y - float(y0);
	float4 p00 = LoadPix(buf, pitch, x0, y0, is16f);
	float4 p10 = LoadPix(buf, pitch, x1, y0, is16f);
	float4 p01 = LoadPix(buf, pitch, x0, y1, is16f);
	float4 p11 = LoadPix(buf, pitch, x1, y1, is16f);
	float4 top = p00 + (p10 - p00) * tx;
	float4 bot = p01 + (p11 - p01) * tx;
	return top + (bot - top) * ty;
}

// ---- Deterministic noise, mirrored from Anonymizer.h / Anonymizer_CPU.cpp ----

inline uint IHash(uint x)
{
	x ^= x >> 16;
	x *= 0x7feb352dU;
	x ^= x >> 15;
	x *= 0x846ca68bU;
	x ^= x >> 16;
	return x;
}

inline float Rand01(int ix, int iy, uint seed, uint channel)
{
	uint h = IHash(uint(ix) * 0x9E3779B1u
		^ uint(iy) * 0x85EBCA77u
		^ seed * 0xC2B2AE3Du
		^ channel * 0x27D4EB2Fu);
	return float(h) * (1.0f / 4294967295.0f);
}

inline float SmoothT(float t)
{
	return t * t * (3.0f - 2.0f * t);
}

inline float VNoise(float px, float py, uint seed, uint channel)
{
	float fx = floor(px);
	float fy = floor(py);
	int ix = int(fx);
	int iy = int(fy);
	float tx = SmoothT(px - fx);
	float ty = SmoothT(py - fy);
	float a = Rand01(ix,     iy,     seed, channel);
	float b = Rand01(ix + 1, iy,     seed, channel);
	float c = Rand01(ix,     iy + 1, seed, channel);
	float d = Rand01(ix + 1, iy + 1, seed, channel);
	float ab = a + (b - a) * tx;
	float cd = c + (d - c) * tx;
	return ab + (cd - ab) * ty;
}

// Two-octave displacement field, mirrored from AnonAlgo::Displacement.
inline float2 AnonDisplacement(float x, float y, float invScale, uint seed)
{
	float px = x * invScale;
	float py = y * invScale;
	float qx = (px * 0.8f - py * 0.6f) * 2.7f;
	float qy = (px * 0.6f + py * 0.8f) * 2.7f;
	float n0 = VNoise(px, py, seed, 0u) * 2.0f - 1.0f;
	float n1 = VNoise(px, py, seed, 1u) * 2.0f - 1.0f;
	float n2 = VNoise(qx, qy, seed, 2u) * 2.0f - 1.0f;
	float n3 = VNoise(qx, qy, seed, 3u) * 2.0f - 1.0f;
	return float2((n0 + n2 * 0.5f) * 0.6666667f, (n1 + n3 * 0.5f) * 0.6666667f);
}

// Mirrored from AnonAlgo::WarpDisplacement.
inline float2 AnonWarpDisplacement(float x, float y, float invScale, constant AnonParams& p)
{
	float2 n = AnonDisplacement(x, y, invScale, p.seed) * p.warpMix;
	if (p.warpMix2 > 0.0f)
		n += AnonDisplacement(x, y, invScale, p.seed2) * p.warpMix2;
	return n;
}

// ---- Layer 1: random spatial distortion ----

kernel void AnonDistort(
	device const uchar* src [[buffer(0)]],
	device uchar* dst [[buffer(1)]],
	constant AnonParams& p [[buffer(2)]],
	uint2 gid [[thread_position_in_grid]])
{
	if (gid.x >= uint(p.width) || gid.y >= uint(p.height))
		return;
	int x = int(gid.x);
	int y = int(gid.y);
	float4 c;
	if (p.distortAmount <= 0.001f)
	{
		c = LoadPix(src, p.srcPitch, x, y, p.is16f);
	}
	else
	{
		float invScale = 1.0f / max(p.distortScale, 2.0f);
		float2 n = AnonWarpDisplacement(float(x), float(y), invScale, p);
		float dx = (n.x - p.distortBiasDx) * p.distortAmount;
		float dy = (n.y - p.distortBiasDy) * p.distortAmount;
		c = SampleBilinear(src, p.srcPitch, p.width, p.height, p.is16f,
			float2(float(x) + dx, float(y) + dy));
		int alpha = p.channelOrder == 2 ? 0 : 3;
		c[alpha] = LoadPix(src, p.srcPitch, x, y, p.is16f)[alpha];
	}
	StorePix(dst, p.dstPitch, x, y, p.is16f, c);
}

// ---- Layer 2: separable Gaussian blur (blurDir 0 = H, 1 = V) ----

kernel void AnonBlur(
	device const uchar* src [[buffer(0)]],
	device uchar* dst [[buffer(1)]],
	constant AnonParams& p [[buffer(2)]],
	uint2 gid [[thread_position_in_grid]])
{
	if (gid.x >= uint(p.width) || gid.y >= uint(p.height))
		return;
	int x = int(gid.x);
	int y = int(gid.y);
	int r = p.blurRadius;
	if (r < 1)
	{
		StorePix(dst, p.dstPitch, x, y, p.is16f, LoadPix(src, p.srcPitch, x, y, p.is16f));
		return;
	}
	float twoSigmaSq = 2.0f * p.blurSigma * p.blurSigma;
	float4 acc = float4(0.0f);
	float sum = 0.0f;
	for (int i = -r; i <= r; ++i)
	{
		int sx = p.blurDir == 0 ? clamp(x + i, 0, p.width - 1) : x;
		int sy = p.blurDir == 1 ? clamp(y + i, 0, p.height - 1) : y;
		float wgt = exp(-float(i * i) / twoSigmaSq);
		acc += LoadPix(src, p.srcPitch, sx, sy, p.is16f) * wgt;
		sum += wgt;
	}
	float4 c = acc / sum;
	int alpha = p.channelOrder == 2 ? 0 : 3;
	if (!p.blurAlpha)
		c[alpha] = LoadPix(src, p.srcPitch, x, y, p.is16f)[alpha];
	StorePix(dst, p.dstPitch, x, y, p.is16f, c);
}

// ---- Blackout: solid black, source alpha preserved (BGRA -> w is alpha) ----

kernel void AnonBlackout(
	device const uchar* src [[buffer(0)]],
	device uchar* dst [[buffer(1)]],
	constant AnonParams& p [[buffer(2)]],
	uint2 gid [[thread_position_in_grid]])
{
	if (gid.x >= uint(p.width) || gid.y >= uint(p.height))
		return;
	int x = int(gid.x);
	int y = int(gid.y);
	float4 c = LoadPix(src, p.srcPitch, x, y, p.is16f);
	StorePix(dst, p.dstPitch, x, y, p.is16f, float4(0.0f, 0.0f, 0.0f, c.w));
}

// ---- Layer 3: mosaic ----

// Rounds halfway cases up, matching the C++ implementation exactly.
inline float RoundHalfUp(float v)
{
	return floor(v + 0.5f);
}

// Cell-center position for the triangle / hexagon tilings, in pixels.
// Keep in sync with the C++ version in AnonymizerAlgo.h.
inline float2 MosaicCellCenter(float x, float y, float b, int shape)
{
	if (shape == 1) // triangle
	{
		float fx = x / b;
		float fy = y / b;
		float ixf = floor(fx);
		float iyf = floor(fy);
		float u = fx - ixf;
		float v = fy - iyf;
		float cx, cy;
		if ((int(ixf + iyf) & 1) == 0)
		{
			if (u > v) { cx = 2.0f / 3.0f; cy = 1.0f / 3.0f; }
			else       { cx = 1.0f / 3.0f; cy = 2.0f / 3.0f; }
		}
		else
		{
			if (u + v < 1.0f) { cx = 1.0f / 3.0f; cy = 1.0f / 3.0f; }
			else              { cx = 2.0f / 3.0f; cy = 2.0f / 3.0f; }
		}
		return float2((ixf + cx) * b, (iyf + cy) * b);
	}
	// hexagon: pointy-top axial coordinates + cube rounding
	float s = b * 0.5f;
	float qa = 0.57735027f * x;
	float qb = 0.33333333f * y;
	float qn = qa - qb;
	float qf = qn / s;
	float rn = 0.66666667f * y;
	float rf = rn / s;
	float xf = qf;
	float zf = rf;
	float yf = -xf - zf;
	float rx = RoundHalfUp(xf);
	float ry = RoundHalfUp(yf);
	float rz = RoundHalfUp(zf);
	float dx = fabs(rx - xf);
	float dy = fabs(ry - yf);
	float dz = fabs(rz - zf);
	if (dx > dy && dx > dz)
		rx = -ry - rz;
	else if (dy > dz)
		ry = -rx - rz;
	else
		rz = -rx - ry;
	float hx = rz * 0.5f;
	float hy = rx + hx;
	float hz = s * 1.73205081f;
	float vy = s * 1.5f;
	return float2(hz * hy, vy * rz);
}

// Mirrored from AnonAlgo::CellOffset.
inline float AnonCellOffset(int sx, int sy, uint seed, float amount)
{
	return (Rand01(sx, sy, seed, 7u) * 2.0f - 1.0f) * amount;
}

// Mirrored from AnonAlgo::ColorOffset.
inline float AnonColorOffset(int sx, int sy, uint seed, int c, float amount)
{
	return (Rand01(sx, sy, seed, 8u + uint(c)) * 2.0f - 1.0f) * amount;
}

// Mirrored from AnonAlgo::LumaWeights.
inline float4 AnonLumaWeights(int channelOrder)
{
	if (channelOrder == 1)
		return float4(0.0722f, 0.7152f, 0.2126f, 0.0f);
	if (channelOrder == 2)
		return float4(0.0f, 0.2126f, 0.7152f, 0.0722f);
	return float4(0.2126f, 0.7152f, 0.0722f, 0.0f);
}

// Mirrored from AnonAlgo::ChromaOffsets.
inline float4 AnonChromaOffsets(int sx, int sy, uint seed, float amount, float4 w, int alpha)
{
	float4 o;
	float luma = 0.0f;
	for (int i = 0; i < 4; ++i)
	{
		o[i] = i == alpha ? 0.0f : AnonColorOffset(sx, sy, seed, i, amount);
		luma += o[i] * w[i];
	}
	for (int i = 0; i < 4; ++i)
		o[i] = i == alpha ? 0.0f : o[i] - luma;
	return o;
}

// Mirrored from AnonAlgo::DesaturatePass.
inline float4 AnonDesaturate(float4 c, float amount, int channelOrder)
{
	float4 w = AnonLumaWeights(channelOrder);
	int alpha = channelOrder == 2 ? 0 : 3;
	float luma = c.x * w.x + c.y * w.y + c.z * w.z + c.w * w.w;
	float4 r = c + (float4(luma) - c) * amount;
	r[alpha] = c[alpha];
	return r;
}

kernel void AnonMosaic(
	device const uchar* src [[buffer(0)]],
	device uchar* dst [[buffer(1)]],
	constant AnonParams& p [[buffer(2)]],
	uint2 gid [[thread_position_in_grid]])
{
	if (gid.x >= uint(p.width) || gid.y >= uint(p.height))
		return;
	int x = int(gid.x);
	int y = int(gid.y);
	int b = max(1, int(p.mosaicSize + 0.5f));
	int sx, sy;
	if (p.mosaicShape == 0 || b <= 1)
	{
		sx = min((x / b) * b + b / 2, p.width - 1);
		sy = min((y / b) * b + b / 2, p.height - 1);
	}
	else
	{
		float2 c = MosaicCellCenter(float(x), float(y), float(b), p.mosaicShape);
		sx = clamp(int(RoundHalfUp(c.x)), 0, p.width - 1);
		sy = clamp(int(RoundHalfUp(c.y)), 0, p.height - 1);
	}
	float4 c = LoadPix(src, p.srcPitch, sx, sy, p.is16f);
	// Each pixel keeps its own alpha (see AnonAlgo::AlphaIndex).
	int alpha = p.channelOrder == 2 ? 0 : 3;
	c[alpha] = LoadPix(src, p.srcPitch, x, y, p.is16f)[alpha];
	if ((p.cellNoise > 0.0f || p.colorNoise > 0.0f) && b > 1)
	{
		float o = AnonCellOffset(sx, sy, p.cellSeed, p.cellNoise);
		float4 chroma = AnonChromaOffsets(sx, sy, p.cellSeed, p.colorNoise,
			AnonLumaWeights(p.channelOrder), alpha);
		for (int i = 0; i < 4; ++i)
			if (i != alpha)
				c[i] = max(c[i] + o + chroma[i], 0.0f);
	}
	if (p.desaturate > 0.0f)
		c = AnonDesaturate(c, min(p.desaturate, 1.0f), p.channelOrder);
	StorePix(dst, p.dstPitch, x, y, p.is16f, c);
}

// ---- Frosted Glass finish (after the frost blur) ----

// Mirrored from AnonAlgo::FrostGrain (FROST_GRAIN_FLOOR, FROST_GRAIN).
inline float AnonFrostGrain(int x, int y, float grainSize, uint seed, float amount)
{
	int gx = int(floor(float(x) / grainSize));
	int gy = int(floor(float(y) / grainSize));
	return (Rand01(gx, gy, seed, 12u) * 2.0f - 1.0f) * (0.015f + 0.03f * amount);
}

kernel void AnonFrost(
	device const uchar* src [[buffer(0)]],
	device uchar* dst [[buffer(1)]],
	constant AnonParams& p [[buffer(2)]],
	uint2 gid [[thread_position_in_grid]])
{
	if (gid.x >= uint(p.width) || gid.y >= uint(p.height))
		return;
	int x = int(gid.x);
	int y = int(gid.y);
	float4 c = LoadPix(src, p.srcPitch, x, y, p.is16f);
	int alpha = p.channelOrder == 2 ? 0 : 3;
	float a = c[alpha];
	float g = AnonFrostGrain(x, y, p.frostGrainSize, p.cellSeed, p.frostAmount);
	float4 target = float4(p.frostColor0, p.frostColor1, p.frostColor2, p.frostColor3);
	for (int i = 0; i < 4; ++i)
		c[i] = max(c[i] + (target[i] - c[i]) * p.frostAmount + g, 0.0f);
	c[alpha] = a;
	StorePix(dst, p.dstPitch, x, y, p.is16f, c);
}

// ---- Premultiplied input (mirrored from AnonAlgo::UnpremultiplyPass) ----

kernel void AnonUnpremultiply(
	device const uchar* src [[buffer(0)]],
	device uchar* dst [[buffer(1)]],
	constant AnonParams& p [[buffer(2)]],
	device const uchar* fill [[buffer(3)]],
	uint2 gid [[thread_position_in_grid]])
{
	if (gid.x >= uint(p.width) || gid.y >= uint(p.height))
		return;
	int x = int(gid.x);
	int y = int(gid.y);
	float4 c = LoadPix(src, p.srcPitch, x, y, p.is16f);
	float4 f = LoadPix(fill, p.width, x, y, p.is16f);
	int alpha = p.channelOrder == 2 ? 0 : 3;
	float a = c[alpha];
	float4 r;
	if (a > 1.0f / 1024.0f)
		r = c / a;
	else if (f[alpha] > 1e-6f)
		r = f / f[alpha];
	else
		r = float4(0.0f);
	r[alpha] = a;
	StorePix(dst, p.dstPitch, x, y, p.is16f, r);
}

kernel void AnonPremultiply(
	device const uchar* src [[buffer(0)]],
	device uchar* dst [[buffer(1)]],
	constant AnonParams& p [[buffer(2)]],
	uint2 gid [[thread_position_in_grid]])
{
	if (gid.x >= uint(p.width) || gid.y >= uint(p.height))
		return;
	int x = int(gid.x);
	int y = int(gid.y);
	float4 c = LoadPix(src, p.srcPitch, x, y, p.is16f);
	int alpha = p.channelOrder == 2 ? 0 : 3;
	for (int i = 0; i < 4; ++i)
		if (i != alpha)
			c[i] *= c[alpha];
	StorePix(dst, p.dstPitch, x, y, p.is16f, c);
}
)MSLSRC";

/*
** Host-side mirror of struct AnonParams above (all 32-bit fields, same
** order). Shared by every Metal host so the two cannot drift apart.
*/
#include <stdint.h>

typedef struct
{
	int mSrcPitch;
	int mDstPitch;
	int m16f;
	int mWidth;
	int mHeight;
	int mBlurRadius;
	int mBlurDir;
	int mMosaicShape;
	float mDistortAmount;
	float mDistortScale;
	float mBlurSigma;
	float mMosaicSize;
	uint32_t mSeed;
	float mDistortBiasDx;
	float mDistortBiasDy;
	float mDesaturate;
	int mChannelOrder;
	float mCellNoise;
	float mColorNoise;
	uint32_t mSeed2;
	float mWarpMix;
	float mWarpMix2;
	uint32_t mCellSeed;
	float mFrostAmount;
	float mFrostGrainSize;
	int mBlurAlpha;
	float mFrostColor0;
	float mFrostColor1;
	float mFrostColor2;
	float mFrostColor3;
} AnonParamsHost;

/*
** Kernels of the layered pass chain, as AnonEncodeLayeredPasses names them
** (blackout is a separate single pass).
*/
enum
{
	ANON_KERNEL_DISTORT = 0,
	ANON_KERNEL_BLUR,
	ANON_KERNEL_MOSAIC,
	ANON_KERNEL_FROST,
	ANON_KERNEL_UNPREMULTIPLY,
	ANON_KERNEL_PREMULTIPLY,
	ANON_KERNEL_LAYER_COUNT
};

static const char* const kAnonLayerKernelNames[ANON_KERNEL_LAYER_COUNT] =
	{ "AnonDistort", "AnonBlur", "AnonMosaic", "AnonFrost", "AnonUnpremultiply", "AnonPremultiply" };

#ifdef __cplusplus
#include "AnonymizerAlgo.h"

/*
** Encodes the whole layered stack - the same passes as
** AnonAlgo::RunLayeredPasses - shared by every Metal host. The stages
** alternate between tmpA and tmpB; the first stage and those marked
** readsSrc read src (row pitch srcPitch, in pixels) and the last writes dst
** (dstPitch). src and
** dst may be the same buffer: src is only read before the last stage.
** dispatch is called as dispatch(kernel, in, out, aux, params), where aux
** (bound at index 3) is the previous stage's output; only
** AnonUnpremultiply reads it.
*/
template <typename Buffer, typename Dispatch>
inline void AnonEncodeLayeredPasses(Dispatch dispatch, const AnonAlgo::LayerSettings& s,
	int width, int height, int is16f,
	Buffer src, int srcPitch, Buffer dst, int dstPitch, Buffer tmpA, Buffer tmpB)
{
	AnonParamsHost p = {};
	p.m16f = is16f;
	p.mWidth = width;
	p.mHeight = height;
	p.mMosaicShape = s.mosaicShape;
	p.mDistortAmount = s.distortAmount;
	p.mDistortScale = s.distortScale;
	p.mMosaicSize = s.mosaicSize;
	p.mSeed = s.warp.seed;
	p.mSeed2 = s.warp.seed2;
	p.mWarpMix = s.warp.mix;
	p.mWarpMix2 = s.warp.mix2;
	// Zero-mean the displacement field (see AnonAlgo::DisplacementBias).
	AnonAlgo::DisplacementBias(width, height, s.distortScale, s.warp,
		&p.mDistortBiasDx, &p.mDistortBiasDy);
	p.mDesaturate = s.desaturate;
	p.mChannelOrder = s.channelOrder;
	p.mCellNoise = s.cellNoise;
	p.mColorNoise = s.colorNoise;
	p.mCellSeed = s.cellSeed;
	p.mFrostAmount = s.frostAmount;
	p.mFrostGrainSize = s.frostGrainSize;
	p.mFrostColor0 = s.frostColor[0];
	p.mFrostColor1 = s.frostColor[1];
	p.mFrostColor2 = s.frostColor[2];
	p.mFrostColor3 = s.frostColor[3];

	struct Stage { int kernel; int blurDir; float blurRadius; bool blurAlpha; bool readsSrc; };
	Stage stages[12];
	int n = 0;
	auto addBlur = [&](float radius, bool blurAlpha) {
		if ((int)ceilf(radius) < 1)
			return;
		stages[n] = { ANON_KERNEL_BLUR, 0, radius, blurAlpha, false };
		stages[n + 1] = { ANON_KERNEL_BLUR, 1, radius, blurAlpha, false };
		n += 2;
	};
	if (s.premultiplied)
	{
		addBlur(AnonAlgo::FillRadius(s), true);
		stages[n++] = { ANON_KERNEL_UNPREMULTIPLY, 0, 0.0f, false, true };
	}
	stages[n++] = { ANON_KERNEL_DISTORT, 0, 0.0f, false, false };
	if (s.blurAfterMosaic)
	{
		stages[n++] = { ANON_KERNEL_MOSAIC, 0, 0.0f, false, false };
		addBlur(s.blurRadius, false);
	}
	else
	{
		addBlur(s.blurRadius, false);
		stages[n++] = { ANON_KERNEL_MOSAIC, 0, 0.0f, false, false };
	}
	if (s.frostedGlass)
	{
		addBlur(s.frostBlurRadius, false);
		stages[n++] = { ANON_KERNEL_FROST, 0, 0.0f, false, false };
	}
	if (s.premultiplied)
		stages[n++] = { ANON_KERNEL_PREMULTIPLY, 0, 0.0f, false, false };

	Buffer cur = src;
	bool curIsSrc = true;
	for (int i = 0; i < n; ++i)
	{
		const Stage& st = stages[i];
		const bool fromSrc = st.readsSrc || curIsSrc;
		const bool last = i == n - 1;
		Buffer in = fromSrc ? src : cur;
		Buffer out = last ? dst : (!curIsSrc && cur == tmpA ? tmpB : tmpA);
		p.mBlurDir = st.blurDir;
		p.mBlurRadius = std::min((int)ceilf(st.blurRadius), 512);
		p.mBlurSigma = std::max(st.blurRadius * 0.5f, 0.1f);
		p.mBlurAlpha = st.blurAlpha ? 1 : 0;
		p.mSrcPitch = fromSrc ? srcPitch : width;
		p.mDstPitch = last ? dstPitch : width;
		dispatch(st.kernel, in, out, cur, p);
		cur = out;
		curIsSrc = false;
	}
}
#endif // __cplusplus

#endif // ANONYMIZER_KERNEL_H
