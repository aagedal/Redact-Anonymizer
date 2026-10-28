/*
** Redact Anonymizer - host-independent algorithm core.
**
** Shared by the Premiere Pro plugin (src/) and the OpenFX plugin for
** DaVinci Resolve (ofx/). Everything here operates on tightly packed
** 4-channel float buffers (row pitch == width) and is channel-order
** agnostic except where noted.
**
** The same noise / sampling code exists in MSL in AnonymizerKernel.h -
** keep the two in sync so CPU and GPU renders are interchangeable.
*/

#ifndef ANONYMIZER_ALGO_H
#define ANONYMIZER_ALGO_H

#include <math.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <vector>
#include <algorithm>

/*
** Parameter ranges and defaults, shared by both host UIs. The three layer
** strengths have floors 25-30 % below their defaults, so a slider cannot
** quietly weaken a layer past that; the Distortion / Blur / Mosaic
** checkboxes switch a layer off entirely instead (for previewing).
*/
#define DISTORT_AMOUNT_MIN   15.0
#define DISTORT_AMOUNT_MAX   200.0
#define DISTORT_AMOUNT_DFLT  20.0

#define DISTORT_SCALE_MIN    4.0
#define DISTORT_SCALE_MAX    400.0
#define DISTORT_SCALE_DFLT   11.0

#define BLUR_RADIUS_MIN      15.0
#define BLUR_RADIUS_MAX      100.0
#define BLUR_RADIUS_DFLT     20.0
#define BLUR_AFTER_MOSAIC_RADIUS_DFLT 40.0
/* Earlier blur defaults, still paired up for older projects. */
#define BLUR_RADIUS_LEGACY_DFLT             15.0
#define BLUR_AFTER_MOSAIC_RADIUS_LEGACY_DFLT 35.0

#define MOSAIC_SIZE_MIN      21.0
#define MOSAIC_SIZE_MAX      256.0
#define MOSAIC_SIZE_DFLT     30.0

/* 2^24 - 1: the largest range a float slider (AE valid_max is a float)
** can step through exactly, and well inside every host's int range. */
#define SEED_MIN             0.0
#define SEED_MAX             16777215.0
#define SEED_DFLT            0.0

#define DESATURATE_MIN       0.0
#define DESATURATE_MAX       100.0
#define DESATURATE_DFLT      0.0

/* Per-cell brightness offset, in percent of full range (max +/-). Small
** values already break exact block-colour matching (Depix-style attacks). */
#define CELL_NOISE_MIN       0.0
#define CELL_NOISE_MAX       25.0
#define CELL_NOISE_DFLT      3.0

/* Per-channel offset per mosaic cell, in percent of full range (max +/-,
** before its luma is removed): shifts each cell's hue, not its brightness.
** With Blur After Mosaic it is multiplied by COLOR_NOISE_BLURRED_GAIN,
** since the blur averages roughly 60 % of it away. Not under Frosted Glass:
** its smooth surface shows even faint colour blotches. */
#define COLOR_NOISE_MIN      0.0
#define COLOR_NOISE_MAX      25.0
#define COLOR_NOISE_DFLT     1.0
#define COLOR_NOISE_BLURRED_GAIN 2.5f

/* Frosted Glass finish: how far the result is lifted toward the Frost
** Colour (white by default), in percent. Runs after the anonymizing layers, so it only changes the look. */
#define FROST_AMOUNT_MIN     0.0
#define FROST_AMOUNT_MAX     100.0
#define FROST_AMOUNT_DFLT    40.0
/* Default Frost Colour: 75 % grey for every channel; full white tends to
** come out too bright. */
#define FROST_COLOR_DFLT     0.75

/* Smooth Jitter (on by default): frames between two independent
** distortion patterns. */
#define SMOOTH_JITTER_FRAMES 12

/*
** Pixel-space parameters are specified at a 1080p reference and scaled by
** the shorter frame dimension, so the anonymization strength is independent
** of both timeline resolution and orientation. Proxy/preview downsampling is
** handled automatically: a half-resolution frame has half the short dimension.
*/
#define ANON_REFERENCE_HEIGHT 1080.0f

/*
** Mosaic cell shapes (popup order in every host UI).
*/
enum
{
	ANON_SHAPE_SQUARE = 0,
	ANON_SHAPE_TRIANGLE = 1,
	ANON_SHAPE_HEXAGON = 2,
};

/*
** Channel layout of the 4-channel buffers, needed only by desaturation
** (every other pass treats the four channels identically).
*/
enum
{
	ANON_ORDER_RGBA = 0, // OFX, FxPlug, Photoshop
	ANON_ORDER_BGRA = 1, // Premiere Pro float / GPU frames
	ANON_ORDER_ARGB = 2, // After Effects 8-bit worlds
};

static inline float AnonResolutionScale(int inFrameWidth, int inFrameHeight)
{
	int shortSide = inFrameWidth < inFrameHeight ? inFrameWidth : inFrameHeight;
	if (shortSide <= 0)
		return 1.0f;
	return (float)shortSide / ANON_REFERENCE_HEIGHT;
}

/*
** Deterministic hash / seed helpers.
*/
static inline uint32_t AnonIHash(uint32_t x)
{
	x ^= x >> 16;
	x *= 0x7feb352dU;
	x ^= x >> 15;
	x *= 0x846ca68bU;
	x ^= x >> 16;
	return x;
}

static inline uint32_t AnonComputeSeed(double inSeedParam, bool inJitter, int32_t inFrame,
	uint32_t inSessionKey = 0u)
{
	uint32_t s = (uint32_t)(inSeedParam + 0.5);
	uint32_t f = inJitter ? (uint32_t)inFrame : 0u;
	uint32_t h = AnonIHash(s * 0x9E3779B1u ^ f * 0x85EBCA77u ^ 0x5BD1E995u);
	return inSessionKey ? AnonIHash(h ^ inSessionKey) : h;
}

namespace AnonAlgo
{

/*
** The distortion pattern for one frame: seed's field weighted by mix, plus
** seed2's weighted by mix2 when Smooth Jitter blends two patterns.
*/
struct Warp
{
	uint32_t seed;
	uint32_t seed2;
	float mix;
	float mix2;
};

inline Warp StaticWarp(uint32_t seed)
{
	return { seed, 0u, 1.0f, 0.0f };
}

/*
** Temporal Jitter draws an unrelated pattern every frame. Smooth Jitter
** instead draws one every SMOOTH_JITTER_FRAMES frames and blends between
** them with cos/sin weights, whose squares sum to 1, so the warp strength
** stays constant while the pattern flows from one to the next.
*/
inline Warp ComputeWarp(double seedParam, bool jitter, bool smooth, int32_t frame,
	uint32_t sessionKey)
{
	if (!jitter || !smooth)
		return StaticWarp(AnonComputeSeed(seedParam, jitter, frame, sessionKey));
	const int32_t n = SMOOTH_JITTER_FRAMES;
	const int32_t k = frame >= 0 ? frame / n : -((-frame + n - 1) / n);
	const float t = (float)(frame - k * n) / (float)n;
	if (t == 0.0f)
		return StaticWarp(AnonComputeSeed(seedParam, true, k, sessionKey));
	const float a = t * 1.5707963f;
	return { AnonComputeSeed(seedParam, true, k, sessionKey),
		AnonComputeSeed(seedParam, true, k + 1, sessionKey), cosf(a), sinf(a) };
}

/*
** Seed for the per-cell colour offsets and the frost grain. It never
** changes with the frame, so those stay still instead of flickering, while
** the session key (or Photoshop's per-run seed) still makes it new on every
** launch.
*/
inline uint32_t CellSeedFor(uint32_t staticSeed)
{
	return AnonIHash(staticSeed ^ 0xA5C3E1F7u);
}

inline uint32_t ComputeCellSeed(double seedParam, uint32_t sessionKey)
{
	return CellSeedFor(AnonComputeSeed(seedParam, false, 0, sessionKey));
}

// Effective layer strength in pixels: raises values saved below the slider
// minimum (older projects) and returns `off` when the layer is switched off.
inline float LayerStrength(double value, double minValue, bool enabled, float ds, float off)
{
	return enabled ? (float)std::max(value, minValue) * ds : off;
}

// Switch between the two blur defaults on user checkbox changes.
// Preserve other values; loading a project and rendering do not rewrite settings.
// Projects saved with the older 15/35 defaults keep switching between those.
inline double BlurRadiusForOrderChange(double radius, bool blurAfterMosaic)
{
	if (blurAfterMosaic && radius == BLUR_RADIUS_DFLT)
		return BLUR_AFTER_MOSAIC_RADIUS_DFLT;
	if (!blurAfterMosaic && radius == BLUR_AFTER_MOSAIC_RADIUS_DFLT)
		return BLUR_RADIUS_DFLT;
	if (blurAfterMosaic && radius == BLUR_RADIUS_LEGACY_DFLT)
		return BLUR_AFTER_MOSAIC_RADIUS_LEGACY_DFLT;
	if (!blurAfterMosaic && radius == BLUR_AFTER_MOSAIC_RADIUS_LEGACY_DFLT)
		return BLUR_RADIUS_LEGACY_DFLT;
	return radius;
}

/*
** Process-lifetime random key for the "Session Key" option. Drawn from the
** system CSPRNG on first use and never stored anywhere - not in the project,
** not on disk - so the exact distortion pattern cannot be reproduced even by
** someone holding the project file and seed. Non-static inline: one key per
** process, shared by the CPU and GPU translation units of a plugin.
*/
inline uint32_t SessionKey()
{
	static const uint32_t sKey = [] {
		uint32_t k = 0;
		while (k == 0) // 0 means "no key" to AnonComputeSeed
			k = arc4random();
		return k;
	}();
	return sKey;
}

/* Random value for the "Randomize Seed" buttons, within the seed range. */
inline int RandomSeedValue()
{
	return (int)arc4random_uniform((uint32_t)SEED_MAX) + 1;
}

inline float Rand01(int32_t ix, int32_t iy, uint32_t seed, uint32_t channel)
{
	uint32_t h = AnonIHash((uint32_t)ix * 0x9E3779B1u
		^ (uint32_t)iy * 0x85EBCA77u
		^ seed * 0xC2B2AE3Du
		^ channel * 0x27D4EB2Fu);
	return (float)h * (1.0f / 4294967295.0f);
}

inline float SmoothT(float t)
{
	return t * t * (3.0f - 2.0f * t);
}

// Smoothly interpolated per-cell value noise in [0,1]
inline float VNoise(float px, float py, uint32_t seed, uint32_t channel)
{
	float fx = floorf(px);
	float fy = floorf(py);
	int32_t ix = (int32_t)fx;
	int32_t iy = (int32_t)fy;
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

inline const float* PixAt(const float* buf, int w, int x, int y)
{
	return buf + ((size_t)y * w + x) * 4;
}

inline void SampleBilinear(const float* buf, int w, int h, float x, float y, float* outPix)
{
	x = std::min(std::max(x, 0.0f), (float)(w - 1));
	y = std::min(std::max(y, 0.0f), (float)(h - 1));
	int x0 = (int)floorf(x);
	int y0 = (int)floorf(y);
	int x1 = std::min(x0 + 1, w - 1);
	int y1 = std::min(y0 + 1, h - 1);
	float tx = x - (float)x0;
	float ty = y - (float)y0;
	const float* p00 = PixAt(buf, w, x0, y0);
	const float* p10 = PixAt(buf, w, x1, y0);
	const float* p01 = PixAt(buf, w, x0, y1);
	const float* p11 = PixAt(buf, w, x1, y1);
	for (int c = 0; c < 4; ++c)
	{
		float top = p00[c] + (p10[c] - p00[c]) * tx;
		float bot = p01[c] + (p11[c] - p01[c]) * tx;
		outPix[c] = top + (bot - top) * ty;
	}
}

/*
** Raw displacement field in [-1,1] per axis, before mean removal. Two
** octaves of value noise: the base octave at the user's scale, plus a
** finer octave (2.7x frequency, half weight) on a lattice rotated by
** atan(3/4) - an exact rotation (0.8 / 0.6) so CPU and GPU agree bit for
** bit. The rotated detail octave breaks up the axis-aligned, locally
** near-affine look of single-octave value noise, so small neighbourhoods
** are no longer just translated copies that a local-shift model could
** undo. Keep in sync with AnonDisplacement in AnonymizerKernel.h.
*/
inline void Displacement(float x, float y, float invScale, uint32_t seed,
	float* outDx, float* outDy)
{
	float px = x * invScale;
	float py = y * invScale;
	float qx = (px * 0.8f - py * 0.6f) * 2.7f;
	float qy = (px * 0.6f + py * 0.8f) * 2.7f;
	float n0 = VNoise(px, py, seed, 0u) * 2.0f - 1.0f;
	float n1 = VNoise(px, py, seed, 1u) * 2.0f - 1.0f;
	float n2 = VNoise(qx, qy, seed, 2u) * 2.0f - 1.0f;
	float n3 = VNoise(qx, qy, seed, 3u) * 2.0f - 1.0f;
	*outDx = (n0 + n2 * 0.5f) * 0.6666667f;
	*outDy = (n1 + n3 * 0.5f) * 0.6666667f;
}

// Displacement of a (possibly blended) warp. With mix == 1 and mix2 == 0
// this is exactly Displacement. Keep in sync with AnonWarpDisplacement.
inline void WarpDisplacement(float x, float y, float invScale, const Warp& warp,
	float* outDx, float* outDy)
{
	float dx, dy;
	Displacement(x, y, invScale, warp.seed, &dx, &dy);
	dx *= warp.mix;
	dy *= warp.mix;
	if (warp.mix2 > 0.0f)
	{
		float dx2, dy2;
		Displacement(x, y, invScale, warp.seed2, &dx2, &dy2);
		dx += dx2 * warp.mix2;
		dy += dy2 * warp.mix2;
	}
	*outDx = dx;
	*outDy = dy;
}

/*
** Value noise has a non-zero spatial mean over any finite domain, causing
** a net DC translation of the whole image. The mean changes with each seed,
** so temporal jitter (different seed per frame) makes the whole image bounce.
** Fix: estimate the mean over a sparse uniform grid and subtract it so the
** net displacement is zero regardless of seed or frame size. Every host
** (CPU and GPU) uses this one function so the bias is identical.
*/
inline void DisplacementBias(int w, int h, float scale, const Warp& warp,
	float* outBiasDx, float* outBiasDy)
{
	float invScale = 1.0f / std::max(scale, 2.0f);
	const int kStep = std::max(1, std::max(w, h) / 64);
	float sumDx = 0.0f, sumDy = 0.0f;
	int count = 0;
	for (int sy = kStep / 2; sy < h; sy += kStep) {
		for (int sx = kStep / 2; sx < w; sx += kStep) {
			float dx, dy;
			WarpDisplacement((float)sx, (float)sy, invScale, warp, &dx, &dy);
			sumDx += dx;
			sumDy += dy;
			++count;
		}
	}
	*outBiasDx = count > 0 ? sumDx / (float)count : 0.0f;
	*outBiasDy = count > 0 ? sumDy / (float)count : 0.0f;
}

/*
** Alpha is never anonymized: every pass keeps each pixel's own alpha, so a
** matte the host passes in alpha (e.g. a Resolve power window) keeps its
** exact shape instead of being warped, pixelated and blurred.
*/
inline int AlphaIndex(int channelOrder)
{
	return channelOrder == ANON_ORDER_ARGB ? 0 : 3;
}

inline void DistortPass(const float* src, float* dst, int w, int h,
	float amount, float scale, const Warp& warp, int channelOrder = ANON_ORDER_RGBA)
{
	if (amount <= 0.001f)
	{
		memcpy(dst, src, (size_t)w * h * 4 * sizeof(float));
		return;
	}
	float invScale = 1.0f / std::max(scale, 2.0f);
	float biasDx, biasDy;
	DisplacementBias(w, h, scale, warp, &biasDx, &biasDy);
	const int alpha = AlphaIndex(channelOrder);
	for (int y = 0; y < h; ++y)
	{
		float* out = dst + (size_t)y * w * 4;
		for (int x = 0; x < w; ++x, out += 4)
		{
			float nx, ny;
			WarpDisplacement((float)x, (float)y, invScale, warp, &nx, &ny);
			float dx = (nx - biasDx) * amount;
			float dy = (ny - biasDy) * amount;
			SampleBilinear(src, w, h, (float)x + dx, (float)y + dy, out);
			out[alpha] = PixAt(src, w, x, y)[alpha];
		}
	}
}

inline void DistortPass(const float* src, float* dst, int w, int h,
	float amount, float scale, uint32_t seed)
{
	DistortPass(src, dst, w, h, amount, scale, StaticWarp(seed));
}

// dir 0 = horizontal, 1 = vertical
// blurAlpha blurs alpha too (only for the premultiplied fill); otherwise
// each pixel keeps its own alpha like every other pass.
inline void BlurPass(const float* src, float* dst, int w, int h, float radius, int dir,
	int channelOrder = ANON_ORDER_RGBA, bool blurAlpha = false)
{
	int r = (int)ceilf(radius);
	if (r < 1)
	{
		memcpy(dst, src, (size_t)w * h * 4 * sizeof(float));
		return;
	}
	float sigma = std::max(radius * 0.5f, 0.1f);
	std::vector<float> weights(2 * r + 1);
	for (int i = -r; i <= r; ++i)
		weights[i + r] = expf(-(float)(i * i) / (2.0f * sigma * sigma));

	const int alpha = AlphaIndex(channelOrder);
	for (int y = 0; y < h; ++y)
	{
		float* out = dst + (size_t)y * w * 4;
		for (int x = 0; x < w; ++x, out += 4)
		{
			float acc[4] = {0, 0, 0, 0};
			float sum = 0.0f;
			for (int i = -r; i <= r; ++i)
			{
				int sx = dir == 0 ? std::min(std::max(x + i, 0), w - 1) : x;
				int sy = dir == 1 ? std::min(std::max(y + i, 0), h - 1) : y;
				const float* p = PixAt(src, w, sx, sy);
				float wgt = weights[i + r];
				acc[0] += p[0] * wgt;
				acc[1] += p[1] * wgt;
				acc[2] += p[2] * wgt;
				acc[3] += p[3] * wgt;
				sum += wgt;
			}
			float inv = 1.0f / sum;
			out[0] = acc[0] * inv;
			out[1] = acc[1] * inv;
			out[2] = acc[2] * inv;
			out[3] = acc[3] * inv;
			if (!blurAlpha)
				out[alpha] = PixAt(src, w, x, y)[alpha];
		}
	}
}

// Rounds halfway cases up, matching the MSL implementation exactly.
inline float RoundHalfUp(float v)
{
	return floorf(v + 0.5f);
}

// Cell-center position for the triangle / hexagon tilings, in pixels.
// Keep in sync with the MSL version in AnonymizerKernel.h.
inline void MosaicCellCenter(float x, float y, float b, int shape, float* outX, float* outY)
{
	if (shape == ANON_SHAPE_TRIANGLE)
	{
		// Each b-sized square splits into two right triangles; the diagonal
		// direction alternates in a checkerboard for a woven look.
		float fx = x / b;
		float fy = y / b;
		float ixf = floorf(fx);
		float iyf = floorf(fy);
		float u = fx - ixf;
		float v = fy - iyf;
		float cx, cy;
		if (((int)(ixf + iyf) & 1) == 0)
		{
			if (u > v) { cx = 2.0f / 3.0f; cy = 1.0f / 3.0f; }
			else       { cx = 1.0f / 3.0f; cy = 2.0f / 3.0f; }
		}
		else
		{
			if (u + v < 1.0f) { cx = 1.0f / 3.0f; cy = 1.0f / 3.0f; }
			else              { cx = 2.0f / 3.0f; cy = 2.0f / 3.0f; }
		}
		*outX = (ixf + cx) * b;
		*outY = (iyf + cy) * b;
	}
	else // ANON_SHAPE_HEXAGON
	{
		// Pointy-top hexagonal grid via axial coordinates + cube rounding,
		// sized so a hexagon is roughly b pixels tall.
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
		float dx = fabsf(rx - xf);
		float dy = fabsf(ry - yf);
		float dz = fabsf(rz - zf);
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
		*outX = hz * hy;
		*outY = vy * rz;
	}
}

// Seeded brightness offset for the mosaic cell sampled at (sx, sy), in
// [-amount, amount]. Keep in sync with AnonCellOffset in the MSL.
inline float CellOffset(int sx, int sy, uint32_t seed, float amount)
{
	return (Rand01(sx, sy, seed, 7u) * 2.0f - 1.0f) * amount;
}

// Rec. 709 luma weights at the R, G, B positions of a channel order.
inline void LumaWeights(int channelOrder, float outW[4])
{
	const float r = 0.2126f, g = 0.7152f, b = 0.0722f;
	if (channelOrder == ANON_ORDER_BGRA)
		{ outW[0] = b; outW[1] = g; outW[2] = r; outW[3] = 0.0f; }
	else if (channelOrder == ANON_ORDER_ARGB)
		{ outW[0] = 0.0f; outW[1] = r; outW[2] = g; outW[3] = b; }
	else
		{ outW[0] = r; outW[1] = g; outW[2] = b; outW[3] = 0.0f; }
}

// Seeded offset for buffer channel c of the cell sampled at (sx, sy), in
// [-amount, amount]. Keep in sync with AnonColorOffset in the MSL.
inline float ColorOffset(int sx, int sy, uint32_t seed, int c, float amount)
{
	return (Rand01(sx, sy, seed, 8u + (uint32_t)c) * 2.0f - 1.0f) * amount;
}

// Per-channel colour offsets of a cell with their luma removed, so colour
// noise shifts hue only (Cell Noise handles brightness). Alpha gets 0.
// Keep in sync with AnonChromaOffsets in the MSL.
inline void ChromaOffsets(int sx, int sy, uint32_t seed, float amount, const float lw[4],
	int alpha, float out[4])
{
	float luma = 0.0f;
	for (int c = 0; c < 4; ++c)
	{
		out[c] = c == alpha ? 0.0f : ColorOffset(sx, sy, seed, c, amount);
		luma += out[c] * lw[c];
	}
	for (int c = 0; c < 4; ++c)
		out[c] = c == alpha ? 0.0f : out[c] - luma;
}

// cellNoise (0..1 of full range) shifts each cell's colour channels by one
// seeded offset, and colorNoise adds a luma-neutral offset per channel; both
// are floored at 0, so cell values no longer equal the averaged source
// colours. Each pixel keeps its own alpha (not the cell's).
inline void MosaicPass(const float* src, float* dst, int w, int h, float blockSize, int shape,
	uint32_t seed = 0u, float cellNoise = 0.0f, int channelOrder = ANON_ORDER_RGBA,
	float colorNoise = 0.0f)
{
	int b = std::max(1, (int)(blockSize + 0.5f));
	if (b <= 1)
	{
		memcpy(dst, src, (size_t)w * h * 4 * sizeof(float));
		return;
	}
	const int alpha = AlphaIndex(channelOrder);
	float lw[4];
	LumaWeights(channelOrder, lw);
	for (int y = 0; y < h; ++y)
	{
		float* out = dst + (size_t)y * w * 4;
		int sqy = std::min((y / b) * b + b / 2, h - 1);
		for (int x = 0; x < w; ++x, out += 4)
		{
			int sx, sy;
			if (shape == ANON_SHAPE_SQUARE)
			{
				sx = std::min((x / b) * b + b / 2, w - 1);
				sy = sqy;
			}
			else
			{
				float cx, cy;
				MosaicCellCenter((float)x, (float)y, (float)b, shape, &cx, &cy);
				sx = std::min(std::max((int)RoundHalfUp(cx), 0), w - 1);
				sy = std::min(std::max((int)RoundHalfUp(cy), 0), h - 1);
			}
			const float* p = PixAt(src, w, sx, sy);
			out[0] = p[0];
			out[1] = p[1];
			out[2] = p[2];
			out[3] = p[3];
			out[alpha] = PixAt(src, w, x, y)[alpha];
			if (cellNoise > 0.0f || colorNoise > 0.0f)
			{
				const float o = CellOffset(sx, sy, seed, cellNoise);
				float chroma[4];
				ChromaOffsets(sx, sy, seed, colorNoise, lw, alpha, chroma);
				for (int c = 0; c < 4; ++c)
					if (c != alpha)
						out[c] = std::max(out[c] + o + chroma[c], 0.0f);
			}
		}
	}
}

// Blends the colour channels toward luma by `amount` (0..1), alpha untouched.
// Every other pass is linear per channel, so desaturating last is identical
// to desaturating first. Keep in sync with AnonDesaturate in the MSL.
inline void DesaturatePass(float* buf, int w, int h, float amount, int channelOrder)
{
	if (amount <= 0.0f)
		return;
	amount = std::min(amount, 1.0f);
	float lw[4];
	LumaWeights(channelOrder, lw);
	const int alpha = channelOrder == ANON_ORDER_ARGB ? 0 : 3;
	for (size_t i = 0; i < (size_t)w * h; ++i)
	{
		float* p = buf + i * 4;
		float luma = p[0] * lw[0] + p[1] * lw[1] + p[2] * lw[2] + p[3] * lw[3];
		for (int c = 0; c < 4; ++c)
			if (c != alpha)
				p[c] = p[c] + (luma - p[c]) * amount;
	}
}

/*
** Frosted Glass: after the anonymizing layers, an extra blur melts the
** mosaic cells, the colour channels are lifted toward `color` (in buffer
** channel order) by `amount` (0..1) and a still grain of grainSize-pixel cells is added. It only sees
** the anonymized result, so it cannot reveal anything the layers removed.
** FROST_GRAIN is the grain's +/- range at full amount, plus a floor so the
** glass keeps some texture at low amounts.
*/
#define FROST_GRAIN       0.03f
#define FROST_GRAIN_FLOOR 0.015f
/* Frost blur radius in mosaic cells. Wider than one cell, so it removes the
** cell-scale structure too: a blur that only melts the cell edges brings a
** blocked face back into view (the Harmon effect) and looks reversible. */
#define FROST_BLUR_CELLS  1.75f

// Frost grain value for pixel (x, y), in [-1, 1] times the grain strength.
// Keep in sync with AnonFrostGrain in the MSL.
inline float FrostGrain(int x, int y, float grainSize, uint32_t seed, float amount)
{
	const int gx = (int)floorf((float)x / grainSize);
	const int gy = (int)floorf((float)y / grainSize);
	return (Rand01(gx, gy, seed, 12u) * 2.0f - 1.0f) * (FROST_GRAIN_FLOOR + FROST_GRAIN * amount);
}

inline void FrostPass(float* buf, int w, int h, float amount, const float color[4],
	float grainSize, uint32_t seed, int channelOrder)
{
	const int alpha = AlphaIndex(channelOrder);
	for (int y = 0; y < h; ++y)
	{
		float* p = buf + (size_t)y * w * 4;
		for (int x = 0; x < w; ++x, p += 4)
		{
			const float g = FrostGrain(x, y, grainSize, seed, amount);
			for (int c = 0; c < 4; ++c)
				if (c != alpha)
					p[c] = std::max(p[c] + (color[c] - p[c]) * amount + g, 0.0f);
		}
	}
}

/*
** Everything one render needs, in pixels of the actual frame and 0..1
** fractions. Hosts build it with MakeLayerSettings, so the scaling and
** floors are applied identically everywhere.
*/
struct LayerSettings
{
	float distortAmount = 0.0f;  // 0 disables the layer
	float distortScale = 2.0f;
	float blurRadius = 0.0f;     // 0 disables the layer
	float mosaicSize = 1.0f;     // 1 disables the layer (and the cell/colour noise)
	int mosaicShape = ANON_SHAPE_SQUARE;
	bool blurAfterMosaic = false;
	float desaturate = 0.0f;
	float cellNoise = 0.0f;
	float colorNoise = 0.0f;
	bool frostedGlass = false;
	float frostAmount = 0.0f;
	float frostColor[4] = { (float)FROST_COLOR_DFLT, (float)FROST_COLOR_DFLT,
		(float)FROST_COLOR_DFLT, (float)FROST_COLOR_DFLT }; // buffer channel order
	float frostBlurRadius = 0.0f;
	float frostGrainSize = 1.0f;
	int channelOrder = ANON_ORDER_RGBA;
	Warp warp = StaticWarp(0u);
	uint32_t cellSeed = 0u;
	// Input colour is premultiplied by alpha (Resolve hands a power window
	// over that way). See UnpremultiplyPass.
	bool premultiplied = false;
};

/* Parameter values as the user sees them: 1080p-reference pixels, percent. */
struct UserParams
{
	double distortAmount = DISTORT_AMOUNT_DFLT;
	double distortScale = DISTORT_SCALE_DFLT;
	double blurRadius = BLUR_RADIUS_DFLT;
	double mosaicSize = MOSAIC_SIZE_DFLT;
	bool enableDistort = true;
	bool enableBlur = true;
	bool enableMosaic = true;
	int mosaicShape = ANON_SHAPE_SQUARE;
	bool blurAfterMosaic = false;
	double desaturate = DESATURATE_DFLT;
	double cellNoise = CELL_NOISE_DFLT;
	double colorNoise = COLOR_NOISE_DFLT;
	bool frostedGlass = false;
	double frostAmount = FROST_AMOUNT_DFLT;
	double frostColor[3] = { FROST_COLOR_DFLT, FROST_COLOR_DFLT, FROST_COLOR_DFLT }; // R, G, B, 0..1
};

// ds is AnonResolutionScale of the frame (or document) being rendered.
inline LayerSettings MakeLayerSettings(const UserParams& u, float ds, const Warp& warp,
	uint32_t cellSeed, int channelOrder)
{
	LayerSettings s;
	s.distortAmount = LayerStrength(u.distortAmount, DISTORT_AMOUNT_MIN, u.enableDistort, ds, 0.0f);
	s.distortScale = std::max((float)u.distortScale * ds, 2.0f);
	s.blurRadius = std::min(LayerStrength(u.blurRadius, BLUR_RADIUS_MIN, u.enableBlur, ds, 0.0f), 512.0f);
	s.mosaicSize = std::max(LayerStrength(u.mosaicSize, MOSAIC_SIZE_MIN, u.enableMosaic, ds, 1.0f), 1.0f);
	s.mosaicShape = (u.mosaicShape >= ANON_SHAPE_SQUARE && u.mosaicShape <= ANON_SHAPE_HEXAGON)
		? u.mosaicShape : ANON_SHAPE_SQUARE;
	s.blurAfterMosaic = u.blurAfterMosaic;
	s.desaturate = (float)std::min(std::max(u.desaturate, 0.0), 100.0) / 100.0f;
	s.cellNoise = (float)u.cellNoise / 100.0f;
	s.colorNoise = (float)u.colorNoise / 100.0f;
	// Blur After Mosaic softens the noise into colour blotches and mostly
	// averages it away; boost it to keep a comparable effect. Frosted Glass
	// is left as set: on its smooth surface the blotches stay visible.
	if (u.blurAfterMosaic && s.blurRadius >= 1.0f && !u.frostedGlass)
		s.colorNoise *= COLOR_NOISE_BLURRED_GAIN;
	s.frostedGlass = u.frostedGlass;
	s.frostAmount = (float)std::min(std::max(u.frostAmount, 0.0), 100.0) / 100.0f;
	const float r = (float)std::max(u.frostColor[0], 0.0);
	const float g = (float)std::max(u.frostColor[1], 0.0);
	const float b = (float)std::max(u.frostColor[2], 0.0);
	const float rgbaOrder[4] = { r, g, b, 1.0f };
	const float bgraOrder[4] = { b, g, r, 1.0f };
	const float argbOrder[4] = { 1.0f, r, g, b };
	const float* color = channelOrder == ANON_ORDER_BGRA ? bgraOrder
		: channelOrder == ANON_ORDER_ARGB ? argbOrder : rgbaOrder;
	std::copy(color, color + 4, s.frostColor);
	// Wider than a cell: melts the cells and the face-scale shapes they form.
	s.frostBlurRadius = std::min(FROST_BLUR_CELLS
		* std::max(s.mosaicSize, (float)MOSAIC_SIZE_DFLT * ds), 512.0f);
	s.frostGrainSize = std::max(roundf(1.5f * ds), 1.0f);
	s.channelOrder = channelOrder;
	s.warp = warp;
	s.cellSeed = cellSeed;
	return s;
}

/*
** Premultiplied input (e.g. a Resolve power window in alpha) has no colour
** outside the matte and dimmed colour along its soft edge. Running the
** layers on that would pull black into every cell and blur that reaches
** across the edge. So the stack runs on straight colour instead: colour /
** alpha where there is enough alpha, and elsewhere colour pulled outward
** from inside the matte (an alpha-normalized blur of radius FillRadius,
** the farthest any layer reads). The result is multiplied by the source
** alpha again at the end, so the edge is exactly the host's matte.
*/
#define ANON_UNPREMULTIPLY_MIN_ALPHA (1.0f / 1024.0f)

inline float FillRadius(const LayerSettings& s)
{
	const float reach = s.distortAmount + s.blurRadius + s.mosaicSize
		+ (s.frostedGlass ? s.frostBlurRadius : 0.0f);
	return std::min(ceilf(reach), 512.0f);
}

// fill is src blurred with blurAlpha. Keep in sync with AnonUnpremultiply.
inline void UnpremultiplyPass(const float* src, const float* fill, float* dst, int w, int h,
	int channelOrder)
{
	const int alpha = AlphaIndex(channelOrder);
	for (size_t i = 0; i < (size_t)w * h; ++i)
	{
		const float* c = src + i * 4;
		const float* f = fill + i * 4;
		float* out = dst + i * 4;
		const float a = c[alpha];
		for (int k = 0; k < 4; ++k)
		{
			if (a > ANON_UNPREMULTIPLY_MIN_ALPHA)
				out[k] = c[k] / a;
			else if (f[alpha] > 1e-6f)
				out[k] = f[k] / f[alpha];
			else
				out[k] = 0.0f;
		}
		out[alpha] = a;
	}
}

inline void PremultiplyPass(float* buf, int w, int h, int channelOrder)
{
	const int alpha = AlphaIndex(channelOrder);
	for (size_t i = 0; i < (size_t)w * h; ++i)
	{
		float* p = buf + i * 4;
		for (int k = 0; k < 4; ++k)
			if (k != alpha)
				p[k] *= p[alpha];
	}
}

// Runs the full stack in place on a packed float buffer, using the caller's
// scratch buffer of the same size. Result lands in `buf`.
inline void RunLayeredPasses(float* buf, float* scratch, int w, int h, const LayerSettings& s)
{
	if (s.premultiplied)
	{
		std::vector<float> fill((size_t)w * h * 4);
		const float radius = FillRadius(s);
		BlurPass(buf, scratch, w, h, radius, 0, s.channelOrder, true);
		BlurPass(scratch, fill.data(), w, h, radius, 1, s.channelOrder, true);
		UnpremultiplyPass(buf, fill.data(), buf, w, h, s.channelOrder);
	}
	DistortPass(buf, scratch, w, h, s.distortAmount, s.distortScale, s.warp, s.channelOrder);
	if (s.blurAfterMosaic)
	{
		MosaicPass(scratch, buf, w, h, s.mosaicSize, s.mosaicShape, s.cellSeed, s.cellNoise,
			s.channelOrder, s.colorNoise);
		BlurPass(buf, scratch, w, h, s.blurRadius, 0, s.channelOrder);
		BlurPass(scratch, buf, w, h, s.blurRadius, 1, s.channelOrder);
	}
	else
	{
		BlurPass(scratch, buf, w, h, s.blurRadius, 0, s.channelOrder);
		BlurPass(buf, scratch, w, h, s.blurRadius, 1, s.channelOrder);
		MosaicPass(scratch, buf, w, h, s.mosaicSize, s.mosaicShape, s.cellSeed, s.cellNoise,
			s.channelOrder, s.colorNoise);
	}
	DesaturatePass(buf, w, h, s.desaturate, s.channelOrder);
	if (s.frostedGlass)
	{
		BlurPass(buf, scratch, w, h, s.frostBlurRadius, 0, s.channelOrder);
		BlurPass(scratch, buf, w, h, s.frostBlurRadius, 1, s.channelOrder);
		FrostPass(buf, w, h, s.frostAmount, s.frostColor, s.frostGrainSize, s.cellSeed,
			s.channelOrder);
	}
	if (s.premultiplied)
		PremultiplyPass(buf, w, h, s.channelOrder);
}

} // namespace AnonAlgo

#endif // ANONYMIZER_ALGO_H
