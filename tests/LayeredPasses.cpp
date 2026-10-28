#include "AnonymizerAlgo.h"
#ifdef ANON_TEST_METAL
#include "AnonymizerPS.h"
#endif
#include <cmath>
#include <cstdio>
#include <utility>
#include <cstdlib>

static void check(bool ok, const char* message)
{
	if (!ok) { std::fprintf(stderr, "%s\n", message); std::exit(1); }
}

int main()
{
	check(AnonAlgo::BlurRadiusForOrderChange(20.0, true) == 40.0,
		"Enabling blur after mosaic should promote the default to 40");
	check(AnonAlgo::BlurRadiusForOrderChange(40.0, false) == 20.0,
		"Disabling blur after mosaic should restore the default to 20");
	check(AnonAlgo::BlurRadiusForOrderChange(15.0, true) == 35.0,
		"Enabling blur after mosaic should promote the legacy default to 35");
	check(AnonAlgo::BlurRadiusForOrderChange(35.0, false) == 15.0,
		"Disabling blur after mosaic should restore the legacy default to 15");
	for (double radius : {0.0, 10.0, 15.0, 18.0, 20.0, 25.0, 35.0, 40.0, 100.0})
	{
		if (radius != BLUR_AFTER_MOSAIC_RADIUS_DFLT && radius != BLUR_AFTER_MOSAIC_RADIUS_LEGACY_DFLT)
			check(AnonAlgo::BlurRadiusForOrderChange(radius, false) == radius,
				"Disabling blur after mosaic should preserve a customized radius");
		if (radius != BLUR_RADIUS_DFLT && radius != BLUR_RADIUS_LEGACY_DFLT)
			check(AnonAlgo::BlurRadiusForOrderChange(radius, true) == radius,
				"Enabling blur after mosaic should preserve a customized radius");
	}
	check(AnonComputeSeed(7.0, true, 3, 0u) != AnonComputeSeed(7.0, true, 3, 0x1234u),
		"A session key should change the derived seed");
	check(AnonComputeSeed(7.0, true, 3, 0x1234u) == AnonComputeSeed(7.0, true, 3, 0x1234u),
		"The derived seed should be deterministic for a fixed session key");
	for (int i = 0; i < 1000; ++i)
	{
		const int v = AnonAlgo::RandomSeedValue();
		check(v >= 1 && v <= (int)SEED_MAX, "Random seed out of range");
	}
	{
		// Cell noise: one offset per cell, alpha untouched, output changed.
		const int w = 40, h = 40;
		std::vector<float> flat((size_t)w * h * 4, 0.5f), plain(flat.size()), noisy(flat.size());
		AnonAlgo::MosaicPass(flat.data(), plain.data(), w, h, 10.0f, ANON_SHAPE_SQUARE);
		AnonAlgo::MosaicPass(flat.data(), noisy.data(), w, h, 10.0f, ANON_SHAPE_SQUARE, 42u, 0.05f);
		check(plain != noisy, "Cell noise should change the mosaic");
		for (int y = 0; y < h; ++y)
		for (int x = 0; x < w; ++x)
		{
			const float* p = &noisy[((size_t)y * w + x) * 4];
			const float* c = &noisy[((size_t)(y / 10 * 10) * w + x / 10 * 10) * 4];
			check(p[0] == c[0] && p[1] == p[0] && p[2] == p[0], "Cell offset should be uniform per cell");
			check(std::abs(p[0] - 0.5f) <= 0.05f + 1e-6f, "Cell offset out of range");
			check(p[3] == 0.5f, "Cell noise must not touch alpha");
		}
		// Colour noise: per-channel offsets, uniform per cell, alpha untouched.
		std::vector<float> tinted(flat.size());
		AnonAlgo::MosaicPass(flat.data(), tinted.data(), w, h, 10.0f, ANON_SHAPE_SQUARE, 42u, 0.0f,
			ANON_ORDER_RGBA, 0.05f);
		bool hueShift = false;
		for (int y = 0; y < h; ++y)
		for (int x = 0; x < w; ++x)
		{
			const float* p = &tinted[((size_t)y * w + x) * 4];
			const float* c = &tinted[((size_t)(y / 10 * 10) * w + x / 10 * 10) * 4];
			for (int ch = 0; ch < 3; ++ch)
			{
				check(p[ch] == c[ch], "Colour offset should be uniform per cell");
				check(std::abs(p[ch] - 0.5f) <= 0.1f + 1e-6f, "Colour offset out of range");
			}
			const float luma = p[0] * 0.2126f + p[1] * 0.7152f + p[2] * 0.0722f;
			check(std::abs(luma - 0.5f) < 1e-5f, "Colour noise should not change brightness");
			check(p[3] == 0.5f, "Colour noise must not touch alpha");
			hueShift = hueShift || p[0] != p[1] || p[1] != p[2];
		}
		check(hueShift, "Colour noise should offset channels independently");
	}
	check(AnonAlgo::LayerStrength(5.0, 13.0, true, 2.0f, 0.0f) == 26.0f,
		"Values below the minimum should be raised to it");
	check(AnonAlgo::LayerStrength(20.0, 13.0, true, 2.0f, 0.0f) == 40.0f,
		"Values above the minimum should only be scaled");
	check(AnonAlgo::LayerStrength(20.0, 13.0, false, 2.0f, 1.0f) == 1.0f,
		"A disabled layer should return its off value");
	{
		// The cell seed ignores the frame but follows the session key.
		check(AnonAlgo::ComputeCellSeed(7.0, 0x1234u) != AnonAlgo::ComputeCellSeed(7.0, 0x5678u),
			"A session key should change the cell seed");
		check(AnonAlgo::ComputeCellSeed(7.0, 0x1234u) != AnonComputeSeed(7.0, false, 0, 0x1234u),
			"The cell seed should differ from the static warp seed");
		// Smooth jitter: keyframes are exact, blends keep constant strength.
		const AnonAlgo::Warp hard = AnonAlgo::ComputeWarp(7.0, true, false, 25, 9u);
		check(hard.seed == AnonComputeSeed(7.0, true, 25, 9u) && hard.mix == 1.0f && hard.mix2 == 0.0f,
			"Plain jitter should use the per-frame seed");
		const AnonAlgo::Warp still = AnonAlgo::ComputeWarp(7.0, false, true, 25, 9u);
		check(still.seed == AnonComputeSeed(7.0, false, 25, 9u) && still.mix2 == 0.0f,
			"Smooth jitter should do nothing without temporal jitter");
		for (int32_t f = -30; f <= 30; ++f)
		{
			const AnonAlgo::Warp a = AnonAlgo::ComputeWarp(7.0, true, true, f, 9u);
			check(std::abs(a.mix * a.mix + a.mix2 * a.mix2 - 1.0f) < 1e-5f,
				"Smooth jitter weights should keep the warp strength constant");
			if (f % SMOOTH_JITTER_FRAMES == 0)
				check(a.mix == 1.0f && a.mix2 == 0.0f, "Keyframes should be a single pattern");
			const AnonAlgo::Warp b = AnonAlgo::ComputeWarp(7.0, true, true, f + 1, 9u);
			check(a.seed == b.seed || a.seed2 == b.seed,
				"Smooth jitter should hand over between neighbouring patterns");
		}
		// Smooth jitter changes the warp far less between frames.
		const int w = 64, h = 48;
		float hardStep = 0.0f, smoothStep = 0.0f;
		for (int f = 1; f < 2 * SMOOTH_JITTER_FRAMES; ++f)
		for (int y = 0; y < h; y += 4)
		for (int x = 0; x < w; x += 4)
		{
			float ax, ay, bx, by;
			AnonAlgo::WarpDisplacement(x, y, 0.1f, AnonAlgo::ComputeWarp(7.0, true, false, f - 1, 9u), &ax, &ay);
			AnonAlgo::WarpDisplacement(x, y, 0.1f, AnonAlgo::ComputeWarp(7.0, true, false, f, 9u), &bx, &by);
			hardStep += std::abs(ax - bx) + std::abs(ay - by);
			AnonAlgo::WarpDisplacement(x, y, 0.1f, AnonAlgo::ComputeWarp(7.0, true, true, f - 1, 9u), &ax, &ay);
			AnonAlgo::WarpDisplacement(x, y, 0.1f, AnonAlgo::ComputeWarp(7.0, true, true, f, 9u), &bx, &by);
			smoothStep += std::abs(ax - bx) + std::abs(ay - by);
		}
		check(smoothStep * 4.0f < hardStep, "Smooth jitter should move the warp gradually");
	}
	{
		// Frost: lifts toward white, adds bounded grain, leaves alpha alone.
		const int w = 16, h = 16;
		std::vector<float> img((size_t)w * h * 4, 0.2f);
		for (size_t i = 3; i < img.size(); i += 4)
			img[i] = 1.0f;
		const float white[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
		AnonAlgo::FrostPass(img.data(), w, h, 0.5f, white, 2.0f, 42u, ANON_ORDER_RGBA);
		const float grain = FROST_GRAIN_FLOOR + FROST_GRAIN * 0.5f;
		bool varies = false;
		for (size_t i = 0; i < img.size(); i += 4)
		{
			check(std::abs(img[i] - 0.6f) <= grain + 1e-6f, "Frost lift out of range");
			check(img[i] == img[i + 1] && img[i] == img[i + 2], "Frost grain should be neutral");
			check(img[i + 3] == 1.0f, "Frost must not touch alpha");
			varies = varies || img[i] != img[0];
		}
		check(varies, "Frost should add grain");
	}
	{
		// Colour noise is boosted only when a blur follows the mosaic.
		AnonAlgo::UserParams u;
		u.colorNoise = 2.0;
		const AnonAlgo::Warp warp = AnonAlgo::StaticWarp(1u);
		check(AnonAlgo::MakeLayerSettings(u, 1.0f, warp, 1u, ANON_ORDER_RGBA).colorNoise == 0.02f,
			"Crisp mosaic should keep the colour noise as set");
		u.blurAfterMosaic = true;
		check(AnonAlgo::MakeLayerSettings(u, 1.0f, warp, 1u, ANON_ORDER_RGBA).colorNoise
			== 0.02f * COLOR_NOISE_BLURRED_GAIN, "Blur After Mosaic should boost the colour noise");
		u.enableBlur = false;
		check(AnonAlgo::MakeLayerSettings(u, 1.0f, warp, 1u, ANON_ORDER_RGBA).colorNoise == 0.02f,
			"A disabled blur should not boost the colour noise");
		u.enableBlur = true;
		u.frostedGlass = true;
		check(AnonAlgo::MakeLayerSettings(u, 1.0f, warp, 1u, ANON_ORDER_RGBA).colorNoise == 0.02f,
			"Frosted Glass should keep the colour noise as set");
	}
	{
		// Premultiplied input: a flat colour inside a soft round matte must
		// come out as that same colour times the matte - no black pulled in
		// from outside, and the matte itself untouched.
		const int w = 96, h = 80;
		std::vector<float> img((size_t)w * h * 4), scratch(img.size());
		for (int y = 0; y < h; ++y)
		for (int x = 0; x < w; ++x)
		{
			const float d = std::sqrt((float)((x - 48) * (x - 48) + (y - 40) * (y - 40)));
			const float a = std::min(std::max((30.0f - d) / 6.0f, 0.0f), 1.0f);
			float* p = &img[((size_t)y * w + x) * 4];
			p[0] = 0.6f * a; p[1] = 0.3f * a; p[2] = 0.2f * a; p[3] = a;
		}
		const std::vector<float> input = img;
		AnonAlgo::LayerSettings s;
		s.distortAmount = 6.0f;
		s.distortScale = 5.0f;
		s.blurRadius = 4.0f;
		s.mosaicSize = 9.0f;
		s.warp = AnonAlgo::StaticWarp(42u);
		s.premultiplied = true;
		AnonAlgo::RunLayeredPasses(img.data(), scratch.data(), w, h, s);
		for (size_t i = 0; i < img.size(); i += 4)
		{
			const float a = input[i + 3];
			check(img[i + 3] == a, "Premultiplied path must keep the matte");
			check(std::abs(img[i] - 0.6f * a) < 1e-4f && std::abs(img[i + 1] - 0.3f * a) < 1e-4f
				&& std::abs(img[i + 2] - 0.2f * a) < 1e-4f,
				"Premultiplied path should not pull black in across the matte edge");
		}
#ifdef ANON_TEST_METAL
		img = input;
		check(RunMetalPasses(img.data(), w, h, s), "Metal pipeline failed");
		for (size_t i = 0; i < img.size(); i += 4)
			check(img[i + 3] == input[i + 3] && std::abs(img[i] - 0.6f * input[i + 3]) < 1e-4f,
				"Metal premultiplied path differs");
#endif
		// Fully opaque premultiplied input matches the straight path.
		std::vector<float> opaque((size_t)w * h * 4, 1.0f);
		for (size_t i = 0; i < opaque.size(); ++i)
			if (i % 4 != 3)
				opaque[i] = (float)((i * 37 + i / 7) % 101) / 100.0f;
		std::vector<float> straight = opaque;
		s.premultiplied = false;
		AnonAlgo::RunLayeredPasses(straight.data(), scratch.data(), w, h, s);
		s.premultiplied = true;
		AnonAlgo::RunLayeredPasses(opaque.data(), scratch.data(), w, h, s);
		for (size_t i = 0; i < opaque.size(); ++i)
			check(std::abs(opaque[i] - straight[i]) < 1e-5f,
				"Opaque premultiplied input should match the straight path");
	}
	{
		// Frost Colour lands in the buffer's channel order.
		AnonAlgo::UserParams u;
		u.frostColor[0] = 0.9; u.frostColor[1] = 0.6; u.frostColor[2] = 0.3;
		const AnonAlgo::Warp warp = AnonAlgo::StaticWarp(1u);
		const auto rgba = AnonAlgo::MakeLayerSettings(u, 1.0f, warp, 1u, ANON_ORDER_RGBA);
		const auto bgra = AnonAlgo::MakeLayerSettings(u, 1.0f, warp, 1u, ANON_ORDER_BGRA);
		const auto argb = AnonAlgo::MakeLayerSettings(u, 1.0f, warp, 1u, ANON_ORDER_ARGB);
		check(rgba.frostColor[0] == 0.9f && rgba.frostColor[2] == 0.3f, "RGBA frost colour order");
		check(bgra.frostColor[0] == 0.3f && bgra.frostColor[2] == 0.9f, "BGRA frost colour order");
		check(argb.frostColor[1] == 0.9f && argb.frostColor[3] == 0.3f, "ARGB frost colour order");
	}
	int cases = 0;
	for (auto size : {std::pair<int, int>{1, 1}, {7, 3}, {31, 23}})
	for (int shape = 0; shape < 3; ++shape)
	for (float radius : {0.0f, 0.5f, 3.5f, 15.0f})
	for (float block : {1.0f, 5.0f, 40.0f})
	for (float amount : {0.0f, 4.0f})
	{
		const int w = size.first, h = size.second;
		std::vector<float> input((size_t)w * h * 4);
		for (size_t i = 0; i < input.size(); ++i)
			input[i] = (float)((i * 37 + i / 7) % 101) / 100.0f;
		std::vector<float> a(input.size()), b(input.size()), scratch(input.size());
		for (bool after : {false, true})
		for (float desat : {0.0f, 0.6f})
		for (float noise : {0.0f, 0.05f})
		for (float tint : {0.0f, 0.04f})
		for (int variant = 0; variant < 4; ++variant)
		{
			// variant 1 blends two warps (Smooth Jitter), variant 2 adds the
			// frost, variant 3 treats the input as premultiplied.
			AnonAlgo::LayerSettings s;
			s.distortAmount = amount;
			s.distortScale = 5.0f;
			s.blurRadius = radius;
			s.mosaicSize = block;
			s.mosaicShape = shape;
			s.blurAfterMosaic = after;
			s.desaturate = desat;
			s.cellNoise = noise;
			s.colorNoise = tint;
			s.warp = variant == 1 ? AnonAlgo::Warp{ 42u, 43u, 0.6f, 0.8f } : AnonAlgo::StaticWarp(42u);
			s.cellSeed = 99u;
			s.frostedGlass = variant == 2;
			s.frostAmount = 0.4f;
			s.frostBlurRadius = 6.5f;
			s.frostGrainSize = 2.0f;
			// A tint, so a channel mix-up between CPU and Metal would show.
			s.frostColor[0] = 0.9f;
			s.frostColor[1] = 0.6f;
			s.frostColor[2] = 0.3f;
			s.premultiplied = variant == 3;

			// Explicit reference stages verify ordering and final buffer ownership.
			AnonAlgo::DistortPass(input.data(), a.data(), w, h, amount, 5.0f, s.warp);
			if (after)
			{
				AnonAlgo::MosaicPass(a.data(), b.data(), w, h, block, shape, 99u, noise, ANON_ORDER_RGBA, tint);
				AnonAlgo::BlurPass(b.data(), a.data(), w, h, radius, 0);
				AnonAlgo::BlurPass(a.data(), b.data(), w, h, radius, 1);
			}
			else
			{
				AnonAlgo::BlurPass(a.data(), b.data(), w, h, radius, 0);
				AnonAlgo::BlurPass(b.data(), a.data(), w, h, radius, 1);
				AnonAlgo::MosaicPass(a.data(), b.data(), w, h, block, shape, 99u, noise, ANON_ORDER_RGBA, tint);
			}
			AnonAlgo::DesaturatePass(b.data(), w, h, desat, ANON_ORDER_RGBA);
			if (s.frostedGlass)
			{
				AnonAlgo::BlurPass(b.data(), a.data(), w, h, 6.5f, 0);
				AnonAlgo::BlurPass(a.data(), b.data(), w, h, 6.5f, 1);
				AnonAlgo::FrostPass(b.data(), w, h, 0.4f, s.frostColor, 2.0f, 99u, ANON_ORDER_RGBA);
			}
			a = input;
			AnonAlgo::RunLayeredPasses(a.data(), scratch.data(), w, h, s);
			for (size_t i = 3; i < a.size(); i += 4)
				check(a[i] == input[i], "The layers must not change alpha");
			if (s.premultiplied)
				b = a; // no explicit reference; checked against Metal below
			// Not bit-exact: the compiler may fuse multiply-adds differently
			// at the two inlined call sites.
			for (size_t i = 0; i < a.size(); ++i)
				check(std::abs(a[i] - b[i]) < 1e-6f, "CPU stage order differs from reference");
#ifdef ANON_TEST_METAL
			a = input;
			check(RunMetalPasses(a.data(), w, h, s), "Metal pipeline failed");
			for (size_t i = 0; i < a.size(); ++i)
				check(std::isfinite(a[i]) && std::abs(a[i] - b[i]) < 2e-5f,
					"CPU/Metal output differs");
#endif
			++cases;
		}
		if (radius == 0.0f)
		{
			AnonAlgo::LayerSettings s;
			s.distortAmount = amount;
			s.distortScale = 5.0f;
			s.mosaicSize = block;
			s.mosaicShape = shape;
			s.warp = AnonAlgo::StaticWarp(42u);
			a = input;
			b = input;
			AnonAlgo::RunLayeredPasses(a.data(), scratch.data(), w, h, s);
			s.blurAfterMosaic = true;
			AnonAlgo::RunLayeredPasses(b.data(), scratch.data(), w, h, s);
			check(a == b, "Zero blur should make both orders identical");
		}
	}
	std::printf("Passed %d processing-order/desaturation/noise/smooth-jitter/frost cases\n", cases);
}
