/*
** Renders the Final Cut Pro wrapper app's icon (fcp/AppIcon.icns) with the
** effect's own passes: a portrait, pixelated by the mosaic layer, half
** covered by a pane of Frosted Glass. Not part of the build - rerun by hand
** after changing the design:
**
**   clang++ -std=c++17 -O2 -fobjc-arc -Ishared scripts/make_app_icon.mm \
**       -framework Foundation -framework CoreGraphics -framework ImageIO \
**       -framework UniformTypeIdentifiers -o /tmp/make_app_icon
**   /tmp/make_app_icon /tmp/AppIcon-1024.png
**   mkdir /tmp/AppIcon.iconset
**   for s in 16 32 128 256 512; do
**     sips -z $s $s /tmp/AppIcon-1024.png --out /tmp/AppIcon.iconset/icon_${s}x${s}.png
**     sips -z $((s*2)) $((s*2)) /tmp/AppIcon-1024.png \
**         --out /tmp/AppIcon.iconset/icon_${s}x${s}@2x.png
**   done
**   iconutil -c icns /tmp/AppIcon.iconset -o fcp/AppIcon.icns
*/

#import <Foundation/Foundation.h>
#import <CoreGraphics/CoreGraphics.h>
#import <ImageIO/ImageIO.h>
#import <UniformTypeIdentifiers/UniformTypeIdentifiers.h>

#include "AnonymizerAlgo.h"

#include <vector>

static const int kSize = 1024;

// macOS icon grid: an 824 px rounded square centred on the 1024 canvas.
static const CGRect kBody = { { 100, 100 }, { 824, 824 } };
static const CGFloat kCorner = 185;

static CGContextRef NewContext(std::vector<uint8_t>& px)
{
	px.assign((size_t)kSize * kSize * 4, 0);
	CGColorSpaceRef cs = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
	CGContextRef ctx = CGBitmapContextCreate(px.data(), kSize, kSize, 8, kSize * 4, cs,
		kCGImageAlphaPremultipliedLast);
	CGColorSpaceRelease(cs);
	return ctx;
}

static void ToFloat(const std::vector<uint8_t>& px, std::vector<float>& out)
{
	out.resize(px.size());
	for (size_t i = 0; i < px.size(); ++i)
		out[i] = px[i] / 255.0f;
}

static void ToBytes(const std::vector<float>& in, std::vector<uint8_t>& px)
{
	for (size_t i = 0; i < in.size(); ++i)
		px[i] = (uint8_t)std::min(std::max(in[i] * 255.0f + 0.5f, 0.0f), 255.0f);
}

int main(int argc, char** argv)
{
	@autoreleasepool
	{
		if (argc < 2)
		{
			fprintf(stderr, "usage: make_app_icon out.png\n");
			return 1;
		}

		// 1. The subject: head and shoulders on a deep blue backdrop.
		std::vector<uint8_t> scene;
		CGContextRef ctx = NewContext(scene);
		CGColorSpaceRef cs = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
		const CGFloat bg[] = { 0.42, 0.66, 0.74, 1.0,  0.20, 0.44, 0.60, 1.0 };
		CGGradientRef grad = CGGradientCreateWithColorComponents(cs, bg, NULL, 2);
		CGContextDrawLinearGradient(ctx, grad, CGPointMake(0, kSize), CGPointMake(kSize, 0), 0);
		CGGradientRelease(grad);
		// CG origin is bottom-left.
		CGContextSetRGBFillColor(ctx, 0.08, 0.12, 0.26, 1.0); // jacket
		CGContextFillEllipseInRect(ctx, CGRectMake(470 - 360, -170, 720, 560));
		CGContextSetRGBFillColor(ctx, 0.95, 0.96, 0.98, 1.0); // collar
		CGContextFillEllipseInRect(ctx, CGRectMake(470 - 95, 290, 190, 120));
		CGContextSetRGBFillColor(ctx, 0.92, 0.70, 0.55, 1.0); // neck and head
		CGContextFillRect(ctx, CGRectMake(470 - 65, 320, 130, 130));
		CGContextFillEllipseInRect(ctx, CGRectMake(470 - 185, 390, 370, 440));
		CGContextSetRGBFillColor(ctx, 0.22, 0.14, 0.10, 1.0); // hair
		CGContextFillEllipseInRect(ctx, CGRectMake(470 - 200, 660, 400, 210));

		// 2. Anonymize it with the effect's mosaic and frost.
		std::vector<float> img, scratch;
		ToFloat(scene, img);
		scratch.resize(img.size());
		AnonAlgo::MosaicPass(img.data(), scratch.data(), kSize, kSize, 56.0f, ANON_SHAPE_SQUARE,
			7u, 0.03f, ANON_ORDER_RGBA, 0.02f);
		std::vector<float> frosted = scratch, tmp(img.size());
		AnonAlgo::BlurPass(frosted.data(), tmp.data(), kSize, kSize, 70.0f, 0);
		AnonAlgo::BlurPass(tmp.data(), frosted.data(), kSize, kSize, 70.0f, 1);
		const float frostColor[4] = { 0.86f, 0.90f, 0.95f, 1.0f };
		AnonAlgo::FrostPass(frosted.data(), kSize, kSize, 0.45f, frostColor, 3.0f, 11u,
			ANON_ORDER_RGBA);

		// The frosted pane: a tilted rounded rectangle over the right third.
		std::vector<uint8_t> paneMask;
		CGContextRef maskCtx = NewContext(paneMask);
		CGContextTranslateCTM(maskCtx, 790, 500);
		CGContextRotateCTM(maskCtx, -0.18);
		CGRect pane = CGRectMake(-210, -440, 420, 880);
		CGPathRef panePath = CGPathCreateWithRoundedRect(pane, 70, 70, NULL);
		CGContextAddPath(maskCtx, panePath);
		CGContextSetRGBFillColor(maskCtx, 1, 1, 1, 1);
		CGContextFillPath(maskCtx);

		std::vector<float> out(img.size());
		for (size_t i = 0; i < out.size(); i += 4)
		{
			const float m = paneMask[i + 3] / 255.0f;
			for (int c = 0; c < 3; ++c)
				out[i + c] = scratch[i + c] + (frosted[i + c] - scratch[i + c]) * m;
			out[i + 3] = 1.0f;
		}
		std::vector<uint8_t> art(scene.size());
		ToBytes(out, art);

		// Glass edge highlight.
		CGContextRef artCtx = NewContext(scene);
		CGDataProviderRef prov = CGDataProviderCreateWithData(NULL, art.data(), art.size(), NULL);
		CGImageRef artImage = CGImageCreate(kSize, kSize, 8, 32, kSize * 4, cs,
			kCGImageAlphaPremultipliedLast, prov, NULL, false, kCGRenderingIntentDefault);
		CGContextDrawImage(artCtx, CGRectMake(0, 0, kSize, kSize), artImage);
		CGContextTranslateCTM(artCtx, 790, 500);
		CGContextRotateCTM(artCtx, -0.18);
		CGContextAddPath(artCtx, panePath);
		CGContextSetRGBStrokeColor(artCtx, 1, 1, 1, 0.75);
		CGContextSetLineWidth(artCtx, 6);
		CGContextStrokePath(artCtx);
		CGImageRef finalArt = CGBitmapContextCreateImage(artCtx);

		// 3. Clip to the icon shape with a soft drop shadow.
		std::vector<uint8_t> iconPx;
		CGContextRef icon = NewContext(iconPx);
		CGPathRef body = CGPathCreateWithRoundedRect(kBody, kCorner, kCorner, NULL);
		CGContextSaveGState(icon);
		CGContextSetShadowWithColor(icon, CGSizeMake(0, -10), 28,
			CGColorCreateSRGB(0, 0, 0, 0.35));
		CGContextAddPath(icon, body);
		CGContextSetRGBFillColor(icon, 0, 0, 0, 1);
		CGContextFillPath(icon);
		CGContextRestoreGState(icon);
		CGContextAddPath(icon, body);
		CGContextClip(icon);
		CGContextDrawImage(icon, CGRectMake(0, 0, kSize, kSize), finalArt);

		CGImageRef result = CGBitmapContextCreateImage(icon);
		CGImageDestinationRef dst = CGImageDestinationCreateWithURL(
			(__bridge CFURLRef)[NSURL fileURLWithPath:@(argv[1])],
			(__bridge CFStringRef)UTTypePNG.identifier, 1, NULL);
		CGImageDestinationAddImage(dst, result, NULL);
		const bool ok = CGImageDestinationFinalize(dst);
		printf("%s %s\n", ok ? "wrote" : "FAILED", argv[1]);
		return ok ? 0 : 1;
	}
}
