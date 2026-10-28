/*
** Redact Anonymizer - GPU render path (Metal).
**
** Premiere Pro binds this GPU filter to the AE-style effect in the same
** binary via the PiPL match name (PrGPUFilterInfo::outMatchName is left null,
** which defaults to the module's PiPL - see PrSDKGPUFilter.h).
**
** Render pipeline (all on the GPU, no readbacks):
**   input -> AnonDistort -> tmpA -> AnonBlur(H) -> tmpB -> AnonBlur(V)
**         -> tmpA -> AnonMosaic -> output
** (see AnonEncodeLayeredPasses; Blur After Mosaic and Frosted Glass change
** the stage list).
*/

#include "Anonymizer.h"

#include "PrGPUFilterModule.h"
#include "PrSDKVideoSegmentProperties.h"

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "AnonymizerKernel.h"

#include <math.h>
#include <algorithm>


static prSuiteError CheckForMetalError(NSError* inError)
{
	if (inError)
	{
		// For debugging: NSLog(@"Anonymizer Metal error: %@", [inError localizedDescription]);
		return suiteError_Fail;
	}
	return suiteError_NoError;
}

/*
** Colour parameters reach GPU filters packed into one integer: 16 bits per
** channel ARGB in an Int64 (8 bits per channel in an Int32 on older hosts).
** Compare against the Software Only renderer if the colour looks wrong.
*/
static void UnpackColorParam(const PrParam& inParam, double outRGB[3])
{
	if (inParam.mType == kPrParamType_Int64)
	{
		const uint64_t v = (uint64_t)inParam.mInt64;
		outRGB[0] = (double)((v >> 32) & 0xFFFF) / 65535.0;
		outRGB[1] = (double)((v >> 16) & 0xFFFF) / 65535.0;
		outRGB[2] = (double)(v & 0xFFFF) / 65535.0;
	}
	else if (inParam.mType == kPrParamType_Int32)
	{
		const uint32_t v = (uint32_t)inParam.mInt32;
		outRGB[0] = (double)((v >> 16) & 0xFF) / 255.0;
		outRGB[1] = (double)((v >> 8) & 0xFF) / 255.0;
		outRGB[2] = (double)(v & 0xFF) / 255.0;
	}
}

static size_t DivideRoundUp(size_t inValue, size_t inMultiple)
{
	return inValue ? (inValue + inMultiple - 1) / inMultiple : 0;
}

/*
** Plugins must not rely on a host autorelease pool (see SDK_ProcAmp sample).
*/
struct ScopedAutoreleasePool
{
	ScopedAutoreleasePool() : mPool([[NSAutoreleasePool alloc] init]) {}
	~ScopedAutoreleasePool() { [mPool release]; }
	NSAutoreleasePool* mPool;
};

enum { kMaxDevices = 12 };
// The layered-pass kernels (ANON_KERNEL_*), then blackout.
enum { kKernelBlackout = ANON_KERNEL_LAYER_COUNT, kKernelCount };
static id<MTLComputePipelineState> sPipelineCache[kMaxDevices][kKernelCount] = {};

static const char* KernelName(int k)
{
	return k == kKernelBlackout ? "AnonBlackout" : kAnonLayerKernelNames[k];
}

/*
**
*/
class Anonymizer :
	public PrGPUFilterBase
{
public:
	virtual prSuiteError Initialize(
		PrGPUFilterInstance* ioInstanceData)
	{
		PrGPUFilterBase::Initialize(ioInstanceData);

		if (mDeviceIndex >= kMaxDevices)
			return suiteError_Fail;

		// This plugin accelerates Metal only. Returning an error here makes
		// Premiere fall back to the software path in Anonymizer_CPU.cpp for
		// other frameworks (OpenCL/CUDA/DirectX).
		if (mDeviceInfo.outDeviceFramework != PrGPUDeviceFramework_Metal)
			return suiteError_Fail;

		if (sPipelineCache[mDeviceIndex][0])
		{
			for (int k = 0; k < kKernelCount; ++k)
				mPipelines[k] = sPipelineCache[mDeviceIndex][k];
			return suiteError_NoError;
		}

		ScopedAutoreleasePool pool;
		prSuiteError result = suiteError_NoError;

		NSString* source = [NSString stringWithCString:kAnonymizerMetalString encoding:NSUTF8StringEncoding];
		NSError* error = nil;
		id<MTLDevice> device = (id<MTLDevice>)mDeviceInfo.outDeviceHandle;
		// Precise math so cell/pixel selection matches the CPU path exactly;
		// fast-math's approximate division flips mosaic cells at boundaries.
		MTLCompileOptions* options = [[[MTLCompileOptions alloc] init] autorelease];
		options.mathMode = MTLMathModeSafe;
		id<MTLLibrary> library = [[device newLibraryWithSource:source options:options error:&error] autorelease];
		result = CheckForMetalError(error);
		if (result != suiteError_NoError)
			return result;

		for (int k = 0; k < kKernelCount; ++k)
		{
			NSString* name = [NSString stringWithCString:KernelName(k) encoding:NSUTF8StringEncoding];
			id<MTLFunction> function = [[library newFunctionWithName:name] autorelease];
			if (!function)
				return suiteError_Fail;
			mPipelines[k] = [device newComputePipelineStateWithFunction:function error:&error];
			result = CheckForMetalError(error);
			if (result != suiteError_NoError)
				return result;
			sPipelineCache[mDeviceIndex][k] = mPipelines[k];
		}
		return suiteError_NoError;
	}

	prSuiteError Render(
		const PrGPUFilterRenderParams* inRenderParams,
		const PPixHand* inFrames,
		csSDK_size_t inFrameCount,
		PPixHand* outFrame)
	{
		if (!outFrame || !*outFrame)
			return suiteError_Fail;

		// For effects (unlike transitions) the host delivers *outFrame
		// pre-filled with the source pixels, and the SDK allows rendering in
		// place (see SDK_ProcAmp_GPU / Vignette_GPU). Operating on *outFrame
		// directly preserves the frame's bounds and placement metadata.
		// Replacing it with a CreateGPUPPix frame resets the bounds origin to
		// (0,0), which shifted clips whose aspect ratio doesn't match the
		// sequence. Reads and writes never alias: the multi-pass chain reads
		// the frame only in the first pass and writes it only in the last,
		// and blackout touches each pixel exactly once.
		PPixHand ioPPix = *outFrame;

		// ---- Frame geometry ----
		PrPixelFormat pixelFormat = PrPixelFormat_Invalid;
		mPPixSuite->GetPixelFormat(ioPPix, &pixelFormat);

		prRect bounds = {};
		mPPixSuite->GetBounds(ioPPix, &bounds);
		const int width = bounds.right - bounds.left;
		const int height = bounds.bottom - bounds.top;
		if (width <= 0 || height <= 0)
			return suiteError_Fail;

		const int bytesPerPixel = GetGPUBytesPerPixel(pixelFormat);
		const int is16f = pixelFormat != PrPixelFormat_GPU_BGRA_4444_32f;

		csSDK_int32 rowBytes = 0;
		mPPixSuite->GetRowBytes(ioPPix, &rowBytes);
		const int srcPitch = rowBytes / bytesPerPixel;
		const int dstPitch = srcPitch;

		void* frameData = 0;
		mGPUDeviceSuite->GetGPUPPixData(ioPPix, &frameData);
		if (!frameData)
			return suiteError_Fail;
		void* srcFrameData = frameData;
		void* dstFrameData = frameData;

		// ---- Parameters ----
		const PrTime clipTime = inRenderParams->inClipTime;

		AnonAlgo::UserParams user;
		user.distortAmount = GetParam(ANON_DISTORT_AMOUNT, clipTime).mFloat64;
		user.distortScale = GetParam(ANON_DISTORT_SCALE, clipTime).mFloat64;
		user.blurRadius = GetParam(ANON_BLUR_RADIUS, clipTime).mFloat64;
		user.mosaicSize = GetParam(ANON_MOSAIC_SIZE, clipTime).mFloat64;
		user.enableDistort = GetParam(ANON_ENABLE_DISTORT, clipTime).mBool != 0;
		user.enableBlur = GetParam(ANON_ENABLE_BLUR, clipTime).mBool != 0;
		user.enableMosaic = GetParam(ANON_ENABLE_MOSAIC, clipTime).mBool != 0;
		// Unlike the AE-style CPU side (u.pd.value, 1-based), the video
		// segment suite delivers popup values 0-based.
		user.mosaicShape = (int)GetParam(ANON_MOSAIC_SHAPE, clipTime).mInt32;
		user.blurAfterMosaic = GetParam(ANON_BLUR_AFTER_MOSAIC, clipTime).mBool != 0;
		user.desaturate = GetParam(ANON_DESATURATE, clipTime).mFloat64;
		user.cellNoise = GetParam(ANON_CELL_NOISE, clipTime).mFloat64;
		user.colorNoise = GetParam(ANON_COLOR_NOISE, clipTime).mFloat64;
		user.frostedGlass = GetParam(ANON_FROSTED_GLASS, clipTime).mBool != 0;
		user.frostAmount = GetParam(ANON_FROST_AMOUNT, clipTime).mFloat64;
		UnpackColorParam(GetParam(ANON_FROST_COLOR, clipTime), user.frostColor);
		const bool blackout = GetParam(ANON_BLACKOUT, clipTime).mBool != 0;

		int32_t frame = 0;
		if (inRenderParams->inRenderTicksPerFrame != 0)
			frame = (int32_t)(clipTime / inRenderParams->inRenderTicksPerFrame);
		const double seedParam = GetParam(ANON_SEED, clipTime).mFloat64;
		const uint32_t key = GetParam(ANON_SESSION_KEY, clipTime).mBool != 0
			? AnonAlgo::SessionKey() : 0u;
		const AnonAlgo::Warp warp = AnonAlgo::ComputeWarp(seedParam,
			GetParam(ANON_TEMPORAL_JITTER, clipTime).mBool != 0,
			GetParam(ANON_SMOOTH_JITTER, clipTime).mBool != 0, frame, key);

		// Pixel-space parameters are relative to 1080p; the shorter frame
		// dimension already reflects any preview downsampling and is
		// orientation-invariant (portrait vs landscape timelines).
		const AnonAlgo::LayerSettings settings = AnonAlgo::MakeLayerSettings(user,
			AnonResolutionScale(width, height), warp, AnonAlgo::ComputeCellSeed(seedParam, key),
			ANON_ORDER_BGRA);

		// Blackout short-circuits the whole stack: one pass, no temp buffers.
		if (blackout)
		{
			ScopedAutoreleasePool blackoutPool;
			id<MTLCommandQueue> queue = (id<MTLCommandQueue>)mDeviceInfo.outCommandQueueHandle;
			id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
			id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

			AnonParamsHost blackoutParams = {};
			blackoutParams.m16f = is16f;
			blackoutParams.mWidth = width;
			blackoutParams.mHeight = height;
			blackoutParams.mSrcPitch = srcPitch;
			blackoutParams.mDstPitch = dstPitch;
			Dispatch(encoder, mPipelines[kKernelBlackout],
				(id<MTLBuffer>)srcFrameData, (id<MTLBuffer>)dstFrameData, blackoutParams);

			[encoder endEncoding];
			[commandBuffer commit];
			return suiteError_NoError;
		}

		// ---- Encode the passes ----
		ScopedAutoreleasePool pool;

		id<MTLCommandQueue> queue = (id<MTLCommandQueue>)mDeviceInfo.outCommandQueueHandle;
		id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
		id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

		id<MTLBuffer> srcBuffer = (id<MTLBuffer>)srcFrameData;
		id<MTLBuffer> dstBuffer = (id<MTLBuffer>)dstFrameData;

		// Temp buffers (packed, pitch == width) are allocated per render and
		// released by the command buffer's completion handler. Renders are
		// committed asynchronously and may overlap or vary in size (preview
		// vs full resolution), so instance-cached buffers would risk being
		// freed while an earlier command buffer still reads them.
		const size_t tmpBytes = (size_t)width * height * bytesPerPixel;
		id<MTLDevice> device = (id<MTLDevice>)mDeviceInfo.outDeviceHandle;
		id<MTLBuffer> tmpA = [device newBufferWithLength:tmpBytes options:MTLResourceStorageModePrivate];
		id<MTLBuffer> tmpB = [device newBufferWithLength:tmpBytes options:MTLResourceStorageModePrivate];
		if (tmpA == nil || tmpB == nil)
		{
			[tmpA release];
			[tmpB release];
			[encoder endEncoding];
			return suiteError_OutOfMemory;
		}

		AnonEncodeLayeredPasses(
			[&](int kernel, id<MTLBuffer> in, id<MTLBuffer> out, id<MTLBuffer> aux,
				const AnonParamsHost& params) {
				Dispatch(encoder, mPipelines[kernel], in, out, params, aux);
			},
			settings, width, height, is16f, srcBuffer, srcPitch, dstBuffer, dstPitch, tmpA, tmpB);

		[encoder endEncoding];
		[commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> cb) {
			[tmpA release];
			[tmpB release];
		}];
		[commandBuffer commit];

		return suiteError_NoError;
	}

private:
	void Dispatch(
		id<MTLComputeCommandEncoder> inEncoder,
		id<MTLComputePipelineState> inPipeline,
		id<MTLBuffer> inSrc,
		id<MTLBuffer> inDst,
		const AnonParamsHost& inParams,
		id<MTLBuffer> inAux = nil)
	{
		[inEncoder setComputePipelineState:inPipeline];
		[inEncoder setBuffer:inSrc offset:0 atIndex:0];
		[inEncoder setBuffer:inDst offset:0 atIndex:1];
		[inEncoder setBytes:&inParams length:sizeof(inParams) atIndex:2];
		[inEncoder setBuffer:(inAux ? inAux : inSrc) offset:0 atIndex:3];
		MTLSize threadsPerGroup = {16, 16, 1};
		MTLSize numThreadgroups = {
			DivideRoundUp(inParams.mWidth, threadsPerGroup.width),
			DivideRoundUp(inParams.mHeight, threadsPerGroup.height),
			1};
		[inEncoder dispatchThreadgroups:numThreadgroups threadsPerThreadgroup:threadsPerGroup];
	}


	id<MTLComputePipelineState> mPipelines[kKernelCount];
};


DECLARE_GPUFILTER_ENTRY(PrGPUFilterModule<Anonymizer>)
