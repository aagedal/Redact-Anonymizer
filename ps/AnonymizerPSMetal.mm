/*
** Redact Anonymizer - Photoshop filter plugin (Metal render path).
**
** Runs the full-resolution render on the GPU with the exact MSL kernels
** shared with the Premiere / Resolve / FCP plugins. Unlike those hosts,
** Photoshop gives us no device or queue, so the plugin owns both; the
** packed RGBA float buffer is copied in and out of shared MTLBuffers.
** Any failure returns false and the caller falls back to the CPU path,
** which produces identical output (verified to ~1e-7 by the parity tests).
*/

#include "AnonymizerPS.h"

#import <Metal/Metal.h>

#include "AnonymizerKernel.h"
#include "AnonymizerAlgo.h"

#include <algorithm>
#include <string.h>

static id<MTLDevice> sDevice = nil;
static id<MTLCommandQueue> sQueue = nil;
static id<MTLComputePipelineState> sPipelines[ANON_KERNEL_LAYER_COUNT] = {};
static bool sInitTried = false;
static bool sInitOK = false;

static bool EnsurePipelines(void)
{
	if (sInitTried)
		return sInitOK;
	sInitTried = true;

	sDevice = MTLCreateSystemDefaultDevice();
	if (sDevice == nil)
		return false;
	sQueue = [sDevice newCommandQueue];
	if (sQueue == nil)
		return false;

	NSError* err = nil;
	MTLCompileOptions* options = [MTLCompileOptions new];
	/* Precise math so cell/pixel selection matches the CPU path exactly. */
	options.mathMode = MTLMathModeSafe;
	id<MTLLibrary> library = [sDevice newLibraryWithSource:@(kAnonymizerMetalString)
		options:options error:&err];
	if (library == nil)
		return false;

	for (int k = 0; k < ANON_KERNEL_LAYER_COUNT; ++k)
	{
		id<MTLFunction> function = [library newFunctionWithName:
			[NSString stringWithUTF8String:kAnonLayerKernelNames[k]]];
		if (function == nil)
			return false;
		sPipelines[k] = [sDevice newComputePipelineStateWithFunction:function error:&err];
		if (sPipelines[k] == nil)
			return false;
	}

	sInitOK = true;
	return true;
}

static void DispatchPass(
	id<MTLComputeCommandEncoder> encoder,
	id<MTLComputePipelineState> pipeline,
	id<MTLBuffer> src,
	id<MTLBuffer> dst,
	const AnonParamsHost& params,
	id<MTLBuffer> aux = nil)
{
	[encoder setComputePipelineState:pipeline];
	[encoder setBuffer:src offset:0 atIndex:0];
	[encoder setBuffer:dst offset:0 atIndex:1];
	[encoder setBytes:&params length:sizeof(params) atIndex:2];
	[encoder setBuffer:(aux ? aux : src) offset:0 atIndex:3];
	MTLSize threadsPerGroup = {16, 16, 1};
	MTLSize numThreadgroups = {
		(NSUInteger)((params.mWidth + 15) / 16),
		(NSUInteger)((params.mHeight + 15) / 16),
		1};
	[encoder dispatchThreadgroups:numThreadgroups threadsPerThreadgroup:threadsPerGroup];
}

bool RunMetalPasses(float* buf, int w, int h, const AnonAlgo::LayerSettings& settings)
{
	@autoreleasepool
	{
		if (!EnsurePipelines())
			return false;

		const size_t bytes = (size_t)w * h * 4 * sizeof(float);
		id<MTLBuffer> src = [sDevice newBufferWithBytes:buf length:bytes
			options:MTLResourceStorageModeShared];
		id<MTLBuffer> dst = [sDevice newBufferWithLength:bytes
			options:MTLResourceStorageModeShared];
		id<MTLBuffer> tmpA = [sDevice newBufferWithLength:bytes
			options:MTLResourceStorageModePrivate];
		id<MTLBuffer> tmpB = [sDevice newBufferWithLength:bytes
			options:MTLResourceStorageModePrivate];
		if (src == nil || dst == nil || tmpA == nil || tmpB == nil)
			return false;

		id<MTLCommandBuffer> commandBuffer = [sQueue commandBuffer];
		commandBuffer.label = @"MultiLayerAnonymizerPS";
		id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

		AnonEncodeLayeredPasses(
			[&](int kernel, id<MTLBuffer> in, id<MTLBuffer> out, id<MTLBuffer> aux,
			const AnonParamsHost& params) {
				DispatchPass(encoder, sPipelines[kernel], in, out, params, aux);
			},
			settings, w, h, 0, src, w, dst, w, tmpA, tmpB);

		[encoder endEncoding];
		[commandBuffer commit];
		[commandBuffer waitUntilCompleted];
		if (commandBuffer.error != nil)
			return false;

		memcpy(buf, dst.contents, bytes);
		return true;
	}
}
