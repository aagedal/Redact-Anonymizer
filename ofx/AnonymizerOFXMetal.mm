/*
** Redact Anonymizer OFX - Metal host code.
**
** Reuses the exact same MSL kernels as the Premiere Pro plugin
** (shared/AnonymizerKernel.h). Resolve hands the plugin an MTLCommandQueue
** and images whose pixel pointers are really id<MTLBuffer> (see Blackmagic's
** GainPlugin sample). OFX GPU images are packed RGBA float, pitch == width;
** alpha is component .w in both RGBA and BGRA, so the kernels are reused
** unmodified with is16f = 0.
*/

#import <Metal/Metal.h>

#include <unordered_map>
#include <mutex>
#include <math.h>
#include <algorithm>
#include <stdint.h>

#include "AnonymizerKernel.h"
#include "AnonymizerAlgo.h"

// The layered-pass kernels (ANON_KERNEL_*), then blackout.
enum { kKernelBlackout = ANON_KERNEL_LAYER_COUNT, kKernelCount };

static const char* KernelName(int k)
{
	return k == kKernelBlackout ? "AnonBlackout" : kAnonLayerKernelNames[k];
}

struct AnonPipelines
{
	id<MTLComputePipelineState> p[kKernelCount];
};

static std::mutex s_PipelineQueueMutex;
static std::unordered_map<id<MTLCommandQueue>, AnonPipelines> s_PipelineQueueMap;

static bool GetPipelines(id<MTLCommandQueue> p_Queue, AnonPipelines& outPipelines)
{
	std::unique_lock<std::mutex> lock(s_PipelineQueueMutex);

	const auto it = s_PipelineQueueMap.find(p_Queue);
	if (it != s_PipelineQueueMap.end())
	{
		outPipelines = it->second;
		return true;
	}

	id<MTLDevice> device = p_Queue.device;
	NSError* err = nil;

	MTLCompileOptions* options = [MTLCompileOptions new];
	// Precise math so cell/pixel selection matches the CPU path exactly.
	options.mathMode = MTLMathModeSafe;
	id<MTLLibrary> library = [device newLibraryWithSource:@(kAnonymizerMetalString) options:options error:&err];
	[options release];
	if (!library)
	{
		fprintf(stderr, "Anonymizer OFX: failed to compile Metal library: %s\n",
			err.localizedDescription.UTF8String);
		return false;
	}

	AnonPipelines pipelines = {};
	for (int k = 0; k < kKernelCount; ++k)
	{
		id<MTLFunction> function = [library newFunctionWithName:
			[NSString stringWithUTF8String:KernelName(k)]];
		if (!function)
		{
			[library release];
			return false;
		}
		pipelines.p[k] = [device newComputePipelineStateWithFunction:function error:&err];
		[function release];
		if (!pipelines.p[k])
		{
			fprintf(stderr, "Anonymizer OFX: failed to create pipeline %s: %s\n",
				KernelName(k), err.localizedDescription.UTF8String);
			[library release];
			return false;
		}
	}
	[library release];

	s_PipelineQueueMap[p_Queue] = pipelines;
	outPipelines = pipelines;
	return true;
}

static void DispatchPass(
	id<MTLComputeCommandEncoder> p_Encoder,
	id<MTLComputePipelineState> p_Pipeline,
	id<MTLBuffer> p_Src,
	id<MTLBuffer> p_Dst,
	const AnonParamsHost& p_Params,
	id<MTLBuffer> p_Aux = nil)
{
	[p_Encoder setComputePipelineState:p_Pipeline];
	[p_Encoder setBuffer:p_Src offset:0 atIndex:0];
	[p_Encoder setBuffer:p_Dst offset:0 atIndex:1];
	[p_Encoder setBytes:&p_Params length:sizeof(p_Params) atIndex:2];
	[p_Encoder setBuffer:(p_Aux ? p_Aux : p_Src) offset:0 atIndex:3];
	MTLSize threadsPerGroup = {16, 16, 1};
	MTLSize numThreadgroups = {
		(NSUInteger)((p_Params.mWidth + 15) / 16),
		(NSUInteger)((p_Params.mHeight + 15) / 16),
		1};
	[p_Encoder dispatchThreadgroups:numThreadgroups threadsPerThreadgroup:threadsPerGroup];
}

void RunMetalAnonymizer(void* p_CmdQ, int p_Width, int p_Height,
	const float* p_Input, float* p_Output, const AnonAlgo::LayerSettings& p_Settings, bool p_Blackout)
{
	// Plugins must not rely on a host autorelease pool.
	@autoreleasepool {

	id<MTLCommandQueue> queue = static_cast<id<MTLCommandQueue>>(p_CmdQ);
	id<MTLDevice> device = queue.device;

	AnonPipelines pipelines;
	if (!GetPipelines(queue, pipelines))
		return;

	id<MTLBuffer> srcBuffer = reinterpret_cast<id<MTLBuffer>>(const_cast<float*>(p_Input));
	id<MTLBuffer> dstBuffer = reinterpret_cast<id<MTLBuffer>>(p_Output);

	id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
	commandBuffer.label = @"MultiLayerAnonymizer";
	id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

	if (p_Blackout)
	{
		AnonParamsHost params = {};
		params.mSrcPitch = p_Width;
		params.mDstPitch = p_Width;
		params.mWidth = p_Width;
		params.mHeight = p_Height;
		DispatchPass(encoder, pipelines.p[kKernelBlackout], srcBuffer, dstBuffer, params);
		[encoder endEncoding];
		[commandBuffer commit];
		return;
	}

	// Two packed temp buffers for the pass chain; result in dst.
	const size_t tmpBytes = (size_t)p_Width * p_Height * 4 * sizeof(float);
	id<MTLBuffer> tmpA = [device newBufferWithLength:tmpBytes options:MTLResourceStorageModePrivate];
	id<MTLBuffer> tmpB = [device newBufferWithLength:tmpBytes options:MTLResourceStorageModePrivate];

	AnonEncodeLayeredPasses(
		[&](int kernel, id<MTLBuffer> in, id<MTLBuffer> out, id<MTLBuffer> aux,
			const AnonParamsHost& params) {
			DispatchPass(encoder, pipelines.p[kernel], in, out, params, aux);
		},
		p_Settings, p_Width, p_Height, 0, srcBuffer, p_Width, dstBuffer, p_Width, tmpA, tmpB);

	[encoder endEncoding];
	[commandBuffer addCompletedHandler:^(id<MTLCommandBuffer>) {
		[tmpA release];
		[tmpB release];
	}];
	[commandBuffer commit];

	} // @autoreleasepool
}
