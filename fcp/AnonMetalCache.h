/*
** Redact Anonymizer FxPlug - per-device Metal pipeline cache.
*/

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

@interface AnonMetalCacheEntry : NSObject
@property (strong) id<MTLDevice> device;
@property (strong) id<MTLCommandQueue> queue;
@property (strong) id<MTLComputePipelineState> psoCopyIn;
@property (strong) id<MTLComputePipelineState> psoBlackout;
// Pipeline for a layered-pass kernel (ANON_KERNEL_* in AnonymizerKernel.h).
- (id<MTLComputePipelineState>)layerPipeline:(int)kernel;
- (id<MTLRenderPipelineState>)renderPipelineForPixelFormat:(MTLPixelFormat)format;
@end

@interface AnonMetalCache : NSObject
+ (instancetype)sharedCache;
- (AnonMetalCacheEntry*)entryForDeviceRegistryID:(uint64_t)registryID;
@end
