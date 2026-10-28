/*
** Redact Anonymizer - OpenFX plugin for DaVinci Resolve (and other OFX
** hosts). Same algorithm as the Premiere Pro plugin: value-noise distortion
** -> Gaussian blur -> mosaic, plus a Blackout override.
**
** Structure follows Blackmagic's GainPlugin OFX sample (bundled with
** Resolve under Developer/OpenFX). GPU rendering is Metal on macOS; other
** frameworks fall back to the CPU path, which produces identical output.
*/

#include <stdio.h>
#include <mutex>
#include <memory>
#include <vector>

#include "ofxsImageEffect.h"
#include "ofxsMultiThread.h"
#include "ofxsProcessing.h"
#include "ofxsLog.h"

#include "AnonymizerAlgo.h"

/* Branding macros are injected by the build (see ANON_EDITION in
** CMakeLists.txt); the fallbacks are the open-source edition. */
#ifndef ANON_EFFECT_NAME
#define ANON_EFFECT_NAME "Redact Anonymizer"
#endif
#ifndef ANON_CATEGORY
#define ANON_CATEGORY "Aagedal"
#endif
#ifndef ANON_OFX_PLUGIN_ID
#define ANON_OFX_PLUGIN_ID "me.aagedal.ofx.MultiLayerAnonymizer"
#endif

#define kPluginName ANON_EFFECT_NAME
#define kPluginGrouping ANON_CATEGORY
#define kPluginDescription \
	"Anonymizes a region with three stacked layers - random distortion, " \
	"Gaussian blur and mosaic - so the result cannot be reversed by " \
	"deblurring or mosaic-reconstruction tools. Desaturate optionally " \
	"removes colour; Frosted Glass softens the result into a frosted-glass " \
	"look; Blackout replaces the stack with solid black."
#define kPluginIdentifier ANON_OFX_PLUGIN_ID
#define kPluginVersionMajor MAJOR_VERSION
#define kPluginVersionMinor MINOR_VERSION

#define kSupportsTiles false
#define kSupportsMultiResolution false
#define kSupportsMultipleClipPARs false

////////////////////////////////////////////////////////////////////////////////

class AnonymizerProcessor : public OFX::ImageProcessor
{
public:
	explicit AnonymizerProcessor(OFX::ImageEffect& p_Instance)
		: OFX::ImageProcessor(p_Instance)
		, _srcImg(0)
	{
	}

	virtual void processImagesMetal();
	virtual void multiThreadProcessImages(OfxRectI p_ProcWindow);

	void setSrcImg(OFX::Image* p_SrcImg) { _srcImg = p_SrcImg; }
	void setSettings(const AnonAlgo::LayerSettings& p_Settings, bool p_Blackout)
	{
		_settings = p_Settings;
		_blackout = p_Blackout;
	}

private:
	void computeFullCPU();

	OFX::Image* _srcImg;
	AnonAlgo::LayerSettings _settings;
	bool _blackout = false;

	// CPU path: the layered passes need the whole frame, so the full result
	// is computed once and the per-thread windows just copy from it.
	std::vector<float> _cpuResult;
	std::once_flag _cpuComputed;
};

#ifdef __APPLE__
extern void RunMetalAnonymizer(void* p_CmdQ, int p_Width, int p_Height,
	const float* p_Input, float* p_Output, const AnonAlgo::LayerSettings& p_Settings, bool p_Blackout);
#endif

void AnonymizerProcessor::processImagesMetal()
{
#ifdef __APPLE__
	const OfxRectI& bounds = _srcImg->getBounds();
	const int width = bounds.x2 - bounds.x1;
	const int height = bounds.y2 - bounds.y1;

	const float* input = static_cast<const float*>(_srcImg->getPixelData());
	float* output = static_cast<float*>(_dstImg->getPixelData());

	RunMetalAnonymizer(_pMetalCmdQ, width, height, input, output, _settings, _blackout);
#endif
}

void AnonymizerProcessor::computeFullCPU()
{
	const OfxRectI dstBounds = _dstImg->getBounds();
	const int w = dstBounds.x2 - dstBounds.x1;
	const int h = dstBounds.y2 - dstBounds.y1;

	_cpuResult.resize((size_t)w * h * 4);

	// Gather the source into a packed buffer over the dst bounds. Rows are
	// walked bottom-up or top-down as the host laid them out - the passes
	// are orientation-agnostic, only consistency matters.
	for (int y = 0; y < h; ++y)
	{
		float* out = &_cpuResult[(size_t)y * w * 4];
		for (int x = 0; x < w; ++x, out += 4)
		{
			float* srcPix = static_cast<float*>(_srcImg
				? _srcImg->getPixelAddress(dstBounds.x1 + x, dstBounds.y1 + y) : 0);
			if (srcPix)
			{
				out[0] = srcPix[0];
				out[1] = srcPix[1];
				out[2] = srcPix[2];
				out[3] = srcPix[3];
			}
			else
			{
				out[0] = out[1] = out[2] = out[3] = 0.0f;
			}
		}
	}

	if (_blackout)
	{
		// RGBA: zero RGB, keep alpha.
		for (size_t i = 0; i < _cpuResult.size(); i += 4)
			_cpuResult[i] = _cpuResult[i + 1] = _cpuResult[i + 2] = 0.0f;
		return;
	}

	std::vector<float> scratch((size_t)w * h * 4);
	AnonAlgo::RunLayeredPasses(_cpuResult.data(), scratch.data(), w, h, _settings);
}

void AnonymizerProcessor::multiThreadProcessImages(OfxRectI p_ProcWindow)
{
	std::call_once(_cpuComputed, [this] { computeFullCPU(); });

	const OfxRectI dstBounds = _dstImg->getBounds();
	const int w = dstBounds.x2 - dstBounds.x1;

	for (int y = p_ProcWindow.y1; y < p_ProcWindow.y2; ++y)
	{
		if (_effect.abort())
			break;

		float* dstPix = static_cast<float*>(_dstImg->getPixelAddress(p_ProcWindow.x1, y));
		const float* resPix = &_cpuResult[
			((size_t)(y - dstBounds.y1) * w + (p_ProcWindow.x1 - dstBounds.x1)) * 4];
		memcpy(dstPix, resPix, (size_t)(p_ProcWindow.x2 - p_ProcWindow.x1) * 4 * sizeof(float));
	}
}

////////////////////////////////////////////////////////////////////////////////

class AnonymizerPlugin : public OFX::ImageEffect
{
public:
	explicit AnonymizerPlugin(OfxImageEffectHandle p_Handle)
		: ImageEffect(p_Handle)
	{
		m_DstClip = fetchClip(kOfxImageEffectOutputClipName);
		m_SrcClip = fetchClip(kOfxImageEffectSimpleSourceClipName);

		m_DistortAmount = fetchDoubleParam("distortAmount");
		m_DistortScale = fetchDoubleParam("distortScale");
		m_BlurRadius = fetchDoubleParam("blurRadius");
		m_MosaicSize = fetchDoubleParam("mosaicSize");
		m_Seed = fetchIntParam("seed");
		m_TemporalJitter = fetchBooleanParam("temporalJitter");
		m_Blackout = fetchBooleanParam("blackout");
		m_MosaicShape = fetchChoiceParam("mosaicShape");
		m_BlurAfterMosaic = fetchBooleanParam("blurAfterMosaic");
		m_Desaturate = fetchDoubleParam("desaturate");
		m_CellNoise = fetchDoubleParam("cellNoise");
		m_ColorNoise = fetchDoubleParam("colorNoise");
		m_EnableDistort = fetchBooleanParam("enableDistort");
		m_EnableBlur = fetchBooleanParam("enableBlur");
		m_EnableMosaic = fetchBooleanParam("enableMosaic");
		m_SessionKey = fetchBooleanParam("sessionKey");
		m_SmoothJitter = fetchBooleanParam("smoothJitter");
		m_FrostedGlass = fetchBooleanParam("frostedGlass");
		m_FrostAmount = fetchDoubleParam("frostAmount");
		m_FrostColor = fetchRGBParam("frostColor");
	}

	virtual void render(const OFX::RenderArguments& p_Args)
	{
		if ((m_DstClip->getPixelDepth() == OFX::eBitDepthFloat)
			&& (m_DstClip->getPixelComponents() == OFX::ePixelComponentRGBA))
		{
			AnonymizerProcessor processor(*this);
			setupAndProcess(processor, p_Args);
		}
		else
		{
			OFX::throwSuiteStatusException(kOfxStatErrUnsupported);
		}
	}

	void changedParam(const OFX::InstanceChangedArgs& args, const std::string& name) override
	{
		if (args.reason != OFX::eChangeUserEdit)
			return;
		if (name == "randomizeSeed")
		{
			m_Seed->setValue(AnonAlgo::RandomSeedValue());
			return;
		}
		if (name != "blurAfterMosaic")
			return;
		const double current = m_BlurRadius->getValueAtTime(args.time);
		const double radius = AnonAlgo::BlurRadiusForOrderChange(current,
			m_BlurAfterMosaic->getValueAtTime(args.time));
		if (radius != current)
		{
			if (m_BlurRadius->getNumKeys() == 0)
				m_BlurRadius->setValue(radius);
			else
				m_BlurRadius->setValueAtTime(args.time, radius);
		}
	}

	// Do not claim identity from unscaled controls: a 1px mosaic at the
	// 1080p reference becomes 2px at 4K. The render pipeline handles
	// disabled stages after scaling to the actual image bounds.

	void setupAndProcess(AnonymizerProcessor& p_Processor, const OFX::RenderArguments& p_Args)
	{
		std::unique_ptr<OFX::Image> dst(m_DstClip->fetchImage(p_Args.time));
		std::unique_ptr<OFX::Image> src(m_SrcClip->fetchImage(p_Args.time));

		if ((src->getPixelDepth() != dst->getPixelDepth())
			|| (src->getPixelComponents() != dst->getPixelComponents()))
		{
			OFX::throwSuiteStatusException(kOfxStatErrValue);
		}

		const double t = p_Args.time;
		AnonAlgo::UserParams user;
		user.distortAmount = m_DistortAmount->getValueAtTime(t);
		user.distortScale = m_DistortScale->getValueAtTime(t);
		user.blurRadius = m_BlurRadius->getValueAtTime(t);
		user.mosaicSize = m_MosaicSize->getValueAtTime(t);
		user.enableDistort = m_EnableDistort->getValueAtTime(t);
		user.enableBlur = m_EnableBlur->getValueAtTime(t);
		user.enableMosaic = m_EnableMosaic->getValueAtTime(t);
		m_MosaicShape->getValueAtTime(t, user.mosaicShape);
		user.blurAfterMosaic = m_BlurAfterMosaic->getValueAtTime(t);
		user.desaturate = m_Desaturate->getValueAtTime(t);
		user.cellNoise = m_CellNoise->getValueAtTime(t);
		user.colorNoise = m_ColorNoise->getValueAtTime(t);
		user.frostedGlass = m_FrostedGlass->getValueAtTime(t);
		user.frostAmount = m_FrostAmount->getValueAtTime(t);
		m_FrostColor->getValueAtTime(t, user.frostColor[0], user.frostColor[1], user.frostColor[2]);

		// OFX time is in frames.
		const int32_t frame = (int32_t)(t + 0.5);
		const double seedParam = (double)m_Seed->getValueAtTime(t);
		const uint32_t key = m_SessionKey->getValueAtTime(t) ? AnonAlgo::SessionKey() : 0u;
		const AnonAlgo::Warp warp = AnonAlgo::ComputeWarp(seedParam,
			m_TemporalJitter->getValueAtTime(t), m_SmoothJitter->getValueAtTime(t), frame, key);

		// Pixel-space parameters are relative to 1080p; the shorter frame
		// dimension reflects proxy/preview scaling and is orientation-invariant
		// (portrait vs landscape timelines).
		const OfxRectI dstBounds = dst->getBounds();
		AnonAlgo::LayerSettings settings = AnonAlgo::MakeLayerSettings(user,
			AnonResolutionScale(dstBounds.x2 - dstBounds.x1, dstBounds.y2 - dstBounds.y1),
			warp, AnonAlgo::ComputeCellSeed(seedParam, key), ANON_ORDER_RGBA);
		// Resolve hands a power window over as premultiplied alpha.
		settings.premultiplied = src->getPreMultiplication() == OFX::eImagePreMultiplied;

		p_Processor.setDstImg(dst.get());
		p_Processor.setSrcImg(src.get());
		p_Processor.setGPURenderArgs(p_Args);
		p_Processor.setRenderWindow(p_Args.renderWindow);
		p_Processor.setSettings(settings, m_Blackout->getValueAtTime(t));
		p_Processor.process();
	}

	OFX::Clip* m_DstClip;
	OFX::Clip* m_SrcClip;

	OFX::DoubleParam* m_DistortAmount;
	OFX::DoubleParam* m_DistortScale;
	OFX::DoubleParam* m_BlurRadius;
	OFX::DoubleParam* m_MosaicSize;
	OFX::IntParam* m_Seed;
	OFX::BooleanParam* m_TemporalJitter;
	OFX::BooleanParam* m_Blackout;
	OFX::ChoiceParam* m_MosaicShape;
	OFX::BooleanParam* m_BlurAfterMosaic;
	OFX::DoubleParam* m_Desaturate;
	OFX::DoubleParam* m_CellNoise;
	OFX::DoubleParam* m_ColorNoise;
	OFX::BooleanParam* m_EnableDistort;
	OFX::BooleanParam* m_EnableBlur;
	OFX::BooleanParam* m_EnableMosaic;
	OFX::BooleanParam* m_SessionKey;
	OFX::BooleanParam* m_SmoothJitter;
	OFX::BooleanParam* m_FrostedGlass;
	OFX::DoubleParam* m_FrostAmount;
	OFX::RGBParam* m_FrostColor;
};

////////////////////////////////////////////////////////////////////////////////

using namespace OFX;

class AnonymizerPluginFactory : public OFX::PluginFactoryHelper<AnonymizerPluginFactory>
{
public:
	AnonymizerPluginFactory()
		: OFX::PluginFactoryHelper<AnonymizerPluginFactory>(
			kPluginIdentifier, kPluginVersionMajor, kPluginVersionMinor)
	{
	}

	virtual void load() {}
	virtual void unload() {}

	virtual void describe(OFX::ImageEffectDescriptor& p_Desc)
	{
		p_Desc.setLabels(kPluginName, kPluginName, kPluginName);
		p_Desc.setPluginGrouping(kPluginGrouping);
		p_Desc.setPluginDescription(kPluginDescription);

		p_Desc.addSupportedContext(eContextFilter);
		p_Desc.addSupportedContext(eContextGeneral);
		p_Desc.addSupportedBitDepth(eBitDepthFloat);

		p_Desc.setSingleInstance(false);
		p_Desc.setHostFrameThreading(false);
		p_Desc.setSupportsMultiResolution(kSupportsMultiResolution);
		p_Desc.setSupportsTiles(kSupportsTiles);
		p_Desc.setTemporalClipAccess(false);
		p_Desc.setRenderTwiceAlways(false);
		p_Desc.setSupportsMultipleClipPARs(kSupportsMultipleClipPARs);

#ifdef __APPLE__
		p_Desc.setSupportsMetalRender(true);
#endif
	}

	virtual void describeInContext(OFX::ImageEffectDescriptor& p_Desc, OFX::ContextEnum /*p_Context*/)
	{
		ClipDescriptor* srcClip = p_Desc.defineClip(kOfxImageEffectSimpleSourceClipName);
		srcClip->addSupportedComponent(ePixelComponentRGBA);
		srcClip->setTemporalClipAccess(false);
		srcClip->setSupportsTiles(kSupportsTiles);
		srcClip->setIsMask(false);

		ClipDescriptor* dstClip = p_Desc.defineClip(kOfxImageEffectOutputClipName);
		dstClip->addSupportedComponent(ePixelComponentRGBA);
		dstClip->setSupportsTiles(kSupportsTiles);

		PageParamDescriptor* page = p_Desc.definePageParam("Controls");

		// The strength sliders stop at their minimums; these switch a layer
		// off entirely, for previewing the others.
		page->addChild(*defineEnable(p_Desc, "enableDistort", "Enable Distortion",
			"Turn off to preview the result without the distortion layer"));
		page->addChild(*defineDouble(p_Desc, "distortAmount", "Distortion Amount",
			"Maximum displacement of the random warp field, in pixels",
			DISTORT_AMOUNT_MIN, DISTORT_AMOUNT_MAX, DISTORT_AMOUNT_DFLT));
		page->addChild(*defineDouble(p_Desc, "distortScale", "Distortion Scale",
			"Size of the warp noise features, in pixels",
			DISTORT_SCALE_MIN, DISTORT_SCALE_MAX, DISTORT_SCALE_DFLT));
		page->addChild(*defineEnable(p_Desc, "enableBlur", "Enable Blur",
			"Turn off to preview the result without the blur layer"));
		page->addChild(*defineDouble(p_Desc, "blurRadius", "Blur Radius",
			"Gaussian blur radius, in pixels",
			BLUR_RADIUS_MIN, BLUR_RADIUS_MAX, BLUR_RADIUS_DFLT));
		page->addChild(*defineEnable(p_Desc, "enableMosaic", "Enable Mosaic",
			"Turn off to preview the result without the mosaic layer"));
		page->addChild(*defineDouble(p_Desc, "mosaicSize", "Mosaic Block Size",
			"Pixelation block size, in pixels",
			MOSAIC_SIZE_MIN, MOSAIC_SIZE_MAX, MOSAIC_SIZE_DFLT));

		IntParamDescriptor* seed = p_Desc.defineIntParam("seed");
		seed->setLabels("Random Seed", "Random Seed", "Random Seed");
		seed->setHint("Change for a different distortion pattern");
		seed->setDefault((int)SEED_DFLT);
		seed->setRange((int)SEED_MIN, (int)SEED_MAX);
		seed->setDisplayRange((int)SEED_MIN, (int)SEED_MAX);
		page->addChild(*seed);

		PushButtonParamDescriptor* randomize = p_Desc.definePushButtonParam("randomizeSeed");
		randomize->setLabels("Randomize Seed", "Randomize Seed", "Randomize Seed");
		randomize->setHint("Pick a new random seed");
		page->addChild(*randomize);

		BooleanParamDescriptor* sessionKey = p_Desc.defineBooleanParam("sessionKey");
		sessionKey->setLabels("Session Key", "Session Key", "Session Key");
		sessionKey->setHint("Mix an unrecorded random key, new on every launch, into the distortion "
			"pattern so it cannot be reproduced from the project file. Turn off for renders "
			"that must be identical across launches.");
		sessionKey->setDefault(true);
		page->addChild(*sessionKey);

		BooleanParamDescriptor* jitter = p_Desc.defineBooleanParam("temporalJitter");
		jitter->setLabels("Temporal Jitter", "Temporal Jitter", "Temporal Jitter");
		jitter->setHint("Use a new distortion pattern on every frame");
		jitter->setDefault(true);
		page->addChild(*jitter);

		BooleanParamDescriptor* smooth = p_Desc.defineBooleanParam("smoothJitter");
		smooth->setLabels("Smooth Jitter", "Smooth Jitter", "Smooth Jitter");
		smooth->setHint("With Temporal Jitter: let the distortion pattern flow smoothly from "
			"frame to frame instead of changing abruptly, for a calmer look");
		smooth->setDefault(true);
		page->addChild(*smooth);

		BooleanParamDescriptor* blackout = p_Desc.defineBooleanParam("blackout");
		blackout->setLabels("Blackout", "Blackout", "Blackout");
		blackout->setHint("Solid black instead of the distort/blur/mosaic stack (alpha preserved)");
		blackout->setDefault(false);
		page->addChild(*blackout);

		ChoiceParamDescriptor* shape = p_Desc.defineChoiceParam("mosaicShape");
		shape->setLabels("Mosaic Shape", "Mosaic Shape", "Mosaic Shape");
		shape->setHint("Tiling used by the mosaic layer");
		shape->appendOption("Square");
		shape->appendOption("Triangle");
		shape->appendOption("Hexagon");
		shape->setDefault(ANON_SHAPE_SQUARE);
		page->addChild(*shape);

		page->addChild(*defineDouble(p_Desc, "cellNoise", "Cell Noise",
			"Random brightness offset per mosaic cell, in percent; defeats block-colour matching",
			CELL_NOISE_MIN, CELL_NOISE_MAX, CELL_NOISE_DFLT));
		page->addChild(*defineDouble(p_Desc, "colorNoise", "Colour Noise",
			"Random per-channel offset per mosaic cell, in percent; shifts each cell's hue",
			COLOR_NOISE_MIN, COLOR_NOISE_MAX, COLOR_NOISE_DFLT));

		BooleanParamDescriptor* blurAfter = p_Desc.defineBooleanParam("blurAfterMosaic");
		blurAfter->setLabels("Blur After Mosaic", "Blur After Mosaic", "Blur After Mosaic");
		blurAfter->setHint("Apply blur after mosaic to soften the visible cell edges");
		blurAfter->setDefault(false);
		page->addChild(*blurAfter);
		DoubleParamDescriptor* desat = defineDouble(p_Desc, "desaturate", "Desaturate",
			"Remove colour from the anonymized region, in percent",
			DESATURATE_MIN, DESATURATE_MAX, DESATURATE_DFLT);
		page->addChild(*desat);

		BooleanParamDescriptor* frost = p_Desc.defineBooleanParam("frostedGlass");
		frost->setLabels("Frosted Glass", "Frosted Glass", "Frosted Glass");
		frost->setHint("Finish the anonymized result as frosted glass: softened, lightened and "
			"grained. Changes only the look - the layers above do the anonymizing");
		frost->setDefault(false);
		page->addChild(*frost);
		page->addChild(*defineDouble(p_Desc, "frostAmount", "Frost Amount",
			"How far Frosted Glass lifts the region toward the Frost Colour, in percent",
			FROST_AMOUNT_MIN, FROST_AMOUNT_MAX, FROST_AMOUNT_DFLT));

		RGBParamDescriptor* frostColor = p_Desc.defineRGBParam("frostColor");
		frostColor->setLabels("Frost Colour", "Frost Colour", "Frost Colour");
		frostColor->setHint("Colour the Frosted Glass finish tints the region toward");
		frostColor->setDefault(FROST_COLOR_DFLT, FROST_COLOR_DFLT, FROST_COLOR_DFLT);
		page->addChild(*frostColor);
	}

	virtual ImageEffect* createInstance(OfxImageEffectHandle p_Handle, ContextEnum /*p_Context*/)
	{
		return new AnonymizerPlugin(p_Handle);
	}

private:
	static BooleanParamDescriptor* defineEnable(OFX::ImageEffectDescriptor& p_Desc,
		const std::string& p_Name, const std::string& p_Label, const std::string& p_Hint)
	{
		BooleanParamDescriptor* param = p_Desc.defineBooleanParam(p_Name);
		param->setLabels(p_Label, p_Label, p_Label);
		param->setHint(p_Hint);
		param->setDefault(true);
		return param;
	}

	static DoubleParamDescriptor* defineDouble(OFX::ImageEffectDescriptor& p_Desc,
		const std::string& p_Name, const std::string& p_Label, const std::string& p_Hint,
		double p_Min, double p_Max, double p_Default)
	{
		DoubleParamDescriptor* param = p_Desc.defineDoubleParam(p_Name);
		param->setLabels(p_Label, p_Label, p_Label);
		param->setScriptName(p_Name);
		param->setHint(p_Hint);
		param->setDefault(p_Default);
		param->setRange(p_Min, p_Max);
		param->setIncrement(0.1);
		param->setDisplayRange(p_Min, p_Max);
		return param;
	}
};

void OFX::Plugin::getPluginIDs(PluginFactoryArray& p_FactoryArray)
{
	static AnonymizerPluginFactory factory;
	p_FactoryArray.push_back(&factory);
}
