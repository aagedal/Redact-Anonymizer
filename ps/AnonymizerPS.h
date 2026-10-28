/*
** Redact Anonymizer - Photoshop filter plugin.
**
** Shared declarations between the filter logic (AnonymizerPS.cpp) and the
** Cocoa parameter dialog (AnonymizerPSUI.mm). The pixel pipeline itself
** lives in shared/AnonymizerAlgo.h and is byte-identical to the CPU path
** of the Premiere / Resolve / FCP plugins.
*/

#ifndef ANONYMIZER_PS_H
#define ANONYMIZER_PS_H

#include "PIDefines.h"
#include "PITypes.h"
#include "PIAbout.h"
#include "PIFilter.h"

#include "AnonymizerAlgo.h"

#ifndef ANONYMIZER_NAME
#define ANONYMIZER_NAME "Redact Anonymizer"
#endif

/*
** Persisted in the host-owned parameters handle, so "Last Filter" (Cmd-F)
** reruns with the previous values. Fixed-size scalar fields only; new
** fields go at the end (older, shorter handles are grown and the new
** fields defaulted - see EnsureParametersHandle).
*/
typedef struct PSParameters
{
	double distortAmount;
	double distortScale;
	double blurRadius;
	double mosaicSize;
	int32 mosaicShape; /* ANON_SHAPE_* */
	Boolean blackout;
	int32 blurAfterMosaic;
	double desaturate; /* 0..100 */
	double cellNoise; /* 0..CELL_NOISE_MAX percent */
	double colorNoise; /* 0..COLOR_NOISE_MAX percent */
	int32 enableDistort; /* layer switches, for previewing the others */
	int32 enableBlur;
	int32 enableMosaic;
	int32 frostedGlass;
	double frostAmount; /* 0..100 */
	double frostColor[3]; /* R, G, B, 0..1 */
} PSParameters;

/*
** The shared layer settings for a run. ds is the resolution scale of the
** document (or of the proxy, for the preview); seed is the run's random
** seed; gray documents have no colour to remove.
*/
inline AnonAlgo::LayerSettings PSLayerSettings(const PSParameters& p, float ds, uint32_t seed,
	bool gray)
{
	AnonAlgo::UserParams u;
	u.distortAmount = p.distortAmount;
	u.distortScale = p.distortScale;
	u.blurRadius = p.blurRadius;
	u.mosaicSize = p.mosaicSize;
	u.enableDistort = p.enableDistort != 0;
	u.enableBlur = p.enableBlur != 0;
	u.enableMosaic = p.enableMosaic != 0;
	u.mosaicShape = (int)p.mosaicShape;
	u.blurAfterMosaic = p.blurAfterMosaic != 0;
	u.desaturate = gray ? 0.0 : p.desaturate;
	u.cellNoise = p.cellNoise;
	u.colorNoise = p.colorNoise;
	u.frostedGlass = p.frostedGlass != 0;
	u.frostAmount = p.frostAmount;
	u.frostColor[0] = p.frostColor[0];
	u.frostColor[1] = p.frostColor[1];
	u.frostColor[2] = p.frostColor[2];
	return AnonAlgo::MakeLayerSettings(u, ds, AnonAlgo::StaticWarp(seed),
		AnonAlgo::CellSeedFor(seed), ANON_ORDER_RGBA);
}

/*
** Downscaled copy of the filtered region for the dialog's live preview.
** pixels is packed RGBA float (pitch == width); ds is the resolution scale
** the preview passes must use so the proxy approximates the full-res
** result; colorPlanes is 1 for grayscale documents, else 3.
*/
typedef struct PSPreviewContext
{
	const float* pixels;
	int width;
	int height;
	float ds;
	int colorPlanes;
	uint32_t seed; /* the same seed the full-resolution render will use */
} PSPreviewContext;

/*
** Implemented in AnonymizerPSUI.mm. Returns true if the user confirmed.
** The dialog edits ioParams in place; the caller restores on cancel.
** preview may be NULL (or have NULL pixels) - the dialog then runs
** without the live preview.
*/
bool DoParamDialog(PSParameters* ioParams, const PSPreviewContext* preview);
void DoAboutDialog(void);

/*
** Implemented in AnonymizerPSMetal.mm. Runs the pass pipeline in place on
** the packed RGBA float buffer using the shared MSL kernels. Returns false
** on any failure - the caller falls back to the identical CPU pipeline.
*/
bool RunMetalPasses(float* buf, int w, int h, const AnonAlgo::LayerSettings& settings);

#endif
