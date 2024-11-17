#pragma once

#include "bbox.hpp"
#include "infer.hpp"

extern "C"
{
	EvaInferContext *eva_init();
	void eva_free(EvaInferContext *ctx);
	int eva_infer(EvaInferContext *ctx, int width, int height, const char *format, int pixelStride, const void *image);
	void eva_get_results(EvaInferContext *ctx, EvaInferResult const **ptr, int *count);
}