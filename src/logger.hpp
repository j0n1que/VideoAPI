#pragma once

#include <NvInfer.h>

struct NVLogger : public nvinfer1::ILogger
{
	void log(Severity severity, const char *msg) noexcept override;
};