#pragma once

#include <memory>
#include <unordered_map>
#include <vector>

#include <NvInfer.h>

#include "bbox.hpp"
#include "cuda.hpp"

struct EvaInferResult
{
	// output shape is 5 columns, 8400 rows
	// [0:4] columns are x_center, y_center, width, height of box
	// [4] column is objectness score
	BBox box = {};
	float score = 0.f;
};

class EvaInferContext
{
public:
	EvaInferContext(const EvaInferContext &) = delete;
	explicit EvaInferContext(std::span<const std::byte> model);

	EvaInferContext &operator=(const EvaInferContext &) = delete;

	int infer(int width, int height, std::string_view format, int pixelStride, std::span<const std::byte> image);

	[[nodiscard]] const std::vector<EvaInferResult> &detections() const;

private:
	void allocIOBuffers();
	void uploadImage(int width, int height, std::string_view format, int pixelStride, std::span<const std::byte> image);
	int filterOutput(int imgWidth, int imgHeight);

	std::unique_ptr<nvinfer1::IRuntime> m_inferRuntime;
	std::unique_ptr<nvinfer1::ICudaEngine> m_cudaEngine;
	std::unique_ptr<nvinfer1::IExecutionContext> m_executionContext;
	std::unique_ptr<std::remove_pointer_t<CUstream>, decltype(cuStreamDestroy) *> m_cudaStream;
	std::unordered_map<std::string, DeviceMemory> m_IOBuffers;
	std::vector<std::byte> m_imageBuffer;
	std::vector<float> m_modelInput, m_modelOutput;
	std::vector<EvaInferResult> m_detections;
};
