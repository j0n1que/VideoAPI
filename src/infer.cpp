#include <cstddef>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <ranges>
#include <stdexcept>

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize2.h"

#include "infer.hpp"
#include "logger.hpp"

static auto inferLogger = NVLogger {};

constexpr auto INPUT_WIDTH = 640;
constexpr auto INPUT_HEIGHT = 640;
constexpr auto INPUT_PLANES = 3;
constexpr auto OUTPUT_ROWS = 8400;
constexpr auto OUTPUT_COLS = 5;
constexpr auto CONFIDENCE_THRESHOLD = 0.25f;
constexpr auto NMS_THRESHOLD = 0.45f;

EvaInferContext::EvaInferContext(std::span<const std::byte> model)
	: m_cudaStream(nullptr, &cuStreamDestroy),
	  m_imageBuffer(INPUT_WIDTH * INPUT_HEIGHT * INPUT_PLANES),
	  m_modelInput(m_imageBuffer.size()),
	  m_modelOutput(OUTPUT_ROWS * OUTPUT_COLS),
	  m_detections(OUTPUT_ROWS)
{
	if (cuInit({}) != CUDA_SUCCESS)
		throw std::runtime_error("Could not initialize CUDA.");

	m_inferRuntime.reset(nvinfer1::createInferRuntime(inferLogger));

	if (m_inferRuntime == nullptr)
		throw std::runtime_error("Could not create nvinfer runtime.");

	m_cudaEngine.reset(m_inferRuntime->deserializeCudaEngine(model.data(), model.size()));

	if (m_cudaEngine == nullptr)
		throw std::runtime_error("Could not create CUDA engine.");

	m_executionContext.reset(m_cudaEngine->createExecutionContext());

	if (m_executionContext == nullptr)
		throw std::runtime_error("Could not create nvinfer execution context.");

	auto cudaStream = CUstream {};

	if (auto err = cuStreamCreate(&cudaStream, CU_STREAM_NON_BLOCKING); err != CUDA_SUCCESS)
		throw std::runtime_error(getCudaErrorString(err));

	m_cudaStream.reset(cudaStream);

	allocIOBuffers();
}

int EvaInferContext::infer(
	int width, int height, std::string_view format, int pixelStride, std::span<const std::byte> image)
{
	uploadImage(width, height, format, pixelStride, image);

	for (const auto &[tensorName, buffer] : m_IOBuffers)
	{
		if (not m_executionContext->setTensorAddress(tensorName.c_str(), reinterpret_cast<void *>(buffer.ptr)))
			throw std::runtime_error("Could not set tensor memory.");
	}

	if (not m_executionContext->enqueueV3(m_cudaStream.get()))
		throw std::runtime_error("Could not enqueue model inference.");

	m_IOBuffers.at("output0").read(
		std::span(reinterpret_cast<std::byte *>(m_modelOutput.data()), m_modelOutput.size() * sizeof(float)));

	if (auto err = cuStreamSynchronize(m_cudaStream.get()); err != CUDA_SUCCESS)
		throw std::runtime_error(getCudaErrorString(err));

	return filterOutput(width, height);
}

const std::vector<EvaInferResult> &EvaInferContext::detections() const
{
	return m_detections;
}

void EvaInferContext::allocIOBuffers()
{
	m_IOBuffers.clear();

	std::cout << "\nModel tensors" << std::endl;

	for (auto i = 0; i < m_cudaEngine->getNbIOTensors(); ++i)
	{
		const auto tensorName = m_cudaEngine->getIOTensorName(i);
		const auto tensorMode = m_cudaEngine->getTensorIOMode(tensorName);
		const auto tensorShape = m_cudaEngine->getTensorShape(tensorName);
		const auto tensorExtent =
			std::accumulate(tensorShape.d, tensorShape.d + tensorShape.nbDims, 1, std::multiplies {});

		if (tensorMode == nvinfer1::TensorIOMode::kINPUT or tensorMode == nvinfer1::TensorIOMode::kOUTPUT)
		{
			std::cout.width(10);
			std::cout << " - [" + std::string(tensorMode == nvinfer1::TensorIOMode::kINPUT ? "in" : "out") + "]";
			std::cout << tensorName << "(";

			for (auto k = 0; k < tensorShape.nbDims; ++k)
			{
				std::cout << tensorShape.d[k];

				if (k < tensorShape.nbDims - 1)
					std::cout << ", ";
			}

			std::cout << ")" << std::endl;
		}

		m_IOBuffers.emplace(
			tensorName, DeviceMemory(m_cudaStream.get(), static_cast<ssize_t>(tensorExtent * sizeof(float))));
	}

	std::cout << std::endl;
}

void EvaInferContext::uploadImage(
	int width, int height, std::string_view format, int pixelStride, std::span<const std::byte> image)
{
	auto fmt = std::string {};

	std::ranges::transform(
		format, std::back_inserter(fmt), [](char ch) { return static_cast<char>(std::tolower(ch)); });

	if (fmt != "bgr")
		throw std::invalid_argument("Unsupported image format.");

	constexpr auto channels = 3; // 3 for BGR

	stbir_resize(
		image.data(), width, height, width * pixelStride, m_imageBuffer.data(), INPUT_WIDTH, INPUT_HEIGHT,
		INPUT_WIDTH * channels, STBIR_BGR, STBIR_TYPE_UINT8, STBIR_EDGE_CLAMP, STBIR_FILTER_DEFAULT);

	struct BGR
	{
		uint8_t b, g, r;
	} const *pixels = reinterpret_cast<BGR *>(m_imageBuffer.data());
	constexpr auto planeSize = INPUT_WIDTH * INPUT_HEIGHT;

	for (auto i = 0; i < planeSize; ++i)
	{
		const auto &pix = *(pixels + i);

		m_modelInput.at(i) = static_cast<float>(pix.b) / 255.f;
		m_modelInput.at(i + planeSize) = static_cast<float>(pix.g) / 255.f;
		m_modelInput.at(i + planeSize * 2) = static_cast<float>(pix.r) / 255.f;
	}

	m_IOBuffers.at("images").write(
		std::span(reinterpret_cast<const std::byte *>(m_modelInput.data()), m_modelInput.size() * sizeof(float)));
}

int EvaInferContext::filterOutput(int imgWidth, int imgHeight)
{
	m_detections.clear();

	for (auto i = 0; i < OUTPUT_ROWS; ++i)
	{
		const auto boxX = m_modelOutput.at(i), boxY = m_modelOutput.at(i + OUTPUT_ROWS),
				   boxWidth = m_modelOutput.at(i + OUTPUT_ROWS * 2), boxHeight = m_modelOutput.at(i + OUTPUT_ROWS * 3),
				   score = m_modelOutput.at(i + OUTPUT_ROWS * 4);
		const auto widthScale = static_cast<float>(imgWidth) / static_cast<float>(INPUT_WIDTH),
				   heightScale = static_cast<float>(imgHeight) / static_cast<float>(INPUT_HEIGHT);

		m_detections.emplace_back(
			BBox {
				(boxX - boxWidth * 0.5f) * widthScale, (boxY - boxHeight * 0.5f) * heightScale, boxWidth * widthScale,
				boxHeight * heightScale},
			score);
	}

	auto outputEnd = std::remove_if(m_detections.begin(), m_detections.end(), [](const auto &detection) {
		return detection.score < CONFIDENCE_THRESHOLD;
	});

	// Filter detections using NMS algorithm
	std::sort(
		m_detections.begin(), outputEnd, [](const auto &left, const auto &right) { return left.score > right.score; });

	for (auto i = m_detections.begin(); i != outputEnd; ++i)
	{
		for (auto j = i + 1; j != outputEnd; ++j)
		{
			if (i->box.IOU(j->box) > NMS_THRESHOLD)
			{
				j->score = -1.f;
			}
		}
	}

	m_detections.erase(
		std::remove_if(m_detections.begin(), outputEnd, [](const auto &detection) { return detection.score < 0.f; }),
		m_detections.end());

	return static_cast<int>(m_detections.size());
}
