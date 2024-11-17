#include <algorithm>
#include <iostream>
#include <ranges>

#include <NvInfer.h>
#include <NvOnnxParser.h>

#include "logger.hpp"
#include "model_loader.hpp"

static auto loaderLogger = NVLogger {};

ModelLoader::ModelLoader(const std::filesystem::path &path)
{
	auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(loaderLogger));

	if (not builder)
		throw std::runtime_error("Could not create infer builder.");

	auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2({}));

	if (not network)
		throw std::runtime_error("Could not create network definition.");

	auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, loaderLogger));

	if (not parser)
		throw std::runtime_error("Could not create ONNX parser.");

	if (not parser->parseFromFile(
			path.generic_string().c_str(), static_cast<int32_t>(nvinfer1::ILogger::Severity::kVERBOSE)))
		throw std::runtime_error("Could not parse ONNX model.");

	for (int32_t i = 0; i < parser->getNbErrors(); ++i)
	{
		std::cout << parser->getError(i)->desc() << std::endl;
	}

	auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());

	if (not config)
		throw std::runtime_error("Could not create builder config.");

	auto serializedModel = std::unique_ptr<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config));

	if (not serializedModel)
		throw std::runtime_error("Could not create serialized model.");

	std::ranges::copy(
		std::span(reinterpret_cast<std::byte *>(serializedModel->data()), serializedModel->size()),
		std::back_inserter(m_model));
}

std::span<const std::byte> ModelLoader::model() const
{
	return m_model;
}