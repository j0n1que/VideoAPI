#include <filesystem>
#include <fstream>
#include <vector>

#include "api.hpp"
#include "model_loader.hpp"

[[nodiscard]] std::vector<std::byte> loadBlob(const std::filesystem::path &modelPath)
{
	const auto fileSize = std::filesystem::file_size(modelPath);
	auto buffer = std::vector<std::byte>(fileSize);
	auto stream = std::ifstream(modelPath, std::ios_base::binary);

	if (not stream.is_open())
		throw std::runtime_error("Could not open model file.");

	stream.read(reinterpret_cast<char *>(buffer.data()), static_cast<std::streamsize>(fileSize));

	return buffer;
}

void dumpModel(std::span<const std::byte> model)
{
	std::ofstream ofs("cig_detector.trt", std::ios::out | std::ios::binary);

	if (not ofs.is_open())
		throw std::runtime_error("Could not dump serialized model.");

	ofs.write(reinterpret_cast<const char *>(model.data()), static_cast<std::streamsize>(model.size()));
	ofs.close();
}

extern "C"
{

	EvaInferContext *eva_init()
	{
		auto model = std::vector<std::byte> {};

		try
		{
			model = loadBlob("cig_detector.trt");
		}
		catch (...)
		{
			std::ranges::copy(ModelLoader("cig_detector.onnx").model(), std::back_inserter(model));
			dumpModel(model);
		}

		return new EvaInferContext(model);
	}

	void eva_free(EvaInferContext *ctx)
	{
		delete ctx;
	}

	int eva_infer(EvaInferContext *ctx, int width, int height, const char *format, int pixelStride, const void *image)
	{
		return ctx->infer(
			width, height, format, pixelStride,
			std::span(reinterpret_cast<const std::byte *>(image), width * height * pixelStride));
	}

	void eva_get_results(EvaInferContext *ctx, EvaInferResult const **ptr, int *count)
	{
		const auto &detections = ctx->detections();

		*count = static_cast<int>(detections.size());
		*ptr = detections.data();
	}
}