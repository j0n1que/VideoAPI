#pragma once

#include <filesystem>
#include <span>
#include <vector>

class ModelLoader
{
public:
	explicit ModelLoader(const std::filesystem::path &path);
	ModelLoader(const ModelLoader &) = delete;

	ModelLoader &operator=(const ModelLoader &) = delete;

	std::span<const std::byte> model() const;

private:
	std::vector<std::byte> m_model;
};