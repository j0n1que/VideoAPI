#include <stdexcept>

#include "cuda.hpp"

DeviceMemory::DeviceMemory(DeviceMemory &&other) noexcept : ptr {}, extent {}, m_stream {}
{
	move(other);
}

DeviceMemory::DeviceMemory(CUstream stream, ssize_t extent) : m_stream(stream), ptr {}, extent(extent)
{
	if (auto err = cuMemAllocAsync(&ptr, extent, m_stream); err != CUDA_SUCCESS)
		throw std::runtime_error(getCudaErrorString(err));
}

DeviceMemory::~DeviceMemory()
{
	reset();
}

DeviceMemory &DeviceMemory::operator=(DeviceMemory &&other) noexcept
{
	reset();
	move(other);

	return *this;
}

void DeviceMemory::write(std::span<const std::byte> data)
{
	if (auto err = cuMemcpyAsync(ptr, reinterpret_cast<CUdeviceptr>(data.data()), data.size(), m_stream);
		err != CUDA_SUCCESS)
		throw std::runtime_error(getCudaErrorString(err));
}

void DeviceMemory::read(std::span<std::byte> buffer)
{
	if (auto err = cuMemcpyAsync(reinterpret_cast<CUdeviceptr>(buffer.data()), ptr, buffer.size(), m_stream);
		err != CUDA_SUCCESS)
		throw std::runtime_error(getCudaErrorString(err));
}

void DeviceMemory::reset()
{
	if (ptr)
	{
		cuMemFreeAsync(ptr, m_stream);

		ptr = {};
		extent = {};
		m_stream = {};
	}
}

void DeviceMemory::move(DeviceMemory &other) noexcept
{
	std::swap(ptr, other.ptr);
	std::swap(extent, other.extent);
	std::swap(m_stream, other.m_stream);
}

std::string getCudaErrorString(CUresult errCode)
{
	const char *string = "CUDA_ERROR_UNKNOWN";

	cuGetErrorString(errCode, &string);

	return string;
}
