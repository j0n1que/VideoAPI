#pragma once

#include <span>
#include <string>

#include <cuda.h>

struct DeviceMemory
{
	DeviceMemory(const DeviceMemory &) = delete;
	DeviceMemory(DeviceMemory &&other) noexcept;
	DeviceMemory(CUstream stream, ssize_t extent);
	~DeviceMemory();

	DeviceMemory &operator=(const DeviceMemory &) = delete;
	DeviceMemory &operator=(DeviceMemory &&other) noexcept;

	void write(std::span<const std::byte> data);
	void read(std::span<std::byte> buffer);

	void reset();
	void move(DeviceMemory &other) noexcept;

	CUdeviceptr ptr;
	ssize_t extent;

private:
	CUstream m_stream;
};

std::string getCudaErrorString(CUresult errCode);