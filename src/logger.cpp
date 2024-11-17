#include <iostream>

#include "logger.hpp"

void NVLogger::log(Severity severity, const char *msg) noexcept
{
	if (severity > Severity::kWARNING)
		return;

	std::cout.width(24);
	std::cout << std::left;

	switch (severity)
	{
		case Severity::kVERBOSE: std::cout << "Verbose:"; break;
		case Severity::kINFO: std::cout << "Info:"; break;
		case Severity::kWARNING: std::cout << "Warning:"; break;
		case Severity::kERROR: std::cout << "Error:"; break;
		case Severity::kINTERNAL_ERROR: std::cout << "Internal error:"; break;
	}

	std::cout << msg << std::endl;
}