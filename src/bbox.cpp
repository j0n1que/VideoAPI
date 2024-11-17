#include <cmath>

#include "bbox.hpp"

float BBox::IOU(const BBox &other) const
{
	const auto _x1 = std::max(x, other.x), _y1 = std::max(y, other.y), _x2 = std::min(x + width, other.x + other.width),
			   _y2 = std::min(y + height, other.y + other.height);

	if (_x2 < _x1 or _y2 < _y1)
		return 0.0f;

	const auto intersectionArea = (_x2 - _x1) * (_y2 - _y1);

	return intersectionArea / (width * height + other.width * other.height - intersectionArea);
}