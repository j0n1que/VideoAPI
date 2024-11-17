#pragma once

struct BBox
{
	[[nodiscard]] float IOU(const BBox &other) const;

	float x = 0.f, y = 0.f, width = 0.f, height = 0.f;
};