#pragma once

#include <TH1.h>
#include <rapidjson/document.h>

#include <memory>

namespace jets {
std::unique_ptr<TH1> restore_histogram(const rapidjson::Value& data);
}  // namespace jets
