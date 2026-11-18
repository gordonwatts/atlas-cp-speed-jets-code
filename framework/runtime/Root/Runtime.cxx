#include "JetsCorrectionRuntime/Runtime.h"

#include <correction_plugin.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>

namespace {
namespace fs = std::filesystem;
using jets::framework::CorrectionCall;

fs::path required_file(const char* key) {
  const char* configured = std::getenv(key);
  if (!configured || !fs::path(configured).is_absolute() || !fs::is_regular_file(configured))
    throw std::runtime_error(std::string(key) + " must name an absolute regular file");
  return fs::canonical(configured);
}

class Runtime {
public:
  Runtime() {
    const fs::path provider = required_file("JETS_PROVIDER");
    const fs::path payload = required_file("JETS_PAYLOAD");
    correction::PluginRegistry registry;
    registry.load_library(provider.string());
    corrections_ = correction::CorrectionSet::from_file(payload.string(), registry);

    if (const char* trace = std::getenv("JETS_FRAMEWORK_TRACE")) {
      if (!fs::path(trace).is_absolute())
        throw std::runtime_error("JETS_FRAMEWORK_TRACE must be an absolute path");
      trace_.open(trace, std::ios::app);
      if (!trace_) throw std::runtime_error("cannot open JETS_FRAMEWORK_TRACE");
      trace_ << "provider " << provider.string() << '\n';
      trace_ << "payload " << payload.string() << '\n';
      trace_.flush();
    }
  }

  ~Runtime() {
    if (!trace_.is_open()) return;
    for (const auto& [name, call] : calls_)
      trace_ << "count " << name << ' ' << call->count() << '\n';
    std::ifstream maps("/proc/self/maps");
    std::set<std::string> loaded;
    for (std::string row; std::getline(maps, row);) {
      const auto start = row.find('/');
      if (start == std::string::npos) continue;
      if (row.find("libJet") == std::string::npos &&
          row.find("libjets.so") == std::string::npos &&
          row.find("libcorrectionlib") == std::string::npos)
        continue;
      loaded.insert(row.substr(start));
    }
    for (const auto& path : loaded) trace_ << "loaded " << path << '\n';
  }

  const CorrectionCall& bind(std::string_view name, size_t outputs,
                            std::initializer_list<std::string_view> inputs) {
    const std::string key(name);
    const auto correction = corrections_->at(key);
    if (correction->output_size() != outputs || correction->inputs().size() != inputs.size())
      throw std::runtime_error("unexpected shape for framework correction " + key);
    size_t index = 0;
    for (auto input : inputs)
      if (correction->inputs()[index++].name() != input)
        throw std::runtime_error("unexpected inputs for framework correction " + key);

    const std::lock_guard hold(mutex_);
    auto& slot = calls_[key];
    if (!slot) slot = std::make_unique<CorrectionCall>(correction, trace_.is_open());
    return *slot;
  }

private:
  std::unique_ptr<correction::CorrectionSet> corrections_;
  std::ofstream trace_;
  std::map<std::string, std::unique_ptr<CorrectionCall>> calls_;
  std::mutex mutex_;
};
}  // namespace

namespace jets::framework {

CorrectionCall::CorrectionCall(correction::Correction::Ref correction, bool tracing)
    : correction_(std::move(correction)), tracing_(tracing) {}

void CorrectionCall::array(const Values& inputs, double* outputs, size_t size) const {
  correction_->evaluate_array(inputs, outputs, size);
  if (tracing_) count_.fetch_add(1, std::memory_order_relaxed);
}

double CorrectionCall::scalar(const Values& inputs) const {
  const auto result = correction_->evaluate(inputs);
  if (tracing_) count_.fetch_add(1, std::memory_order_relaxed);
  return result;
}

uint64_t CorrectionCall::count() const {
  return count_.load(std::memory_order_relaxed);
}

const CorrectionCall& bind(std::string_view name, size_t outputs,
                           std::initializer_list<std::string_view> inputs) {
  static Runtime runtime;
  return runtime.bind(name, outputs, inputs);
}

}  // namespace jets::framework
