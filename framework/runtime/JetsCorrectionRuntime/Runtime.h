#ifndef JETS_CORRECTION_FRAMEWORK_RUNTIME_H
#define JETS_CORRECTION_FRAMEWORK_RUNTIME_H

#include <correction.h>

#include <atomic>
#include <cstdint>
#include <initializer_list>
#include <string_view>
#include <vector>

namespace jets::framework {

using Values = std::vector<correction::Variable::Type>;

class CorrectionCall {
public:
  CorrectionCall(correction::Correction::Ref correction, bool tracing);
  void array(const Values& inputs, double* outputs, size_t size) const;
  double scalar(const Values& inputs) const;
  uint64_t count() const;

private:
  const correction::Correction::Ref correction_;
  const bool tracing_;
  mutable std::atomic<uint64_t> count_{0};
};

const CorrectionCall& bind(std::string_view name, size_t outputs,
                            std::initializer_list<std::string_view> inputs);

}  // namespace jets::framework

#endif
