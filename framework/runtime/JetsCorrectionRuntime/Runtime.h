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

// Own the input values for the synchronous correction call. The optional
// build keeps capacity per thread and uses an independent vector for a
// nested call until the outer call releases its borrow.
class ScalarInputs {
public:
  explicit ScalarInputs(std::size_t capacity) {
#ifdef JETS_FRAMEWORK_REUSE_INPUTS
    static thread_local Scratch per_thread;
    if (!per_thread.borrowed) {
      per_thread.values.reserve(capacity);
      per_thread.borrowed = true;
      scratch_ = &per_thread;
      return;
    }
#endif
    owned_.reserve(capacity);
  }

#ifdef JETS_FRAMEWORK_REUSE_INPUTS
  explicit ScalarInputs(std::initializer_list<correction::Variable::Type> inputs)
      : ScalarInputs(inputs.size()) {
    values().assign(inputs);
  }

  ~ScalarInputs() {
    if (scratch_) {
      scratch_->values.clear();
      scratch_->borrowed = false;
    }
  }
#else
  explicit ScalarInputs(std::initializer_list<correction::Variable::Type> inputs)
      : owned_(inputs) {}
#endif

  ScalarInputs(const ScalarInputs&) = delete;
  ScalarInputs& operator=(const ScalarInputs&) = delete;

  Values& values() noexcept {
#ifdef JETS_FRAMEWORK_REUSE_INPUTS
    if (scratch_) return scratch_->values;
#endif
    return owned_;
  }

private:
  Values owned_;
#ifdef JETS_FRAMEWORK_REUSE_INPUTS
  struct Scratch {
    Values values;
    bool borrowed = false;
  };
  Scratch* scratch_ = nullptr;
#endif
};

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

void* provider_symbol(const char* name);

}  // namespace jets::framework

#endif
