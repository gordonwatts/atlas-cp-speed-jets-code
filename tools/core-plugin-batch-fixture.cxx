#include <correction_plugin.h>

#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using correction::BatchInput;
using correction::PluginNode;
using Value = correction::Variable::Type;

namespace {

void require(bool condition, std::string_view label) {
  if (!condition) throw std::runtime_error(std::string(label));
}

template<class Exception, class Check>
void rejects(Check check, std::string_view message) {
  try {
    check();
  } catch (const Exception& error) {
    require(error.what() == message, error.what());
    return;
  }
  throw std::runtime_error("Expected exception " + std::string(message));
}

class Fallback : public PluginNode {
public:
  explicit Fallback(size_t width = 0) : width_(width) {}
  size_t output_size() const override { return width_; }
  double evaluate(const std::vector<Value>& values) const override {
    require(std::get<std::string>(values.at(0)) == std::string(90, 's'), "long owned string changed");
    return std::get<double>(values.at(1)) + 10 * std::get<int64_t>(values.at(2))
           + std::get<int64_t>(values.at(3)) + std::get<double>(values.at(4));
  }
  void evaluate_array(const std::vector<Value>& values, double* out, size_t size) const override {
    require(size == width_, "array fallback component count");
    out[0] = evaluate(values);
    out[1] = -out[0];
  }

private:
  size_t width_;
};

class BorrowedAlias : public PluginNode {
public:
  double evaluate(const std::vector<Value>& values) const override {
    return std::get<double>(values.at(0)) + 1;
  }
};

class Overrides : public PluginNode {
public:
  explicit Overrides(size_t width) : width_(width) {}
  size_t output_size() const override { return width_; }
  double evaluate(const std::vector<Value>&) const override {
    throw std::runtime_error("scalar fallback incorrectly reached an override");
  }
  void evaluate_batch(const std::vector<BatchInput>& inputs, size_t rows, double* out, size_t size) const override {
    require(width_ == 0 && size == rows, "override scalar call shape");
    for (size_t row = 0; row < rows; ++row)
      out[row] = std::get<double>(inputs.at(0).at(row)) + 100;
  }
  void evaluate_batch_array(const std::vector<BatchInput>& inputs, size_t rows, double* out, size_t size) const override {
    require(width_ == 2 && size == width_ * rows, "override array call shape");
    for (size_t row = 0; row < rows; ++row) {
      out[2 * row] = std::get<double>(inputs.at(0).at(row)) + 400;
      out[2 * row + 1] = -out[2 * row];
    }
  }

private:
  size_t width_;
};

void default_batches() {
  const std::array<double, 3> real{1.5, 2.5, 4.5};
  const std::array<int64_t, 6> integers{1, -1, 3, -1, 6, -1};
  const std::string label(90, 's');
  const std::vector<BatchInput> inputs{
    BatchInput(label), BatchInput::reals(real.data() + 2, real.size(), -1),
    BatchInput::integers(integers.data(), 3, 2), BatchInput(int64_t{8}), BatchInput(0.25)
  };
  const Fallback scalar, array(2);
  std::array<double, 3> output{};
  std::array<double, 6> pairs{};
  scalar.evaluate_batch(inputs, 3, output.data(), output.size());
  array.evaluate_batch_array(inputs, 3, pairs.data(), pairs.size());
  require(output == std::array{22.75, 40.75, 69.75}, "mixed owned and strided scalar");
  require(pairs == std::array{22.75, -22.75, 40.75, -40.75, 69.75, -69.75},
          "mixed owned and strided array");
  const std::vector<BatchInput> cached{
    BatchInput(label), BatchInput(1.5), BatchInput(int64_t{3}), BatchInput(int64_t{8}), BatchInput(0.25)
  };
  scalar.evaluate_batch(cached, 1, output.data(), 1);
  require(output[0] == 39.75, "one row");
  scalar.evaluate_batch(cached, 3, output.data(), output.size());
  require(output == std::array{39.75, 39.75, 39.75}, "all owned scalars");
  const std::vector<BatchInput> invalid_on_access{BatchInput::reals(nullptr, 0)};
  scalar.evaluate_batch(invalid_on_access, 0, nullptr, 0);
  array.evaluate_batch_array(invalid_on_access, 0, nullptr, 0);
  rejects<std::invalid_argument>([&] { scalar.evaluate_batch(inputs, 3, nullptr, 3); },
                                 "Batch output buffer too small or null");
  rejects<std::invalid_argument>([&] { array.evaluate_batch_array(inputs, 3, pairs.data(), 5); },
                                 "Array batch output buffer too small or size overflows");
  rejects<std::invalid_argument>([&] {
    array.evaluate_batch_array(inputs, std::numeric_limits<size_t>::max(), pairs.data(), pairs.size());
  }, "Array batch output buffer too small or size overflows");
  rejects<std::runtime_error>([&] { scalar.evaluate_batch_array({}, 0, nullptr, 0); },
                             "Scalar plugin requires evaluate_batch");

  std::vector<BatchInput> short_input(inputs);
  short_input[2] = BatchInput::integers(integers.data(), 1, 2);
  output.fill(-1);
  rejects<std::out_of_range>([&] { scalar.evaluate_batch(short_input, 3, output.data(), output.size()); },
                              "Batch input index");
  require(output == std::array{22.75, -1.0, -1.0}, "row-major partial output preserved");
}

void mutable_borrowed_broadcast() {
  std::array<double, 4> out{2.0, 0.0, 0.0, 0.0};
  const auto view = BatchInput::reals(out.data(), out.size(), 0);
  BorrowedAlias().evaluate_batch({view}, out.size(), out.data(), out.size());
  require(out == std::array{3.0, 4.0, 4.0, 4.0}, "borrowed stride-zero view may alias output");
}

void extension_override_dispatch() {
  correction::PluginRegistry registry;
  registry.register_node("trial.overrides", 1, [](const auto&, const auto& ctx) {
    return std::make_shared<Overrides>(ctx.output_size());
  });
  auto cset = correction::CorrectionSet::from_string(R"({
    "schema_version":3,"corrections":[
      {"name":"scalar","version":1,"inputs":[{"name":"x","type":"real"}],"output":{"name":"y","type":"real"},
       "data":{"nodetype":"extension","type":"trial.overrides","version":1,"config":{}}},
      {"name":"array","version":1,"inputs":[{"name":"x","type":"real"}],
       "output":{"name":"y","type":"real","shape":[2]},
       "data":{"nodetype":"extension","type":"trial.overrides","version":1,"config":{},"output_shape":[2]}}
    ]})", registry);
  const std::vector<BatchInput> invalid{BatchInput::reals(nullptr, 2)};
  std::array<double, 2> scalar{};
  rejects<std::invalid_argument>([&] { cset->at("scalar")->evaluate_batch(invalid, 2, scalar.data(), 2); },
                                 "Null batch input");
  const std::vector<BatchInput> input{BatchInput(2.5)};
  cset->at("scalar")->evaluate_batch(input, 2, scalar.data(), scalar.size());
  require(scalar == std::array{102.5, 102.5}, "ExtensionNode scalar override");
  std::array<double, 4> array{};
  cset->at("array")->evaluate_batch_array(input, 2, array.data(), array.size());
  require(array == std::array{402.5, -402.5, 402.5, -402.5}, "ExtensionNode array override");
}

}  // namespace

int main() try {
  default_batches();
  mutable_borrowed_broadcast();
  extension_override_dispatch();
  std::cout << "generic plugin batch fixture passed\n";
  std::cout << "BatchInput sizeof=" << sizeof(BatchInput) << " align=" << alignof(BatchInput)
            << ", Variable::Type sizeof=" << sizeof(Value) << ", ABI=" << correction::plugin_api_version << '\n';
} catch (const std::exception& error) {
  std::cerr << error.what() << '\n';
  return 1;
}
