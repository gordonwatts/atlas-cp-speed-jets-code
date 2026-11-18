#pragma once

#include <lwtnn/generic/FastGraph.hh>
#include <lwtnn/parse_json.hh>

namespace jets::portable {
// Both providers use the same release lwtnn graph and Eigen configuration.
using JvtNetwork = lwt::generic::FastGraph<double>;
using JvtNetworkFeatures = lwt::VectorX<double>;
using JvtNetworkNodes = lwt::generic::NodeVec<double>;
}  // namespace jets::portable
