#pragma once
#include <cstddef>
#include <cstdint>

namespace correction { class PluginRegistry; }
void register_event_inputs(correction::PluginRegistry&);

// Bind and release on the calling thread. Truth input coordinates are copied
// into the same P4 representation as the release. Vertex/weight arrays and
// their signed offsets are borrowed; the caller keeps them alive until release.
// Every function reports failure with nonzero return and jets_buffer_error().
extern "C" {
int jets_bind_truth(const std::int64_t* offsets, std::size_t n_offsets, const double* pt,
                    const double* eta, const double* phi, const double* mass, std::size_t rows);
int jets_unbind_truth();
int jets_bind_vertices(const std::int64_t* offsets, std::size_t n_offsets, const std::int32_t* types,
                       const std::int32_t* tracks, std::size_t rows);
int jets_unbind_vertices();
int jets_bind_weights(const std::int64_t* offsets, std::size_t n_offsets, const double* weights,
                      std::size_t rows);
int jets_unbind_weights();
const char* jets_buffer_error();
}
