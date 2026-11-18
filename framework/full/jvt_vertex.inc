        const auto vertex_info = jets::framework::event_inputs(vertices, 0, false, 0., false);
        const auto vertex = static_cast<std::int64_t>(vertex_info[4]);
        if (vertex == 0) {
            ATH_MSG_ERROR("No primary vertex or single NoVtx sentinel for NNJvt");
            return StatusCode::FAILURE;
        }
        const auto vertex_slot = static_cast<size_t>(vertex_info[3]);
