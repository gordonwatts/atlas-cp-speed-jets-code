    try {
        SG::ReadHandle<xAOD::VertexContainer> vertexHandle = SG::makeHandle(m_vertexContainer_key);
        const xAOD::VertexContainer& vertices = *vertexHandle;
        const xAOD::Vertex* hs = findHSVertex(vertices);
        if (!hs) {
            ATH_MSG_ERROR("No primary vertex or single NoVtx sentinel for NNJvt");
            return StatusCode::FAILURE;
        }

        SG::WriteDecorHandle<xAOD::JetContainer, float> score(m_jvtKey);
        SG::WriteDecorHandle<xAOD::JetContainer, char> pass(m_passJvtKey);
        ClassicHandleHolder moments(m_jvfCorrKey, m_sumPtTrkKey, m_rptKey);
        const std::int64_t vertex = hs->vertexType() == xAOD::VxType::PriVtx ? 2 : 1;
        static const auto& update = jets::framework::bind("small_nnjvt_update", 4,
            {"pt", "eta", "sum_pt_trk", "jvf_corr", "vertex_kind", "feature_status"});

        for (const xAOD::Jet* jet : jetCont) {
            std::int64_t features = 0;
            float track_sum = 0;
            float corr_jvf = 0;
            if (vertex == 2) {
                const auto& track_sums = moments.sumPtTrkHandle(*jet);
                if (!track_sums.empty()) {
                    track_sum = track_sums.at(hs->index() - vertices[0]->index());
                    corr_jvf = moments.jvfCorrHandle(*jet);
                    features = 1;
                }
            }
            const jets::framework::Values inputs{jet->pt(), jet->eta(), double(track_sum),
                                               double(corr_jvf), vertex, features};
            double result[4];
            update.array(inputs, result, 4);
            if (result[3] == 0) continue;  // Empty track vector: preserve inherited decorations.
            if (result[3] != 1) {
                ATH_MSG_ERROR("CorrectionLib NNJvt update returned an error");
                return StatusCode::FAILURE;
            }
            score(*jet) = static_cast<float>(result[0]);
            moments.decorate(*jet, static_cast<float>(result[1]));
            if (!pass.key().empty()) pass(*jet) = static_cast<char>(result[2]);
        }
        return StatusCode::SUCCESS;
    } catch (const std::exception& error) {
        ATH_MSG_ERROR("CorrectionLib NNJvt update failed: " << error.what());
        return StatusCode::FAILURE;
    }
