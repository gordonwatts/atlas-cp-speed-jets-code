    CorrectionCode NNJvtEfficiencyTool::getEfficiencyScaleFactor(const xAOD::Jet &jet, float &sf) const {
        return correctionlibScaleFactor(jet, sf, true);
    }

    CorrectionCode NNJvtEfficiencyTool::getInefficiencyScaleFactor(const xAOD::Jet &jet, float &sf) const {
        return correctionlibScaleFactor(jet, sf, false);
    }

    CorrectionCode NNJvtEfficiencyTool::correctionlibScaleFactor(const xAOD::Jet &jet, float &sf,
                                                                  bool passed) const {
        try {
            static const auto& scale_factor = jets::framework::bind("small_nnjvt_sf", 3,
                {"pt", "eta", "pass", "is_jvt_hs", "variation", "preselected"});
            const std::int64_t truth = m_accIsHS && m_accIsHS->isAvailable(jet)
                ? std::int64_t((*m_accIsHS)(jet) != 0) : -1;
            jets::framework::ScalarInputs inputs{jet.pt(), double(m_etaAcc(jet)),
                std::int64_t(passed), truth, std::int64_t(m_appliedSysSigma), std::int64_t(1)};
            double result[3];
            scale_factor.array(inputs.values(), result, 3);
            if (result[1] == 1 || result[1] == 2) {
                sf = static_cast<float>(result[0]);
                return result[1] == 1 ? CorrectionCode::Ok : CorrectionCode::OutOfValidityRange;
            }
            ATH_MSG_ERROR("CorrectionLib NNJvt SF returned an error");
        } catch (const std::exception& error) {
            ATH_MSG_ERROR("CorrectionLib NNJvt SF failed: " << error.what());
        }
        return CorrectionCode::Error;
    }
