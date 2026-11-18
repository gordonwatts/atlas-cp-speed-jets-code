    if (!m_isInit) {
        ATH_MSG_ERROR("Jet uncertainty tool must be initialized before CorrectionLib evaluation");
        return CP::CorrectionCode::Error;
    }
    try {
        using jets::framework::Values;
        using jets::framework::MomentumAux;
        static const MomentumAux assigned(""), calo_scale("JetJMSScaleMomentumCalo_");
        static const SG::ConstAccessor<float> npv("NPV");
        static const SG::ConstAccessor<int> parton("PartonTruthLabelID");
        static const SG::ConstAccessor<int> large_label("R10TruthLabel_R21Precision_2022v1");

        const std::string state = [&] {
            if (m_currentSystSet.empty()) return std::string("NOSYS");
            if (m_currentSystSet.size() != 1) throw std::runtime_error("expected one jet variation");
            const auto& variation = *m_currentSystSet.begin();
            if (!m_pseudoDataJERsmearingMode) return variation.name();
            return CP::SystematicVariation(variation.basename() + "_PseudoData", variation.parameter()).name();
        }();

        const std::initializer_list<std::string_view> inputs_names{"pt", "eta", "phi", "m", "average_mu", "uncertainty_npv", "event_available",
            "preselected", "parton_truth_label", "parton_truth_label_available", "large_r_truth_label",
            "large_r_truth_label_available", "calo_pt", "calo_eta", "calo_phi", "calo_m", "calo_available",
            "systematic"};
        const bool small = m_jetDef == "AntiKt4EMPFlow";
        const auto& call = [&]() -> const jets::framework::CorrectionCall& {
            if (small) {
                static const auto& small_call = jets::framework::bind("small_uncertainty", 8, inputs_names);
                return small_call;
            }
            static const auto& large_call = jets::framework::bind("large_uncertainty", 8, inputs_names);
            return large_call;
        }();

        Values inputs;
        inputs.reserve(18);
        assigned.append(jet, inputs);
        inputs.emplace_back(static_cast<double>(eInfo.averageInteractionsPerCrossing()));
        inputs.emplace_back(npv.isAvailable(eInfo) ? static_cast<double>(npv(eInfo)) : -1.);
        inputs.emplace_back(std::int64_t(1));
        inputs.emplace_back(std::int64_t(1));
        const bool has_parton = parton.isAvailable(jet);
        const bool has_label = large_label.isAvailable(jet);
        inputs.emplace_back(std::int64_t(has_parton ? parton(jet) : 0));
        inputs.emplace_back(std::int64_t(has_parton));
        inputs.emplace_back(std::int64_t(has_label ? large_label(jet) : 0));
        inputs.emplace_back(std::int64_t(has_label));
        const bool has_calo = calo_scale.available(jet);
        if (has_calo) calo_scale.append(jet, inputs);
        else for (int i = 0; i < 4; ++i) inputs.emplace_back(0.);
        inputs.emplace_back(std::int64_t(has_calo));
        inputs.emplace_back(state);

        double result[8];
        call.array(inputs, result, 8);
        if (result[5] != 1.) throw std::runtime_error("uncertainty tool needs a status");
        if (result[4] == CP::CorrectionCode::Ok) {
            assigned.write(jet, result);
            return CP::CorrectionCode::Ok;
        }
        if (result[4] == CP::CorrectionCode::OutOfValidityRange)
            return CP::CorrectionCode::OutOfValidityRange;
        return CP::CorrectionCode::Error;
    } catch (const std::exception& error) {
        ATH_MSG_ERROR("CorrectionLib jet uncertainty stage failed: " << error.what());
        return CP::CorrectionCode::Error;
    }
