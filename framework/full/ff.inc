        if (!m_currentSysData) return CP::CorrectionCode::Error;
        try {
            using jets::framework::Values;
            using jets::framework::MomentumAux;
            static const MomentumAux momentum("");
            static const SG::ConstAccessor<int> label("R10TruthLabel_R21Precision_2022v1");
            static const auto& call = jets::framework::bind("large_jmr", 6,
                {"pt", "eta", "phi", "m", "truth_label", "truth_label_available", "systematic", "event_index"});
            const xAOD::JetContainer* truth_jets = nullptr;
            jets::framework::FourColumns truth;
            if (evtStore()->retrieve(truth_jets, m_truth_jetColl).isSuccess())
                truth = jets::framework::truth_columns(*truth_jets);
            const auto lease = jets::framework::borrow_truth(truth);

            const std::string state =
                CP::SystematicVariation(m_currentSysData->SysBaseName, m_currentSysData->SysParameter).name();
            Values inputs;
            inputs.reserve(8);
            momentum.append(jet_reco, inputs);
            const bool has_label = label.isAvailable(jet_reco);
            inputs.emplace_back(std::int64_t(has_label ? label(jet_reco) : 0));
            inputs.emplace_back(std::int64_t(has_label));
            inputs.emplace_back(state);
            inputs.emplace_back(std::int64_t(0));
            double result[6];
            call.array(inputs, result, 6);
            if (result[5] != 1.) throw std::runtime_error("FF jet smearing needs a returned status");
            if (result[4] == CP::CorrectionCode::Ok) {
                momentum.write(jet_reco, result);
                return CP::CorrectionCode::Ok;
            }
            if (result[4] == CP::CorrectionCode::OutOfValidityRange)
                return CP::CorrectionCode::OutOfValidityRange;
            return CP::CorrectionCode::Error;
        } catch (const std::exception& error) {
            ATH_MSG_ERROR("CorrectionLib FF jet smearing failed: " << error.what());
            return CP::CorrectionCode::Error;
        }
