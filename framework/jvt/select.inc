        static const SG::ConstAccessor<float> score("NNJvt");
        static const auto& selection = jets::framework::bind("small_nnjvt_select", 2,
            {"pt", "eta", "score", "score_present"});
        const bool present = score.isAvailable(*jet);
        const jets::framework::Values inputs{jet->pt(), double(m_etaAcc(*jet)),
            present ? double(score(*jet)) : 0.0, std::int64_t(present)};
        double result[2];
        selection.array(inputs, result, 2);
        if (result[1] != 1) throw std::runtime_error("CorrectionLib NNJvt selection needs a score");
        return result[0] != 0;
