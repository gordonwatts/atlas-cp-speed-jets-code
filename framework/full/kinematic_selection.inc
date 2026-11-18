    if (m_minPt == 200000.f && m_maxPt == 3000000.f && m_minRapidity == 0.f &&
        m_maxRapidity == 2.f && m_minEta == 0.f && m_maxEta == 0.f &&
        m_etaGapLow == 0.f && m_etaGapHigh == 0.f &&
        !m_useDressedProperties && !m_useClusterEta && !m_useConstituentMomentum) {
      static const auto& correction = jets::framework::bind("large_selection", 3, {"pt", "eta", "phi", "m"});
      const jets::framework::Values inputs{particle->pt(), particle->eta(), particle->phi(), particle->m()};
      double result[3];
      correction.array(inputs, result, 3);
      const auto cuts = static_cast<std::uint32_t>(result[0]);
      asg::AcceptData accept(&m_accept);
      accept.setCutResult(m_minPtCutIndex, (cuts & 1u) != 0);
      accept.setCutResult(m_maxPtCutIndex, (cuts & 2u) != 0);
      accept.setCutResult(m_maxRapidityCutIndex, (cuts & 4u) != 0);
      return accept;
    }
