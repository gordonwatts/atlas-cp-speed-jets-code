    if (m_minM == 40000.f && m_maxM == 600000.f) {
      static const auto& correction = jets::framework::bind("large_selection", 3, {"pt", "eta", "phi", "m"});
      const jets::framework::Values inputs{particle->pt(), particle->eta(), particle->phi(), particle->m()};
      double result[3];
      correction.array(inputs, result, 3);
      const auto cuts = static_cast<std::uint32_t>(result[1]);
      asg::AcceptData accept(&m_accept);
      accept.setCutResult(m_minMassCutIndex, (cuts & 1u) != 0);
      accept.setCutResult(m_maxMassCutIndex, (cuts & 2u) != 0);
      return accept;
    }
