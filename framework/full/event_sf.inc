      std::vector<double> factors;
      factors.reserve(particles->size());
      for (const xAOD::IParticle *particle : *particles)
        if (m_preselection.getBool(*particle, sys))
          factors.push_back(m_scaleFactorInputDecoration.get(*particle, sys));
      try {
        const auto lease = jets::framework::borrow_weights(factors);
        static const auto& call = jets::framework::bind("jet_event_sf", 0, {"event"});
        const auto scaleFactor = static_cast<float>(call.scalar({std::int64_t(0)}));
        m_scaleFactorOutputDecoration.set(*eventInfo, scaleFactor, sys);
      } catch (const std::exception& error) {
        ANA_MSG_ERROR("CorrectionLib event scale factor failed: " << error.what());
        return StatusCode::FAILURE;
      }
