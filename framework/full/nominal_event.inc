    // The frozen small-R recipe needs both nominal NPV and the GSC context.
    // All container retrieval and original missing-input fallbacks remain here.
    if (m_doResidual || m_doGSC) {
      const bool has_pv = PVIndexAccessor.isAvailable(*eventObj);
      const int pv_index = has_pv ? PVIndexAccessor(*eventObj) : 0;
      SG::ReadHandle<xAOD::VertexContainer> vertices(m_pvKey);
      if (!vertices.isValid()) {
        ATH_MSG_WARNING("JetCalibrationTool could not retrieve primary vertices");
        jetEventInfo.setNPV(0);
        jetEventInfo.setPVIndex(pv_index);
        return StatusCode::SUCCESS;
      }
      try {
        const auto result = jets::framework::event_inputs(*vertices, pv_index, has_pv, 0., false);
        jetEventInfo.setNPV(result[0]);
        jetEventInfo.setPVIndex(static_cast<int>(result[2]));
      } catch (const std::exception& error) {
        ATH_MSG_ERROR("CorrectionLib nominal event inputs failed: " << error.what());
        return StatusCode::FAILURE;
      }
    }
