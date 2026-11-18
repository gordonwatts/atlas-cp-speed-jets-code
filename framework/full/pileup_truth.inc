  SG::ReadHandle<xAOD::JetContainer> truthHandle = SG::makeHandle(m_truthJetsKey);
  if (!truthHandle.isValid()) {
    ATH_MSG_ERROR("Invalid truth jet collection for CorrectionLib pile-up labels");
    return StatusCode::FAILURE;
  }
  try {
    const auto truth = jets::framework::truth_columns(*truthHandle);
    const auto lease = jets::framework::borrow_truth(truth);
    static const auto& call = jets::framework::bind("jet_pileup_truth", 2,
        {"pt", "eta", "phi", "m", "event"});
    SG::WriteDecorHandle<xAOD::JetContainer, char> hs(m_decIsHSKey), pu(m_decIsPUKey);
    for (const auto* jet : jets) {
      const jets::framework::Values input{jet->pt(), jet->eta(), jet->phi(), jet->m(), std::int64_t(0)};
      double result[2];
      call.array(input, result, 2);
      hs(*jet) = static_cast<char>(result[0]);
      pu(*jet) = static_cast<char>(result[1]);
    }
    return StatusCode::SUCCESS;
  } catch (const std::exception& error) {
    ATH_MSG_ERROR("CorrectionLib pile-up truth labeling failed: " << error.what());
    return StatusCode::FAILURE;
  }
