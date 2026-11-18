# Standalone package in the generated AnalysisBase WorkDir.
atlas_subdir( JetsCorrectionRuntime )

find_package( correctionlib CONFIG REQUIRED )

atlas_add_library( JetsCorrectionRuntime
   JetsCorrectionRuntime/*.h Root/*.cxx
   PUBLIC_HEADERS JetsCorrectionRuntime
   LINK_LIBRARIES correctionlib )
