#include <pybind11/pybind11.h>

#include <optional>

#include "absl/strings/string_view.h"
#include "conformance.h"
#include "file_path.h"
#include "model_file_path.h"
#include "pybind11_abseil/absl_casters.h"
#include "pybind11_abseil/status_casters.h"
#include "pybind11_protobuf/native_proto_caster.h"
#include "visqol_api.h"
#include "visqol_manager.h"

namespace Visqol {

PYBIND11_MODULE(visqol_lib_py, m) {
  pybind11::google::ImportStatusModule();
  pybind11_protobuf::ImportNativeProtoCasters();

  m.doc() = "ViSQOL plugin";
  m.def("ConformanceSpeechCA01TranscodedValue",
        []() { return kConformanceSpeechCA01TranscodedLattice; });
  // Default model paths
  m.def("DefaultSpeechModelFile", []() { return kDefaultSpeechModelFile; });
  m.def("DefaultAudioModelFile", []() { return kDefaultAudioModelFile; });

  pybind11::class_<Visqol::VisqolManager>(m, "VisqolManager")
      .def(pybind11::init<>())
      .def("Init",
           pybind11::overload_cast<std::optional<Visqol::FilePath>, bool, bool,
                                   int, bool, bool, bool>(
               &Visqol::VisqolManager::Init),
           pybind11::arg("similarity_to_quality_mapper_model"),
           pybind11::arg("use_speech_mode"),
           pybind11::arg("use_unscaled_speech"), pybind11::arg("search_window"),
           pybind11::arg("use_lattice_model") = true,
           pybind11::arg("disable_global_alignment") = false,
           pybind11::arg("disable_realignment") = false)
      .def("Run", pybind11::overload_cast<const Visqol::FilePath&,
                                          const Visqol::FilePath&>(
                      &Visqol::VisqolManager::Run));

  pybind11::class_<Visqol::VisqolApi>(m, "VisqolApi")
      .def(pybind11::init<>())
      .def("Create", &Visqol::VisqolApi::Create,
           pybind11::call_guard<pybind11::gil_scoped_release>())
      .def("Measure", &Visqol::VisqolApi::Measure,
           pybind11::call_guard<pybind11::gil_scoped_release>());

  pybind11::class_<Visqol::FilePath>(m, "FilePath")
      .def(pybind11::init<absl::string_view>());
}

}  // namespace Visqol
