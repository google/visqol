#include <pybind11/pybind11.h>

#include "pybind11_abseil/absl_casters.h"
#include "pybind11_abseil/absl_numpy_span_caster.h"
#include "pybind11_abseil/status_casters.h"
#include "pybind11_protobuf/proto_casters.h"
#include "conformance.h"
#include "visqol_api.h"
#include "visqol_manager.h"
#include "third_party/visqol/src/proto/similarity_result.proto.h"
#include "src/proto/visqol_config.pb.h"

namespace Visqol {

PYBIND11_MODULE(visqol_lib_py, m) {
  pybind11::google::ImportStatusModule();
  pybind11::google::ImportProtoModule();

  m.doc() = "ViSQOL plugin";
  m.def("ConformanceSpeechCA01TranscodedValue",
        []() { return kConformanceSpeechCA01Transcoded; });

  pybind11::class_<Visqol::VisqolManager>(m, "VisqolManager")
      .def(pybind11::init<>())
      .def("Init", &Visqol::VisqolManager::Init)
      .def("Run", pybind11::overload_cast<const Visqol::FilePath &,
                                          const Visqol::FilePath &>(
                      &Visqol::VisqolManager::Run));

  pybind11::class_<Visqol::VisqolApi>(m, "VisqolApi")
      .def(pybind11::init<>())
      .def("Create", &Visqol::VisqolApi::Create)
      .def("Measure", &Visqol::VisqolApi::Measure);

  pybind11::class_<Visqol::FilePath>(m, "FilePath")
      .def(pybind11::init<const std::string &>());

  pybind11::google::RegisterProtoMessageType<SimilarityResultMsg>(m);

  m.def("MakeVisqolConfig", []() { return Visqol::VisqolConfig(); });
}
}  // namespace Visqol
