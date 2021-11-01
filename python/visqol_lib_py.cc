#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "visqol_manager.h"

#include <string>
#include <vector>
#include <iostream>

namespace Visqol {

namespace py = pybind11;

class ViSQOL {
public:
    ViSQOL(const std::string& sim_to_quality_mapper_model,
           const bool use_speech_mode,
           const bool use_unscaled_speech,
           const int search_window)
    {
        absl::Status status =  manager.Init(Visqol::FilePath(sim_to_quality_mapper_model), use_speech_mode, use_unscaled_speech, search_window);
        if (!status.ok())
        {
            std::cerr << "Failed to initialize ViSQOL" << std::endl;
        }
    }

    double run(const std::vector<double>& ref, const size_t ref_sr, const std::vector<double>& deg, const size_t deg_sr)
    {
        double score = 0.0;
        AudioSignal ref_signal{ref, ref_sr};
        AudioSignal deg_signal{deg, deg_sr};
        const auto status_or = manager.Run(ref_signal, deg_signal);
        if (status_or.ok())
        {
            score = status_or.value().moslqo();
        }
        else
        {
            std::cerr << "Failed to get score!" << std::endl;
        }
        return score;
    }

    double run(const std::string& ref, const std::string& deg)
    {
        double score = 0.0;
        const auto status_or = manager.Run(ref, deg);
        if (status_or.ok())
        {
            score = status_or.value().moslqo();
        }
        else
        {
            std::cerr << "Failed to get score!" << std::endl;
        }
        return score;
    }

private:
    Visqol::VisqolManager manager;
};

PYBIND11_MODULE(pyvisqol, m) {

  m.doc() = "ViSQOL plugin";

  py::class_<ViSQOL>(m, "ViSQOL")
        .def(py::init<const std::string &, const bool, const bool, const int>(),
            py::arg("sim_to_quality_mapper_model"),
            py::arg("use_speech_mode")=false,
            py::arg("use_unscaled_speech")=false,
            py::arg("search_window")=60
        )
        .def("run", static_cast<double (ViSQOL::*)(const std::vector<double>&, const size_t, const std::vector<double>&, const size_t)>(&ViSQOL::run),
            "Measure MOS for buffers",
            py::arg("ref"),
            py::arg("ref_sr"),
            py::arg("deg"),
            py::arg("deg_sr"))
        .def("run", static_cast<double (ViSQOL::*)(const std::string&, const std::string&)>(&ViSQOL::run),
            "Measure MOS for files",
            py::arg("ref"),
            py::arg("deg"));
}

}  // namespace Visqol
