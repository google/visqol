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

    py::array_t<double> run(py::array_t<double> reference, const size_t reference_sr, py::array_t<double> degraded, const size_t degraded_sr)
    {
        py::buffer_info buf1 = reference.request();
        py::buffer_info buf2 = degraded.request();

        if (buf1.size != buf2.size) {
            throw std::runtime_error("Input shapes must match");
        }

        double *ptr1 = (double*) buf1.ptr;
        double *ptr2 = (double*) buf2.ptr;

        // allocate the buffer
        py::array_t<double> result = py::array_t<double>((buf1.shape.size() > 1) ? buf1.shape[0] : 1);
        py::buffer_info buf3 = result.request();
        double *ptr3 = (double*) buf3.ptr;

        size_t num_channels = 1;
        size_t num_samples = 0;

        if (buf1.shape.size() == 1)
        {
            num_samples = buf1.shape[0];
        }
        else if (buf1.shape.size() == 2)
        {
            num_channels = buf1.shape[0];
            num_samples = buf1.shape[1];
        }

        for (size_t channel = 0; channel < num_channels; channel++)
        {
            AudioSignal ref_signal{std::vector<double>(&ptr1[channel*num_samples], &ptr1[channel*num_samples + num_samples]), reference_sr};
            AudioSignal deg_signal{std::vector<double>(&ptr2[channel*num_samples], &ptr2[channel*num_samples + num_samples]), degraded_sr};
            const auto status_or = manager.Run(ref_signal, deg_signal);
            if (status_or.ok())
            {
                ptr3[channel] = status_or.value().moslqo();
            }
            else
            {
                ptr3[channel] = 0.0;
                std::cerr << "Failed to get score!" << std::endl;
            }
        }

        result.resize({num_channels});
        return result;
    }

    double run(const std::string& reference, const std::string& degraded)
    {
        double score = 0.0;
        const auto status_or = manager.Run(reference, degraded);
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
        .def("run", static_cast<py::array_t<double> (ViSQOL::*)(
            py::array_t<double>, const size_t,
            py::array_t<double>, const size_t)>(&ViSQOL::run),
            "Measure MOS for array of buffers",
            py::arg("reference"),
            py::arg("reference_sr"),
            py::arg("degraded"),
            py::arg("degraded_sr"))
        .def("run", static_cast<double (ViSQOL::*)(const std::string&, const std::string&)>(&ViSQOL::run),
            "Measure MOS for files",
            py::arg("reference"),
            py::arg("degraded"));
}

}  // namespace Visqol
