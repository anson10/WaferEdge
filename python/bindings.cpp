// waferedge._core: the CPU signatures, the rule classifier and the cluster finder on NumPy
// arrays (docs/python.md).
//
// Maps come in through the buffer protocol without a copy: a 2-D (rows, cols) or 3-D
// (maps, rows, cols) C-contiguous uint8 array is read in place as WaferMapViews. Anything
// else (another dtype, a strided view) is refused with TypeError rather than copied behind
// the caller's back; np.ascontiguousarray(a, dtype=np.uint8) makes one that fits.
// Batches release the GIL while they run.
#include "waferedge/backend.hpp"
#include "waferedge/classifier.hpp"
#include "waferedge/clusters.hpp"
#include "waferedge/features.hpp"
#include "waferedge/geometry.hpp"
#include "waferedge/hough.hpp"
#include "waferedge/map_file.hpp"
#include "waferedge/wafer_map.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;
using namespace waferedge;

namespace {

// Maps in a uint8 buffer, read in place: one map (2-D) or a stack of equal-sized maps (3-D).
struct Maps {
    py::buffer_info info;
    std::size_t count = 0;
    int rows = 0;
    int cols = 0;

    [[nodiscard]] WaferMapView view(std::size_t i) const {
        const auto per = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
        const auto* base = static_cast<const std::uint8_t*>(info.ptr);
        return {std::span<const std::uint8_t>(base + i * per, per), rows, cols};
    }
};

Maps maps_from(const py::buffer& b, bool batch) {
    Maps m{b.request(), 0, 0, 0};
    const auto& info = m.info;
    if (info.format != py::format_descriptor<std::uint8_t>::format()) {
        throw py::type_error("wafer maps must be uint8 (0 off-wafer, 1 pass, >= 2 fail)");
    }
    const auto want = batch ? 3 : 2;
    if (info.ndim != want) {
        throw py::type_error(batch ? "a batch must be a 3-D array (maps, rows, cols)"
                                   : "a wafer map must be a 2-D array (rows, cols)");
    }
    // C-contiguous: the last stride is one byte, each earlier one the product of the rest.
    py::ssize_t expect = 1;
    for (auto d = info.ndim; d-- > 0;) {
        if (info.shape[static_cast<std::size_t>(d)] > 1 &&
            info.strides[static_cast<std::size_t>(d)] != expect) {
            throw py::type_error("wafer maps must be C-contiguous: use np.ascontiguousarray");
        }
        expect *= info.shape[static_cast<std::size_t>(d)];
    }
    const auto& shape = info.shape;
    m.count = batch ? static_cast<std::size_t>(shape[0]) : 1;
    m.rows = static_cast<int>(shape[batch ? 1 : 0]);
    m.cols = static_cast<int>(shape[batch ? 2 : 1]);
    if (m.rows < 1 || m.cols < 1 || m.rows > 4096 || m.cols > 4096) {
        throw py::value_error("a wafer map needs 1 to 4096 rows and columns");
    }
    return m;
}

template <typename T, std::size_t N>
py::array_t<T> to_array(const std::array<T, N>& a) {
    return py::array_t<T>(static_cast<py::ssize_t>(N), a.data());
}

py::dict features_dict(const Features& f) {
    py::dict d;
    d["dies"] = f.dies;
    d["fails"] = f.fails;
    d["zone_dies"] = to_array(f.zone_dies);
    d["zone_fails"] = to_array(f.zone_fails);
    d["sector_dies"] = to_array(f.sector_dies);
    d["sector_fails"] = to_array(f.sector_fails);
    d["ring_dies"] = to_array(f.ring_dies);
    d["ring_fails"] = to_array(f.ring_fails);
    return d;
}

py::dict cluster_dict(const Cluster& c) {
    const ClusterShape s = shape(c);
    py::dict d;
    d["size"] = c.size;
    d["first"] = c.first;
    d["row"] = s.row;
    d["col"] = s.col;
    d["elongation"] = s.elongation;
    d["angle_deg"] = s.angle_deg;
    d["min_row"] = c.min_r;
    d["max_row"] = c.max_r;
    d["min_col"] = c.min_c;
    d["max_col"] = c.max_c;
    return d;
}

py::dict line_dict(const HoughLine& l) {
    py::dict d;
    d["angle"] = l.angle;
    d["rho"] = l.rho;
    d["votes"] = l.votes;
    d["line_dies"] = l.line_dies;
    return d;
}

std::string name_of(Pattern p) {
    return std::string(pattern_name(p));
}

// The classifier with the extractor it needs (its caches are per object, not shared).
class Classifier {
public:
    explicit Classifier(RuleClassifier rules) : rules_(std::move(rules)) {}
    // Move-only, said out loud: the extractor's geometry cache isn't copyable, but
    // std::is_copy_constructible can't see that through std::map, and pybind11 asks it.
    Classifier(const Classifier&) = delete;
    Classifier& operator=(const Classifier&) = delete;
    Classifier(Classifier&&) noexcept = default;
    Classifier& operator=(Classifier&&) noexcept = default;
    ~Classifier() = default;

    std::string classify(const py::buffer& map) {
        const Maps m = maps_from(map, false);
        return name_of(rules_.classify(extractor_.extract(m.view(0))).pattern);
    }

    std::vector<std::string> classify_batch(const py::buffer& maps) {
        const Maps m = maps_from(maps, true);
        std::vector<Pattern> out(m.count);
        {
            py::gil_scoped_release release;
            for (std::size_t i = 0; i < m.count; ++i) {
                out[i] = rules_.classify(extractor_.extract(m.view(i))).pattern;
            }
        }
        std::vector<std::string> names;
        names.reserve(out.size());
        for (const Pattern p : out) {
            names.push_back(name_of(p));
        }
        return names;
    }

    std::string explain(const py::buffer& map) {
        const Maps m = maps_from(map, false);
        return rules_.explain(extractor_.extract(m.view(0)));
    }

    [[nodiscard]] std::string to_text() const { return rules_.to_text(); }

private:
    RuleClassifier rules_;
    SignalExtractor extractor_;
};

Classifier load_rules(const std::string& path) {
    auto rules = RuleClassifier::load(path);
    if (!rules) {
        throw py::value_error(rules.error());
    }
    return Classifier(std::move(*rules));
}

Classifier parse_rules(const std::string& text) {
    auto rules = RuleClassifier::parse(text);
    if (!rules) {
        throw py::value_error(rules.error());
    }
    return Classifier(std::move(*rules));
}

} // namespace

PYBIND11_MODULE(_core, m) {
    m.doc() = "WaferEdge's CPU wafer-map signatures, rule classifier and clusters on NumPy";
    m.attr("__version__") = WAFEREDGE_PY_VERSION;

    std::vector<std::string> patterns;
    for (std::size_t i = 0; i < kPatternCount; ++i) {
        patterns.push_back(name_of(static_cast<Pattern>(i)));
    }
    m.attr("PATTERNS") = patterns;
    std::vector<std::string> signals;
    for (std::size_t i = 0; i < kSignalCount; ++i) {
        signals.emplace_back(signal_name(static_cast<Signal>(i)));
    }
    m.attr("SIGNALS") = signals;

    m.def("backend", [] { return std::string(best_features_name()); },
          "The feature backend this CPU runs: avx2 or scalar.");

    m.def(
        "features",
        [](const py::buffer& map) {
            const Maps m = maps_from(map, false);
            const Geometry geometry(m.rows, m.cols);
            return features_dict(best_features()(m.view(0), geometry));
        },
        py::arg("map"),
        "Die and fail counts of one map: overall, per zone (5, equal area), sector (8) and ring "
        "(10).");

    m.def(
        "features_batch",
        [](const py::buffer& maps) {
            const Maps m = maps_from(maps, true);
            const auto n = static_cast<py::ssize_t>(m.count);
            py::array_t<std::uint32_t> dies(n);
            py::array_t<std::uint32_t> fails(n);
            py::array_t<std::uint32_t> zone_fails({n, py::ssize_t{kZones}});
            py::array_t<std::uint32_t> sector_fails({n, py::ssize_t{kSectors}});
            py::array_t<std::uint32_t> ring_fails({n, py::ssize_t{kRings}});
            auto d = dies.mutable_unchecked<1>();
            auto f = fails.mutable_unchecked<1>();
            auto z = zone_fails.mutable_unchecked<2>();
            auto s = sector_fails.mutable_unchecked<2>();
            auto r = ring_fails.mutable_unchecked<2>();
            {
                py::gil_scoped_release release;
                const Geometry geometry(m.rows, m.cols);
                const auto fn = best_features();
                for (py::ssize_t i = 0; i < n; ++i) {
                    const Features x = fn(m.view(static_cast<std::size_t>(i)), geometry);
                    d(i) = x.dies;
                    f(i) = x.fails;
                    for (py::ssize_t k = 0; k < kZones; ++k) {
                        z(i, k) = x.zone_fails[static_cast<std::size_t>(k)];
                    }
                    for (py::ssize_t k = 0; k < kSectors; ++k) {
                        s(i, k) = x.sector_fails[static_cast<std::size_t>(k)];
                    }
                    for (py::ssize_t k = 0; k < kRings; ++k) {
                        r(i, k) = x.ring_fails[static_cast<std::size_t>(k)];
                    }
                }
            }
            py::dict out;
            out["dies"] = dies;
            out["fails"] = fails;
            out["zone_fails"] = zone_fails;
            out["sector_fails"] = sector_fails;
            out["ring_fails"] = ring_fails;
            return out;
        },
        py::arg("maps"),
        "Counts of a stack of equal-sized maps as arrays: dies and fails (N,), zone_fails "
        "(N, 5), sector_fails (N, 8), ring_fails (N, 10).");

    m.def(
        "signals",
        [](const py::buffer& maps) {
            const py::buffer_info probe = maps.request();
            const bool batch = probe.ndim == 3;
            const Maps m = maps_from(maps, batch);
            const auto n = static_cast<py::ssize_t>(m.count);
            py::array_t<double> out({n, static_cast<py::ssize_t>(kSignalCount)});
            auto o = out.mutable_unchecked<2>();
            {
                py::gil_scoped_release release;
                SignalExtractor extractor;
                for (py::ssize_t i = 0; i < n; ++i) {
                    const Signals s = extractor.extract(m.view(static_cast<std::size_t>(i)));
                    for (py::ssize_t k = 0; k < static_cast<py::ssize_t>(kSignalCount); ++k) {
                        o(i, k) = s[static_cast<std::size_t>(k)];
                    }
                }
            }
            if (!batch) {
                return py::array_t<double>(out[py::int_(0)]);
            }
            return out;
        },
        py::arg("maps"),
        "The 13 classifier signals (names in SIGNALS) of one map (13,) or a stack (N, 13).");

    m.def(
        "clusters",
        [](const py::buffer& map) {
            const Maps m = maps_from(map, false);
            ClusterFinder finder;
            finder.run(m.view(0));
            py::array_t<std::int32_t> labels({m.rows, m.cols});
            const auto src = finder.labels();
            std::copy(src.begin(), src.end(), labels.mutable_data());
            py::list clusters;
            for (const Cluster& c : finder.clusters()) {
                clusters.append(cluster_dict(c));
            }
            return py::make_tuple(labels, clusters);
        },
        py::arg("map"),
        "Connected fail clusters (8-neighbour): a (rows, cols) int32 label per die (-1 for none, "
        "else the cluster's index) and one dict per cluster, in the order of their first dies.");

    m.def(
        "hough_line",
        [](const py::buffer& map) {
            const Maps m = maps_from(map, false);
            HoughTransform hough;
            return line_dict(hough.run(m.view(0)));
        },
        py::arg("map"), "The strongest line of fail dies (a scratch): angle, rho, votes, dies.");

    py::class_<Classifier>(m, "Classifier",
                           "The rule classifier (ADR-0005): the first rule whose conditions all "
                           "hold names the pattern; none fired means 'none'.")
        .def_static("defaults", [] { return Classifier(RuleClassifier::defaults()); })
        .def_static("load", &load_rules, py::arg("path"), "Rules from a file (config/rules.txt).")
        .def_static("parse", &parse_rules, py::arg("text"), "Rules from text.")
        .def("classify", &Classifier::classify, py::arg("map"))
        .def("classify_batch", &Classifier::classify_batch, py::arg("maps"))
        .def("explain", &Classifier::explain, py::arg("map"),
             "Each rule's conditions on this map: which held, which failed.")
        .def("to_text", &Classifier::to_text);

    m.def(
        "read_wmap",
        [](const std::string& path) {
            auto set = MapSet::load(path);
            if (!set) {
                throw py::value_error(set.error());
            }
            py::list maps;
            py::list truth;
            py::list split;
            const auto n = static_cast<py::ssize_t>(set->size());
            py::array_t<std::int32_t> wafer(n);
            py::array_t<std::int32_t> lot(n);
            py::array_t<std::int64_t> tested_at(n);
            auto w = wafer.mutable_unchecked<1>();
            auto l = lot.mutable_unchecked<1>();
            auto t = tested_at.mutable_unchecked<1>();
            py::ssize_t i = 0;
            for (const WaferRecord& r : set->records()) {
                py::array_t<std::uint8_t> map({r.map.rows(), r.map.cols()});
                const auto bins = r.map.bins();
                std::copy(bins.begin(), bins.end(), map.mutable_data());
                maps.append(map);
                truth.append(r.truth == Pattern::unknown ? std::string("unknown")
                                                         : name_of(r.truth));
                split.append(std::string(split_name(r.split)));
                w(i) = r.wafer_id;
                l(i) = r.lot_id;
                t(i) = r.tested_at_us;
                ++i;
            }
            py::dict out;
            out["maps"] = maps;
            out["truth"] = truth;
            out["split"] = split;
            out["wafer_id"] = wafer;
            out["lot_id"] = lot;
            out["tested_at_us"] = tested_at;
            return out;
        },
        py::arg("path"),
        "A .wmap file (tools/export_maps.py): maps as a list of 2-D uint8 arrays (copies; maps "
        "differ in size), and per-wafer truth, split, ids and sort time.");
}
