#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

struct Event {
    int x = 0;
    int y = 0;
    int polarity = 0;
    std::int64_t timestamp = 0;
};

struct Vec3 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

struct Mat3 {
    std::array<std::array<double, 3>, 3> m{};
};

struct Intrinsics {
    double fx = 250.0;
    double fy = 250.0;
    double cx = 160.0;
    double cy = 160.0;
};

struct StereoGeometry {
    Intrinsics master_intrinsics{};
    Intrinsics slave_intrinsics{};

    // Rotation from slave camera frame to master camera frame.
    Mat3 R_slave_to_master{{
        std::array<double, 3>{1.0, 0.0, 0.0},
        std::array<double, 3>{0.0, 1.0, 0.0},
        std::array<double, 3>{0.0, 0.0, 1.0},
    }};

    // Slave camera center position in master camera frame [m].
    Vec3 t_slave_in_master{0.10, 0.0, 0.0};

    // Matching gate in pixel units.
    int max_vertical_diff_px = 2;
};

constexpr StereoGeometry kStereoGeometry{};

Vec3 add(const Vec3& a, const Vec3& b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

Vec3 sub(const Vec3& a, const Vec3& b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

Vec3 mul(const Vec3& v, double s) {
    return {v.x * s, v.y * s, v.z * s};
}

double dot(const Vec3& a, const Vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

double norm(const Vec3& v) {
    return std::sqrt(dot(v, v));
}

Vec3 normalize(const Vec3& v) {
    const double n = norm(v);
    if (n < 1e-12) {
        return {0.0, 0.0, 0.0};
    }
    return mul(v, 1.0 / n);
}

Vec3 matmul(const Mat3& m, const Vec3& v) {
    return {
        m.m[0][0] * v.x + m.m[0][1] * v.y + m.m[0][2] * v.z,
        m.m[1][0] * v.x + m.m[1][1] * v.y + m.m[1][2] * v.z,
        m.m[2][0] * v.x + m.m[2][1] * v.y + m.m[2][2] * v.z,
    };
}

bool parse_event_line(const std::string& line, Event& out) {
    if (line.empty() || line[0] == '%') {
        return false;
    }

    std::stringstream ss(line);
    std::string field;
    std::array<std::string, 4> fields{};

    for (int i = 0; i < 4; ++i) {
        if (!std::getline(ss, field, ',')) {
            return false;
        }
        fields[static_cast<std::size_t>(i)] = field;
    }

    try {
        out.x = std::stoi(fields[0]);
        out.y = std::stoi(fields[1]);
        out.polarity = std::stoi(fields[2]);
        out.timestamp = std::stoll(fields[3]);
    } catch (const std::exception&) {
        return false;
    }
    return true;
}

std::vector<Event> read_events(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs) {
        throw std::runtime_error("Failed to open: " + path);
    }

    std::vector<Event> events;
    std::string line;
    while (std::getline(ifs, line)) {
        Event e;
        if (parse_event_line(line, e)) {
            events.push_back(e);
        }
    }
    return events;
}

Vec3 pixel_to_ray_in_camera(int x, int y, const Intrinsics& intr) {
    const double nx = (static_cast<double>(x) - intr.cx) / intr.fx;
    const double ny = (static_cast<double>(y) - intr.cy) / intr.fy;
    return normalize(Vec3{nx, ny, 1.0});
}

bool triangulate_two_rays(
    const Vec3& C1,
    const Vec3& d1,
    const Vec3& C2,
    const Vec3& d2,
    Vec3& out_point,
    double& out_line_distance,
    double& out_lambda1,
    double& out_lambda2) {
    const Vec3 w0 = sub(C1, C2);
    const double a = dot(d1, d1);
    const double b = dot(d1, d2);
    const double c = dot(d2, d2);
    const double d = dot(d1, w0);
    const double e = dot(d2, w0);
    const double denom = a * c - b * b;

    if (std::abs(denom) < 1e-12) {
        return false;
    }

    const double lambda1 = (b * e - c * d) / denom;
    const double lambda2 = (a * e - b * d) / denom;

    const Vec3 P1 = add(C1, mul(d1, lambda1));
    const Vec3 P2 = add(C2, mul(d2, lambda2));

    out_point = mul(add(P1, P2), 0.5);
    out_line_distance = norm(sub(P1, P2));
    out_lambda1 = lambda1;
    out_lambda2 = lambda2;
    return true;
}

struct ReconstructionPoint {
    std::int64_t timestamp = 0;
    int polarity = 0;
    Vec3 point{};
    double line_distance = 0.0;
};

std::vector<ReconstructionPoint> reconstruct(
    const std::vector<Event>& master_events,
    const std::vector<Event>& slave_events,
    const StereoGeometry& geometry) {
    std::unordered_map<std::int64_t, std::vector<Event>> slave_by_ts;
    slave_by_ts.reserve(slave_events.size());
    for (const auto& e : slave_events) {
        slave_by_ts[e.timestamp].push_back(e);
    }

    std::vector<ReconstructionPoint> points;
    points.reserve(master_events.size() / 4);

    const Vec3 C_master{0.0, 0.0, 0.0};
    const Vec3 C_slave = geometry.t_slave_in_master;

    for (const auto& em : master_events) {
        const auto it = slave_by_ts.find(em.timestamp);
        if (it == slave_by_ts.end()) {
            continue;
        }

        const auto& candidates = it->second;
        int best_idx = -1;
        int best_score = std::numeric_limits<int>::max();

        for (int i = 0; i < static_cast<int>(candidates.size()); ++i) {
            const Event& es = candidates[static_cast<std::size_t>(i)];
            if (es.polarity != em.polarity) {
                continue;
            }
            const int dy = std::abs(es.y - em.y);
            if (dy > geometry.max_vertical_diff_px) {
                continue;
            }
            if (dy < best_score) {
                best_score = dy;
                best_idx = i;
            }
        }

        if (best_idx < 0) {
            continue;
        }

        const Event& es = candidates[static_cast<std::size_t>(best_idx)];

        const Vec3 ray_master = pixel_to_ray_in_camera(
            em.x,
            em.y,
            geometry.master_intrinsics);
        const Vec3 ray_slave_cam = pixel_to_ray_in_camera(
            es.x,
            es.y,
            geometry.slave_intrinsics);
        const Vec3 ray_slave_in_master = normalize(matmul(
            geometry.R_slave_to_master,
            ray_slave_cam));

        Vec3 p{};
        double line_dist = 0.0;
        double lambda_master = 0.0;
        double lambda_slave = 0.0;

        if (!triangulate_two_rays(
                C_master,
                ray_master,
                C_slave,
                ray_slave_in_master,
                p,
                line_dist,
                lambda_master,
                lambda_slave)) {
            continue;
        }

        // Keep only points that lie in front of both cameras.
        if (lambda_master <= 0.0 || lambda_slave <= 0.0) {
            continue;
        }

        points.push_back(ReconstructionPoint{
            em.timestamp,
            em.polarity,
            p,
            line_dist,
        });
    }

    return points;
}

void write_points_csv(
    const std::string& path,
    const std::vector<ReconstructionPoint>& points) {
    std::ofstream ofs(path);
    if (!ofs) {
        throw std::runtime_error("Failed to write: " + path);
    }

    ofs << "X,Y,Z,timestamp,polarity,ray_distance\n";
    ofs << std::fixed << std::setprecision(6);
    for (const auto& p : points) {
        ofs << p.point.x << ','
            << p.point.y << ','
            << p.point.z << ','
            << p.timestamp << ','
            << p.polarity << ','
            << p.line_distance << '\n';
    }
}

}  // namespace

int main() {
    try {
        const std::string master_path = "events_master.csv";
        const std::string slave_path = "events_slave.csv";
        const std::string output_path = "points3d.csv";

        const auto master_events = read_events(master_path);
        const auto slave_events = read_events(slave_path);

        const auto points = reconstruct(
            master_events,
            slave_events,
            kStereoGeometry);

        write_points_csv(output_path, points);

        std::cout << "Read master events: " << master_events.size() << '\n';
        std::cout << "Read slave events: " << slave_events.size() << '\n';
        std::cout << "Reconstructed points: " << points.size() << '\n';
        std::cout << "Output: " << output_path << '\n';

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}
