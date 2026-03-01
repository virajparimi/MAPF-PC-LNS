#pragma once

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace lns_mapfpc_internal {

inline std::optional<std::filesystem::path> resolveTaskAssignmentExecutable() {
  namespace fs = std::filesystem;
  std::vector<fs::path> candidates;

  if (const char* envExe = std::getenv("MAPF_PC_TASK_ASSIGNMENT_EXE");
      envExe != nullptr && envExe[0] != '\0') {
    candidates.emplace_back(envExe);
  }

  candidates.emplace_back("./MAPF-PC/build_local/bin/task_assignment");
  candidates.emplace_back("./MAPF-PC/build/bin/task_assignment");

  std::error_code ec;
  const fs::path selfExe = fs::read_symlink("/proc/self/exe", ec);
  if (!ec && !selfExe.empty()) {
    const fs::path projectRoot = selfExe.parent_path().parent_path();
    if (!projectRoot.empty()) {
      candidates.emplace_back(projectRoot /
                              "MAPF-PC/build_local/bin/task_assignment");
      candidates.emplace_back(projectRoot / "MAPF-PC/build/bin/task_assignment");
    }
  }

  for (const auto& candidate : candidates) {
    std::error_code candidateEc;
    if (candidate.empty() || !fs::exists(candidate, candidateEc) || candidateEc ||
        fs::is_directory(candidate, candidateEc)) {
      continue;
    }
    return fs::absolute(candidate, candidateEc);
  }
  return std::nullopt;
}

inline std::optional<int> parsePositiveEnvInt(const char* varName) {
  if (const char* raw = std::getenv(varName); raw != nullptr && raw[0] != '\0') {
    try {
      const int value = std::stoi(raw);
      if (value > 0) {
        return value;
      }
    } catch (...) {
    }
  }
  return std::nullopt;
}

inline std::optional<std::string> normalizeMapfpcSolverVariant(
    const std::string& solverVariant) {
  if (solverVariant == "sota_cbs" || solverVariant == "cbs" ||
      solverVariant == "CBS") {
    return std::string("CBS");
  }
  if (solverVariant == "sota_pbs" || solverVariant == "pbs" ||
      solverVariant == "PBS") {
    return std::string("PBS");
  }
  return std::nullopt;
}

inline uint64_t hashAssignments(
    const std::vector<std::vector<int>>& assignments) {
  uint64_t h = 1469598103934665603ULL;
  const uint64_t kPrime = 1099511628211ULL;
  auto mix = [&](uint64_t v) {
    h ^= v;
    h *= kPrime;
  };
  for (int agent = 0; agent < static_cast<int>(assignments.size()); agent++) {
    mix(static_cast<uint64_t>(agent) + 0x9e3779b97f4a7c15ULL);
    mix(0xfeedfaceULL);
    for (int task : assignments[agent]) {
      mix(static_cast<uint64_t>(static_cast<uint32_t>(task)));
    }
    mix(0xdeadbeefULL);
  }
  return h;
}

}  // namespace lns_mapfpc_internal
