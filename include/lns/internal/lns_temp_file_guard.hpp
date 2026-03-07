#pragma once

#include <filesystem>
#include <vector>

namespace lns_temp_file {

class ScopedPathCleanup {
 public:
  explicit ScopedPathCleanup(bool enabled = true) : enabled_(enabled) {}

  ScopedPathCleanup(const ScopedPathCleanup&) = delete;
  ScopedPathCleanup& operator=(const ScopedPathCleanup&) = delete;

  ~ScopedPathCleanup() noexcept { cleanup(); }

  void add(const std::filesystem::path& path) { paths_.push_back(path); }

  void setEnabled(bool enabled) { enabled_ = enabled; }

  void release() { enabled_ = false; }

 private:
  void cleanup() noexcept {
    if (!enabled_) {
      return;
    }
    std::error_code removeEc;
    for (const auto& path : paths_) {
      std::filesystem::remove(path, removeEc);
    }
  }

  bool enabled_ = true;
  std::vector<std::filesystem::path> paths_;
};

}  // namespace lns_temp_file
