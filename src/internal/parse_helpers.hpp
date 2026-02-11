#pragma once

#include <sstream>
#include <string>

namespace parse_helpers {

inline bool parseIntStrict(const std::string& token, int& out) {
  std::istringstream iss(token);
  int value = 0;
  char extra = '\0';
  if (!(iss >> value)) {
    return false;
  }
  if (iss >> extra) {
    return false;
  }
  out = value;
  return true;
}

template <typename Iter>
inline bool parseNextInt(Iter& it, const Iter& end, int& out) {
  if (it == end) {
    return false;
  }
  const std::string token = *it;
  if (!parseIntStrict(token, out)) {
    return false;
  }
  ++it;
  return true;
}

}  // namespace parse_helpers
