// Print array offsets for a sealed 3.2.7 MJB file using its serialization schema.
// No simulator library is linked. The caller authenticates the input before use.
#include <mujoco/mjmodel.h>
#include <mujoco/mjxmacro.h>
#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  std::ifstream file(argv[1], std::ios::binary | std::ios::ate);
  file.exceptions(std::ios::failbit | std::ios::badbit);
  const auto length = static_cast<std::uint64_t>(file.tellg());
  file.seekg(0);
  std::array<int, 5> header;
  file.read(reinterpret_cast<char*>(header.data()), sizeof(header));
  if (header != std::array<int,5>{54321,8,79,2,383}) return 3;
  mjModel model{};
  auto* m = &model;
#define X(name) file.read(reinterpret_cast<char*>(&m->name), sizeof(m->name));
  MJMODEL_INTS
#undef X
  std::uint64_t offset = static_cast<std::uint64_t>(file.tellg())
      + sizeof(mjOption) + sizeof(mjVisual) + sizeof(mjStatistic);
  if (offset > length) return 4;
  MJMODEL_POINTERS_PREAMBLE(m)
#define X(type, name, nr, nc) { \
  const std::uint64_t rows = m->nr, cols = nc; \
  if (rows && cols > (length-offset)/sizeof(type)/rows) return 5; \
  std::cout << #name << " " << #type << " " << offset << " " << rows << " " << cols << "\n"; \
  offset += rows * cols * sizeof(type); \
}
  MJMODEL_POINTERS
#undef X
  return offset == length ? 0 : 6;
}
