#pragma once

#include <rapidjson/document.h>

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string_view>

namespace jets::histogram_encoding {

inline constexpr std::string_view binary32_le = "base64-ieee754-binary32-le-v1";

inline constexpr auto base64_digits = [] {
  std::array<unsigned char, 256> digits{};
  digits.fill(255);
  constexpr std::string_view alphabet =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  for (std::size_t i = 0; i < alphabet.size(); ++i) {
    digits[static_cast<unsigned char>(alphabet[i])] = static_cast<unsigned char>(i);
  }
  return digits;
}();

inline unsigned base64_digit(char value) {
  const unsigned digit = base64_digits[static_cast<unsigned char>(value)];
  if (digit >= 64) throw std::invalid_argument("invalid packed histogram base64 digit");
  return digit;
}

inline std::string_view packed_binary32_data(const rapidjson::Value& value, std::string_view histogram_class,
                                              std::size_t cells) {
  if (histogram_class != "TH1F" || !value.IsObject() || value.MemberCount() != 3 ||
      !value.HasMember("encoding") || !value.HasMember("count") || !value.HasMember("data")) {
    throw std::invalid_argument("packed float cells require a TH1F and a three-field encoding object");
  }
  const auto& encoding = value["encoding"];
  const auto& count = value["count"];
  const auto& data = value["data"];
  if (!encoding.IsString() ||
      std::string_view(encoding.GetString(), encoding.GetStringLength()) != binary32_le ||
      !count.IsUint64() || count.GetUint64() != cells || !data.IsString()) {
    throw std::invalid_argument("invalid packed histogram encoding, cell count or data type");
  }
  return std::string_view(data.GetString(), data.GetStringLength());
}

template <class Put>
void decode_binary32(const rapidjson::Value& value, std::string_view histogram_class, std::size_t cells, Put put) {
  static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);
  auto data = packed_binary32_data(value, histogram_class, cells);
  constexpr auto maximum = std::numeric_limits<std::size_t>::max();
  if (cells > (maximum - 2) / 4) throw std::invalid_argument("packed histogram byte count overflow");
  const std::size_t bytes = 4 * cells;
  if ((bytes + 2) / 3 > maximum / 4 || data.size() != 4 * ((bytes + 2) / 3)) {
    throw std::invalid_argument("packed histogram base64 length does not match its cell count");
  }
  std::size_t cell = 0;
  unsigned shift = 0;
  std::uint32_t word = 0;
  const auto emit_byte = [&](std::uint32_t byte) {
    word |= byte << shift;
    shift += 8;
    if (shift == 32) {
      put(cell++, word);
      shift = 0;
      word = 0;
    }
  };
  const std::size_t full = (bytes / 3) * 4;
  for (std::size_t i = 0; i < full; i += 4) {
    const unsigned a = base64_digit(data[i]);
    const unsigned b = base64_digit(data[i + 1]);
    const unsigned c = base64_digit(data[i + 2]);
    const unsigned d = base64_digit(data[i + 3]);
    emit_byte((a << 2) | (b >> 4));
    emit_byte(((b & 15) << 4) | (c >> 2));
    emit_byte(((c & 3) << 6) | d);
  }
  const auto tail = bytes % 3;
  if (tail != 0) {
    const unsigned a = base64_digit(data[full]);
    const unsigned b = base64_digit(data[full + 1]);
    if (data[full + 3] != '=') throw std::invalid_argument("invalid packed histogram base64 padding");
    if (tail == 1) {
      if (data[full + 2] != '=' || (b & 15) != 0)
        throw std::invalid_argument("invalid packed histogram base64 padding bits");
    } else {
      const unsigned c = base64_digit(data[full + 2]);
      if ((c & 3) != 0) throw std::invalid_argument("invalid packed histogram base64 padding bits");
      emit_byte((a << 2) | (b >> 4));
      emit_byte(((b & 15) << 4) | (c >> 2));
      return;
    }
    emit_byte((a << 2) | (b >> 4));
  }
}

inline double finite_binary32(std::uint32_t word) {
  // The decimal histogram loaders accept finite cells. Check the raw exponent
  // before promotion so even a signaling NaN cannot affect the FP environment.
  if ((word & 0x7f800000u) == 0x7f800000u)
    throw std::invalid_argument("expected finite packed histogram cell");
  return std::bit_cast<float>(word);
}

}  // namespace jets::histogram_encoding
