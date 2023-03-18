#ifndef FASTFLOAT_FAST_FLOAT_H
#define FASTFLOAT_FAST_FLOAT_H

#include <system_error>

namespace fast_float {
enum chars_format {
    scientific = 1<<0,
    fixed = 1<<2,
    hex = 1<<3,
    general = fixed | scientific
};


struct from_chars_result {
  const char *ptr;
  std::errc ec;
};

struct parse_options {
  constexpr explicit parse_options(chars_format fmt = chars_format::general,
                         char dot = '.')
    : format(fmt), decimal_point(dot) {}

  chars_format format;
  char decimal_point;
};

template<typename T>
from_chars_result from_chars(const char *first, const char *last,
                             T &value, chars_format fmt = chars_format::general)  noexcept;

template<typename T>
from_chars_result from_chars_advanced(const char *first, const char *last,
                                      T &value, parse_options options)  noexcept;

}
#include "parse_number.h"
#endif // FASTFLOAT_FAST_FLOAT_H
