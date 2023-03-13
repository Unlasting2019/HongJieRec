/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <fstream>

#include "coding.h"
#include "crc32c.h"
// A macro to disallow the copy constructor and operator= functions.
// This should be used in the private: declarations for a class.
#ifndef DISALLOW_COPY_AND_ASSIGN
#define DISALLOW_COPY_AND_ASSIGN(TypeName) \
  TypeName(const TypeName &) = delete;     \
  void operator=(const TypeName &) = delete
#endif // #ifndef DISALLOW_COPY_AND_ASSIGN


namespace tfrecord {

class RecordWriter {
 public:
  // Format of a single record:
  //  uint64_t    length
  //  uint32_t    masked crc of length
  //  byte      data[length]
  //  uint32_t    masked crc of data
  static constexpr size_t kHeaderSize = sizeof(uint64_t) + sizeof(uint32_t);
  static constexpr size_t kFooterSize = sizeof(uint32_t);

  void WriteRecord(const std::string& dest, const char* data, int size);
  explicit RecordWriter() {}

  // Flushes any buffered data held by underlying containers of the
  // RecordWriter to the WritableFile. Does *not* flush the
  // WritableFile.
  //int Flush();

  // Writes all output to the file. Does *not* close the WritableFile.
  //
  // After calling Close(), any further calls to `WriteRecord()` or `Flush()`
  // are invalid.
  //int Close();

  // Utility method to populate TFRecord headers.  Populates record-header in
  // "header[0,kHeaderSize-1]".  The record-header is based on data[0, n-1].
  inline static void PopulateHeader(char* header, const char* data, size_t n);

  // Utility method to populate TFRecord footers.  Populates record-footer in
  // "footer[0,kFooterSize-1]".  The record-footer is based on data[0, n-1].
  inline static void PopulateFooter(char* footer, const char* data, size_t n);

 private:

  inline static uint32_t MaskedCrc(const char* data, size_t n) {
    return crc32c::Mask(crc32c::Value(data, n));
  }

  DISALLOW_COPY_AND_ASSIGN(RecordWriter);
};

void RecordWriter::PopulateHeader(char* header, const char*, size_t n) {
  EncodeFixed64(header + 0, n);
  EncodeFixed32(header + sizeof(uint64_t),
                      MaskedCrc(header, sizeof(uint64_t)));
}

void RecordWriter::PopulateFooter(char* footer, const char* data, size_t n) {
  EncodeFixed32(footer, MaskedCrc(data, n));
}

}  // namespace tfrecord

