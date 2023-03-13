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

#include "record_writer.h"
#include <string>
#include <stdio.h>

#include "coding.h"
#include "crc32c.h"

#include <iostream>

using namespace std;

namespace tfrecord {

void RecordWriter::WriteRecord(const std::string& f_dir, const char* data, int size) {
  // Format of a single record:
  //  uint64    length
  //  uint32    masked crc of length
  //  byte      data[length]
  //  uint32    masked crc of data
  FILE* f_ptr = fopen(f_dir.c_str(), "w+");
  char header[kHeaderSize];
  char footer[kFooterSize];
  PopulateHeader(header, data, size);
  PopulateFooter(footer, data, size);
  fwrite(header, sizeof(header), 1, f_ptr);
  fwrite(data, size, 1, f_ptr);
  fwrite(footer, sizeof(footer), 1, f_ptr);
  fclose(f_ptr);
}


}  // namespace tfrecord
