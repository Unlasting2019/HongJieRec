// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: example.proto

#include "example.pb.h"

#include <algorithm>

#include <google/protobuf/stubs/common.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/extension_set.h>
#include <google/protobuf/wire_format_lite.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/generated_message_reflection.h>
#include <google/protobuf/reflection_ops.h>
#include <google/protobuf/wire_format.h>
// @@protoc_insertion_point(includes)
#include <google/protobuf/port_def.inc>
extern PROTOBUF_INTERNAL_EXPORT_feature_2eproto ::PROTOBUF_NAMESPACE_ID::internal::SCCInfo<1> scc_info_FeatureLists_feature_2eproto;
extern PROTOBUF_INTERNAL_EXPORT_feature_2eproto ::PROTOBUF_NAMESPACE_ID::internal::SCCInfo<1> scc_info_Features_feature_2eproto;
namespace tensorflow {
class ExampleDefaultTypeInternal {
 public:
  ::PROTOBUF_NAMESPACE_ID::internal::ExplicitlyConstructed<Example> _instance;
} _Example_default_instance_;
class SequenceExampleDefaultTypeInternal {
 public:
  ::PROTOBUF_NAMESPACE_ID::internal::ExplicitlyConstructed<SequenceExample> _instance;
} _SequenceExample_default_instance_;
}  // namespace tensorflow
static void InitDefaultsscc_info_Example_example_2eproto() {
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  {
    void* ptr = &::tensorflow::_Example_default_instance_;
    new (ptr) ::tensorflow::Example();
    ::PROTOBUF_NAMESPACE_ID::internal::OnShutdownDestroyMessage(ptr);
  }
  ::tensorflow::Example::InitAsDefaultInstance();
}

::PROTOBUF_NAMESPACE_ID::internal::SCCInfo<1> scc_info_Example_example_2eproto =
    {{ATOMIC_VAR_INIT(::PROTOBUF_NAMESPACE_ID::internal::SCCInfoBase::kUninitialized), 1, InitDefaultsscc_info_Example_example_2eproto}, {
      &scc_info_Features_feature_2eproto.base,}};

static void InitDefaultsscc_info_SequenceExample_example_2eproto() {
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  {
    void* ptr = &::tensorflow::_SequenceExample_default_instance_;
    new (ptr) ::tensorflow::SequenceExample();
    ::PROTOBUF_NAMESPACE_ID::internal::OnShutdownDestroyMessage(ptr);
  }
  ::tensorflow::SequenceExample::InitAsDefaultInstance();
}

::PROTOBUF_NAMESPACE_ID::internal::SCCInfo<2> scc_info_SequenceExample_example_2eproto =
    {{ATOMIC_VAR_INIT(::PROTOBUF_NAMESPACE_ID::internal::SCCInfoBase::kUninitialized), 2, InitDefaultsscc_info_SequenceExample_example_2eproto}, {
      &scc_info_Features_feature_2eproto.base,
      &scc_info_FeatureLists_feature_2eproto.base,}};

static ::PROTOBUF_NAMESPACE_ID::Metadata file_level_metadata_example_2eproto[2];
static constexpr ::PROTOBUF_NAMESPACE_ID::EnumDescriptor const** file_level_enum_descriptors_example_2eproto = nullptr;
static constexpr ::PROTOBUF_NAMESPACE_ID::ServiceDescriptor const** file_level_service_descriptors_example_2eproto = nullptr;

const ::PROTOBUF_NAMESPACE_ID::uint32 TableStruct_example_2eproto::offsets[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  ~0u,  // no _has_bits_
  PROTOBUF_FIELD_OFFSET(::tensorflow::Example, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  PROTOBUF_FIELD_OFFSET(::tensorflow::Example, features_),
  ~0u,  // no _has_bits_
  PROTOBUF_FIELD_OFFSET(::tensorflow::SequenceExample, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  PROTOBUF_FIELD_OFFSET(::tensorflow::SequenceExample, context_),
  PROTOBUF_FIELD_OFFSET(::tensorflow::SequenceExample, feature_lists_),
};
static const ::PROTOBUF_NAMESPACE_ID::internal::MigrationSchema schemas[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  { 0, -1, sizeof(::tensorflow::Example)},
  { 6, -1, sizeof(::tensorflow::SequenceExample)},
};

static ::PROTOBUF_NAMESPACE_ID::Message const * const file_default_instances[] = {
  reinterpret_cast<const ::PROTOBUF_NAMESPACE_ID::Message*>(&::tensorflow::_Example_default_instance_),
  reinterpret_cast<const ::PROTOBUF_NAMESPACE_ID::Message*>(&::tensorflow::_SequenceExample_default_instance_),
};

const char descriptor_table_protodef_example_2eproto[] =
  "\n\rexample.proto\022\ntensorflow\032\rfeature.pro"
  "to\"1\n\007Example\022&\n\010features\030\001 \001(\0132\024.tensor"
  "flow.Features\"i\n\017SequenceExample\022%\n\007cont"
  "ext\030\001 \001(\0132\024.tensorflow.Features\022/\n\rfeatu"
  "re_lists\030\002 \001(\0132\030.tensorflow.FeatureLists"
  "Bi\n\026org.tensorflow.exampleB\rExampleProto"
  "sP\001Z;github.com/tensorflow/tensorflow/te"
  "nsorflow/go/core/example\370\001\001b\006proto3"
  ;
static const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable*const descriptor_table_example_2eproto_deps[1] = {
  &::descriptor_table_feature_2eproto,
};
static ::PROTOBUF_NAMESPACE_ID::internal::SCCInfoBase*const descriptor_table_example_2eproto_sccs[2] = {
  &scc_info_Example_example_2eproto.base,
  &scc_info_SequenceExample_example_2eproto.base,
};
static ::PROTOBUF_NAMESPACE_ID::internal::once_flag descriptor_table_example_2eproto_once;
static bool descriptor_table_example_2eproto_initialized = false;
const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_example_2eproto = {
  &descriptor_table_example_2eproto_initialized, descriptor_table_protodef_example_2eproto, "example.proto", 315,
  &descriptor_table_example_2eproto_once, descriptor_table_example_2eproto_sccs, descriptor_table_example_2eproto_deps, 2, 1,
  schemas, file_default_instances, TableStruct_example_2eproto::offsets,
  file_level_metadata_example_2eproto, 2, file_level_enum_descriptors_example_2eproto, file_level_service_descriptors_example_2eproto,
};

// Force running AddDescriptors() at dynamic initialization time.
static bool dynamic_init_dummy_example_2eproto = (  ::PROTOBUF_NAMESPACE_ID::internal::AddDescriptors(&descriptor_table_example_2eproto), true);
namespace tensorflow {

// ===================================================================

void Example::InitAsDefaultInstance() {
  ::tensorflow::_Example_default_instance_._instance.get_mutable()->features_ = const_cast< ::tensorflow::Features*>(
      ::tensorflow::Features::internal_default_instance());
}
class Example::HasBitSetters {
 public:
  static const ::tensorflow::Features& features(const Example* msg);
};

const ::tensorflow::Features&
Example::HasBitSetters::features(const Example* msg) {
  return *msg->features_;
}
void Example::unsafe_arena_set_allocated_features(
    ::tensorflow::Features* features) {
  if (GetArenaNoVirtual() == nullptr) {
    delete features_;
  }
  features_ = features;
  if (features) {
    
  } else {
    
  }
  // @@protoc_insertion_point(field_unsafe_arena_set_allocated:tensorflow.Example.features)
}
void Example::clear_features() {
  if (GetArenaNoVirtual() == nullptr && features_ != nullptr) {
    delete features_;
  }
  features_ = nullptr;
}
#if !defined(_MSC_VER) || _MSC_VER >= 1900
const int Example::kFeaturesFieldNumber;
#endif  // !defined(_MSC_VER) || _MSC_VER >= 1900

Example::Example()
  : ::PROTOBUF_NAMESPACE_ID::Message(), _internal_metadata_(nullptr) {
  SharedCtor();
  // @@protoc_insertion_point(constructor:tensorflow.Example)
}
Example::Example(::PROTOBUF_NAMESPACE_ID::Arena* arena)
  : ::PROTOBUF_NAMESPACE_ID::Message(),
  _internal_metadata_(arena) {
  SharedCtor();
  RegisterArenaDtor(arena);
  // @@protoc_insertion_point(arena_constructor:tensorflow.Example)
}
Example::Example(const Example& from)
  : ::PROTOBUF_NAMESPACE_ID::Message(),
      _internal_metadata_(nullptr) {
  _internal_metadata_.MergeFrom(from._internal_metadata_);
  if (from.has_features()) {
    features_ = new ::tensorflow::Features(*from.features_);
  } else {
    features_ = nullptr;
  }
  // @@protoc_insertion_point(copy_constructor:tensorflow.Example)
}

void Example::SharedCtor() {
  ::PROTOBUF_NAMESPACE_ID::internal::InitSCC(&scc_info_Example_example_2eproto.base);
  features_ = nullptr;
}

Example::~Example() {
  // @@protoc_insertion_point(destructor:tensorflow.Example)
  SharedDtor();
}

void Example::SharedDtor() {
  GOOGLE_DCHECK(GetArenaNoVirtual() == nullptr);
  if (this != internal_default_instance()) delete features_;
}

void Example::ArenaDtor(void* object) {
  Example* _this = reinterpret_cast< Example* >(object);
  (void)_this;
}
void Example::RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena*) {
}
void Example::SetCachedSize(int size) const {
  _cached_size_.Set(size);
}
const Example& Example::default_instance() {
  ::PROTOBUF_NAMESPACE_ID::internal::InitSCC(&::scc_info_Example_example_2eproto.base);
  return *internal_default_instance();
}


void Example::Clear() {
// @@protoc_insertion_point(message_clear_start:tensorflow.Example)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  if (GetArenaNoVirtual() == nullptr && features_ != nullptr) {
    delete features_;
  }
  features_ = nullptr;
  _internal_metadata_.Clear();
}

#if GOOGLE_PROTOBUF_ENABLE_EXPERIMENTAL_PARSER
const char* Example::_InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) {
#define CHK_(x) if (PROTOBUF_PREDICT_FALSE(!(x))) goto failure
  ::PROTOBUF_NAMESPACE_ID::Arena* arena = GetArenaNoVirtual(); (void)arena;
  while (!ctx->Done(&ptr)) {
    ::PROTOBUF_NAMESPACE_ID::uint32 tag;
    ptr = ::PROTOBUF_NAMESPACE_ID::internal::ReadTag(ptr, &tag);
    CHK_(ptr);
    switch (tag >> 3) {
      // .tensorflow.Features features = 1;
      case 1:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 10)) {
          ptr = ctx->ParseMessage(mutable_features(), ptr);
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      default: {
      handle_unusual:
        if ((tag & 7) == 4 || tag == 0) {
          ctx->SetLastTag(tag);
          goto success;
        }
        ptr = UnknownFieldParse(tag, &_internal_metadata_, ptr, ctx);
        CHK_(ptr != nullptr);
        continue;
      }
    }  // switch
  }  // while
success:
  return ptr;
failure:
  ptr = nullptr;
  goto success;
#undef CHK_
}
#else  // GOOGLE_PROTOBUF_ENABLE_EXPERIMENTAL_PARSER
bool Example::MergePartialFromCodedStream(
    ::PROTOBUF_NAMESPACE_ID::io::CodedInputStream* input) {
#define DO_(EXPRESSION) if (!PROTOBUF_PREDICT_TRUE(EXPRESSION)) goto failure
  ::PROTOBUF_NAMESPACE_ID::uint32 tag;
  // @@protoc_insertion_point(parse_start:tensorflow.Example)
  for (;;) {
    ::std::pair<::PROTOBUF_NAMESPACE_ID::uint32, bool> p = input->ReadTagWithCutoffNoLastTag(127u);
    tag = p.first;
    if (!p.second) goto handle_unusual;
    switch (::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::GetTagFieldNumber(tag)) {
      // .tensorflow.Features features = 1;
      case 1: {
        if (static_cast< ::PROTOBUF_NAMESPACE_ID::uint8>(tag) == (10 & 0xFF)) {
          DO_(::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::ReadMessage(
               input, mutable_features()));
        } else {
          goto handle_unusual;
        }
        break;
      }

      default: {
      handle_unusual:
        if (tag == 0) {
          goto success;
        }
        DO_(::PROTOBUF_NAMESPACE_ID::internal::WireFormat::SkipField(
              input, tag, _internal_metadata_.mutable_unknown_fields()));
        break;
      }
    }
  }
success:
  // @@protoc_insertion_point(parse_success:tensorflow.Example)
  return true;
failure:
  // @@protoc_insertion_point(parse_failure:tensorflow.Example)
  return false;
#undef DO_
}
#endif  // GOOGLE_PROTOBUF_ENABLE_EXPERIMENTAL_PARSER

void Example::SerializeWithCachedSizes(
    ::PROTOBUF_NAMESPACE_ID::io::CodedOutputStream* output) const {
  // @@protoc_insertion_point(serialize_start:tensorflow.Example)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  // .tensorflow.Features features = 1;
  if (this->has_features()) {
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteMessageMaybeToArray(
      1, HasBitSetters::features(this), output);
  }

  if (_internal_metadata_.have_unknown_fields()) {
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::SerializeUnknownFields(
        _internal_metadata_.unknown_fields(), output);
  }
  // @@protoc_insertion_point(serialize_end:tensorflow.Example)
}

::PROTOBUF_NAMESPACE_ID::uint8* Example::InternalSerializeWithCachedSizesToArray(
    ::PROTOBUF_NAMESPACE_ID::uint8* target) const {
  // @@protoc_insertion_point(serialize_to_array_start:tensorflow.Example)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  // .tensorflow.Features features = 1;
  if (this->has_features()) {
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::
      InternalWriteMessageToArray(
        1, HasBitSetters::features(this), target);
  }

  if (_internal_metadata_.have_unknown_fields()) {
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::SerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields(), target);
  }
  // @@protoc_insertion_point(serialize_to_array_end:tensorflow.Example)
  return target;
}

size_t Example::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:tensorflow.Example)
  size_t total_size = 0;

  if (_internal_metadata_.have_unknown_fields()) {
    total_size +=
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::ComputeUnknownFieldsSize(
        _internal_metadata_.unknown_fields());
  }
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  // .tensorflow.Features features = 1;
  if (this->has_features()) {
    total_size += 1 +
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::MessageSize(
        *features_);
  }

  int cached_size = ::PROTOBUF_NAMESPACE_ID::internal::ToCachedSize(total_size);
  SetCachedSize(cached_size);
  return total_size;
}

void Example::MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) {
// @@protoc_insertion_point(generalized_merge_from_start:tensorflow.Example)
  GOOGLE_DCHECK_NE(&from, this);
  const Example* source =
      ::PROTOBUF_NAMESPACE_ID::DynamicCastToGenerated<Example>(
          &from);
  if (source == nullptr) {
  // @@protoc_insertion_point(generalized_merge_from_cast_fail:tensorflow.Example)
    ::PROTOBUF_NAMESPACE_ID::internal::ReflectionOps::Merge(from, this);
  } else {
  // @@protoc_insertion_point(generalized_merge_from_cast_success:tensorflow.Example)
    MergeFrom(*source);
  }
}

void Example::MergeFrom(const Example& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:tensorflow.Example)
  GOOGLE_DCHECK_NE(&from, this);
  _internal_metadata_.MergeFrom(from._internal_metadata_);
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  if (from.has_features()) {
    mutable_features()->::tensorflow::Features::MergeFrom(from.features());
  }
}

void Example::CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) {
// @@protoc_insertion_point(generalized_copy_from_start:tensorflow.Example)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

void Example::CopyFrom(const Example& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:tensorflow.Example)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool Example::IsInitialized() const {
  return true;
}

void Example::Swap(Example* other) {
  if (other == this) return;
  if (GetArenaNoVirtual() == other->GetArenaNoVirtual()) {
    InternalSwap(other);
  } else {
    Example* temp = New(GetArenaNoVirtual());
    temp->MergeFrom(*other);
    other->CopyFrom(*this);
    InternalSwap(temp);
    if (GetArenaNoVirtual() == nullptr) {
      delete temp;
    }
  }
}
void Example::UnsafeArenaSwap(Example* other) {
  if (other == this) return;
  GOOGLE_DCHECK(GetArenaNoVirtual() == other->GetArenaNoVirtual());
  InternalSwap(other);
}
void Example::InternalSwap(Example* other) {
  using std::swap;
  _internal_metadata_.Swap(&other->_internal_metadata_);
  swap(features_, other->features_);
}

::PROTOBUF_NAMESPACE_ID::Metadata Example::GetMetadata() const {
  return GetMetadataStatic();
}


// ===================================================================

void SequenceExample::InitAsDefaultInstance() {
  ::tensorflow::_SequenceExample_default_instance_._instance.get_mutable()->context_ = const_cast< ::tensorflow::Features*>(
      ::tensorflow::Features::internal_default_instance());
  ::tensorflow::_SequenceExample_default_instance_._instance.get_mutable()->feature_lists_ = const_cast< ::tensorflow::FeatureLists*>(
      ::tensorflow::FeatureLists::internal_default_instance());
}
class SequenceExample::HasBitSetters {
 public:
  static const ::tensorflow::Features& context(const SequenceExample* msg);
  static const ::tensorflow::FeatureLists& feature_lists(const SequenceExample* msg);
};

const ::tensorflow::Features&
SequenceExample::HasBitSetters::context(const SequenceExample* msg) {
  return *msg->context_;
}
const ::tensorflow::FeatureLists&
SequenceExample::HasBitSetters::feature_lists(const SequenceExample* msg) {
  return *msg->feature_lists_;
}
void SequenceExample::unsafe_arena_set_allocated_context(
    ::tensorflow::Features* context) {
  if (GetArenaNoVirtual() == nullptr) {
    delete context_;
  }
  context_ = context;
  if (context) {
    
  } else {
    
  }
  // @@protoc_insertion_point(field_unsafe_arena_set_allocated:tensorflow.SequenceExample.context)
}
void SequenceExample::clear_context() {
  if (GetArenaNoVirtual() == nullptr && context_ != nullptr) {
    delete context_;
  }
  context_ = nullptr;
}
void SequenceExample::unsafe_arena_set_allocated_feature_lists(
    ::tensorflow::FeatureLists* feature_lists) {
  if (GetArenaNoVirtual() == nullptr) {
    delete feature_lists_;
  }
  feature_lists_ = feature_lists;
  if (feature_lists) {
    
  } else {
    
  }
  // @@protoc_insertion_point(field_unsafe_arena_set_allocated:tensorflow.SequenceExample.feature_lists)
}
void SequenceExample::clear_feature_lists() {
  if (GetArenaNoVirtual() == nullptr && feature_lists_ != nullptr) {
    delete feature_lists_;
  }
  feature_lists_ = nullptr;
}
#if !defined(_MSC_VER) || _MSC_VER >= 1900
const int SequenceExample::kContextFieldNumber;
const int SequenceExample::kFeatureListsFieldNumber;
#endif  // !defined(_MSC_VER) || _MSC_VER >= 1900

SequenceExample::SequenceExample()
  : ::PROTOBUF_NAMESPACE_ID::Message(), _internal_metadata_(nullptr) {
  SharedCtor();
  // @@protoc_insertion_point(constructor:tensorflow.SequenceExample)
}
SequenceExample::SequenceExample(::PROTOBUF_NAMESPACE_ID::Arena* arena)
  : ::PROTOBUF_NAMESPACE_ID::Message(),
  _internal_metadata_(arena) {
  SharedCtor();
  RegisterArenaDtor(arena);
  // @@protoc_insertion_point(arena_constructor:tensorflow.SequenceExample)
}
SequenceExample::SequenceExample(const SequenceExample& from)
  : ::PROTOBUF_NAMESPACE_ID::Message(),
      _internal_metadata_(nullptr) {
  _internal_metadata_.MergeFrom(from._internal_metadata_);
  if (from.has_context()) {
    context_ = new ::tensorflow::Features(*from.context_);
  } else {
    context_ = nullptr;
  }
  if (from.has_feature_lists()) {
    feature_lists_ = new ::tensorflow::FeatureLists(*from.feature_lists_);
  } else {
    feature_lists_ = nullptr;
  }
  // @@protoc_insertion_point(copy_constructor:tensorflow.SequenceExample)
}

void SequenceExample::SharedCtor() {
  ::PROTOBUF_NAMESPACE_ID::internal::InitSCC(&scc_info_SequenceExample_example_2eproto.base);
  ::memset(&context_, 0, static_cast<size_t>(
      reinterpret_cast<char*>(&feature_lists_) -
      reinterpret_cast<char*>(&context_)) + sizeof(feature_lists_));
}

SequenceExample::~SequenceExample() {
  // @@protoc_insertion_point(destructor:tensorflow.SequenceExample)
  SharedDtor();
}

void SequenceExample::SharedDtor() {
  GOOGLE_DCHECK(GetArenaNoVirtual() == nullptr);
  if (this != internal_default_instance()) delete context_;
  if (this != internal_default_instance()) delete feature_lists_;
}

void SequenceExample::ArenaDtor(void* object) {
  SequenceExample* _this = reinterpret_cast< SequenceExample* >(object);
  (void)_this;
}
void SequenceExample::RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena*) {
}
void SequenceExample::SetCachedSize(int size) const {
  _cached_size_.Set(size);
}
const SequenceExample& SequenceExample::default_instance() {
  ::PROTOBUF_NAMESPACE_ID::internal::InitSCC(&::scc_info_SequenceExample_example_2eproto.base);
  return *internal_default_instance();
}


void SequenceExample::Clear() {
// @@protoc_insertion_point(message_clear_start:tensorflow.SequenceExample)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  if (GetArenaNoVirtual() == nullptr && context_ != nullptr) {
    delete context_;
  }
  context_ = nullptr;
  if (GetArenaNoVirtual() == nullptr && feature_lists_ != nullptr) {
    delete feature_lists_;
  }
  feature_lists_ = nullptr;
  _internal_metadata_.Clear();
}

#if GOOGLE_PROTOBUF_ENABLE_EXPERIMENTAL_PARSER
const char* SequenceExample::_InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) {
#define CHK_(x) if (PROTOBUF_PREDICT_FALSE(!(x))) goto failure
  ::PROTOBUF_NAMESPACE_ID::Arena* arena = GetArenaNoVirtual(); (void)arena;
  while (!ctx->Done(&ptr)) {
    ::PROTOBUF_NAMESPACE_ID::uint32 tag;
    ptr = ::PROTOBUF_NAMESPACE_ID::internal::ReadTag(ptr, &tag);
    CHK_(ptr);
    switch (tag >> 3) {
      // .tensorflow.Features context = 1;
      case 1:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 10)) {
          ptr = ctx->ParseMessage(mutable_context(), ptr);
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // .tensorflow.FeatureLists feature_lists = 2;
      case 2:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 18)) {
          ptr = ctx->ParseMessage(mutable_feature_lists(), ptr);
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      default: {
      handle_unusual:
        if ((tag & 7) == 4 || tag == 0) {
          ctx->SetLastTag(tag);
          goto success;
        }
        ptr = UnknownFieldParse(tag, &_internal_metadata_, ptr, ctx);
        CHK_(ptr != nullptr);
        continue;
      }
    }  // switch
  }  // while
success:
  return ptr;
failure:
  ptr = nullptr;
  goto success;
#undef CHK_
}
#else  // GOOGLE_PROTOBUF_ENABLE_EXPERIMENTAL_PARSER
bool SequenceExample::MergePartialFromCodedStream(
    ::PROTOBUF_NAMESPACE_ID::io::CodedInputStream* input) {
#define DO_(EXPRESSION) if (!PROTOBUF_PREDICT_TRUE(EXPRESSION)) goto failure
  ::PROTOBUF_NAMESPACE_ID::uint32 tag;
  // @@protoc_insertion_point(parse_start:tensorflow.SequenceExample)
  for (;;) {
    ::std::pair<::PROTOBUF_NAMESPACE_ID::uint32, bool> p = input->ReadTagWithCutoffNoLastTag(127u);
    tag = p.first;
    if (!p.second) goto handle_unusual;
    switch (::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::GetTagFieldNumber(tag)) {
      // .tensorflow.Features context = 1;
      case 1: {
        if (static_cast< ::PROTOBUF_NAMESPACE_ID::uint8>(tag) == (10 & 0xFF)) {
          DO_(::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::ReadMessage(
               input, mutable_context()));
        } else {
          goto handle_unusual;
        }
        break;
      }

      // .tensorflow.FeatureLists feature_lists = 2;
      case 2: {
        if (static_cast< ::PROTOBUF_NAMESPACE_ID::uint8>(tag) == (18 & 0xFF)) {
          DO_(::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::ReadMessage(
               input, mutable_feature_lists()));
        } else {
          goto handle_unusual;
        }
        break;
      }

      default: {
      handle_unusual:
        if (tag == 0) {
          goto success;
        }
        DO_(::PROTOBUF_NAMESPACE_ID::internal::WireFormat::SkipField(
              input, tag, _internal_metadata_.mutable_unknown_fields()));
        break;
      }
    }
  }
success:
  // @@protoc_insertion_point(parse_success:tensorflow.SequenceExample)
  return true;
failure:
  // @@protoc_insertion_point(parse_failure:tensorflow.SequenceExample)
  return false;
#undef DO_
}
#endif  // GOOGLE_PROTOBUF_ENABLE_EXPERIMENTAL_PARSER

void SequenceExample::SerializeWithCachedSizes(
    ::PROTOBUF_NAMESPACE_ID::io::CodedOutputStream* output) const {
  // @@protoc_insertion_point(serialize_start:tensorflow.SequenceExample)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  // .tensorflow.Features context = 1;
  if (this->has_context()) {
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteMessageMaybeToArray(
      1, HasBitSetters::context(this), output);
  }

  // .tensorflow.FeatureLists feature_lists = 2;
  if (this->has_feature_lists()) {
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteMessageMaybeToArray(
      2, HasBitSetters::feature_lists(this), output);
  }

  if (_internal_metadata_.have_unknown_fields()) {
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::SerializeUnknownFields(
        _internal_metadata_.unknown_fields(), output);
  }
  // @@protoc_insertion_point(serialize_end:tensorflow.SequenceExample)
}

::PROTOBUF_NAMESPACE_ID::uint8* SequenceExample::InternalSerializeWithCachedSizesToArray(
    ::PROTOBUF_NAMESPACE_ID::uint8* target) const {
  // @@protoc_insertion_point(serialize_to_array_start:tensorflow.SequenceExample)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  // .tensorflow.Features context = 1;
  if (this->has_context()) {
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::
      InternalWriteMessageToArray(
        1, HasBitSetters::context(this), target);
  }

  // .tensorflow.FeatureLists feature_lists = 2;
  if (this->has_feature_lists()) {
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::
      InternalWriteMessageToArray(
        2, HasBitSetters::feature_lists(this), target);
  }

  if (_internal_metadata_.have_unknown_fields()) {
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::SerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields(), target);
  }
  // @@protoc_insertion_point(serialize_to_array_end:tensorflow.SequenceExample)
  return target;
}

size_t SequenceExample::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:tensorflow.SequenceExample)
  size_t total_size = 0;

  if (_internal_metadata_.have_unknown_fields()) {
    total_size +=
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::ComputeUnknownFieldsSize(
        _internal_metadata_.unknown_fields());
  }
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  // .tensorflow.Features context = 1;
  if (this->has_context()) {
    total_size += 1 +
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::MessageSize(
        *context_);
  }

  // .tensorflow.FeatureLists feature_lists = 2;
  if (this->has_feature_lists()) {
    total_size += 1 +
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::MessageSize(
        *feature_lists_);
  }

  int cached_size = ::PROTOBUF_NAMESPACE_ID::internal::ToCachedSize(total_size);
  SetCachedSize(cached_size);
  return total_size;
}

void SequenceExample::MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) {
// @@protoc_insertion_point(generalized_merge_from_start:tensorflow.SequenceExample)
  GOOGLE_DCHECK_NE(&from, this);
  const SequenceExample* source =
      ::PROTOBUF_NAMESPACE_ID::DynamicCastToGenerated<SequenceExample>(
          &from);
  if (source == nullptr) {
  // @@protoc_insertion_point(generalized_merge_from_cast_fail:tensorflow.SequenceExample)
    ::PROTOBUF_NAMESPACE_ID::internal::ReflectionOps::Merge(from, this);
  } else {
  // @@protoc_insertion_point(generalized_merge_from_cast_success:tensorflow.SequenceExample)
    MergeFrom(*source);
  }
}

void SequenceExample::MergeFrom(const SequenceExample& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:tensorflow.SequenceExample)
  GOOGLE_DCHECK_NE(&from, this);
  _internal_metadata_.MergeFrom(from._internal_metadata_);
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  if (from.has_context()) {
    mutable_context()->::tensorflow::Features::MergeFrom(from.context());
  }
  if (from.has_feature_lists()) {
    mutable_feature_lists()->::tensorflow::FeatureLists::MergeFrom(from.feature_lists());
  }
}

void SequenceExample::CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) {
// @@protoc_insertion_point(generalized_copy_from_start:tensorflow.SequenceExample)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

void SequenceExample::CopyFrom(const SequenceExample& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:tensorflow.SequenceExample)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool SequenceExample::IsInitialized() const {
  return true;
}

void SequenceExample::Swap(SequenceExample* other) {
  if (other == this) return;
  if (GetArenaNoVirtual() == other->GetArenaNoVirtual()) {
    InternalSwap(other);
  } else {
    SequenceExample* temp = New(GetArenaNoVirtual());
    temp->MergeFrom(*other);
    other->CopyFrom(*this);
    InternalSwap(temp);
    if (GetArenaNoVirtual() == nullptr) {
      delete temp;
    }
  }
}
void SequenceExample::UnsafeArenaSwap(SequenceExample* other) {
  if (other == this) return;
  GOOGLE_DCHECK(GetArenaNoVirtual() == other->GetArenaNoVirtual());
  InternalSwap(other);
}
void SequenceExample::InternalSwap(SequenceExample* other) {
  using std::swap;
  _internal_metadata_.Swap(&other->_internal_metadata_);
  swap(context_, other->context_);
  swap(feature_lists_, other->feature_lists_);
}

::PROTOBUF_NAMESPACE_ID::Metadata SequenceExample::GetMetadata() const {
  return GetMetadataStatic();
}


// @@protoc_insertion_point(namespace_scope)
}  // namespace tensorflow
PROTOBUF_NAMESPACE_OPEN
template<> PROTOBUF_NOINLINE ::tensorflow::Example* Arena::CreateMaybeMessage< ::tensorflow::Example >(Arena* arena) {
  return Arena::CreateMessageInternal< ::tensorflow::Example >(arena);
}
template<> PROTOBUF_NOINLINE ::tensorflow::SequenceExample* Arena::CreateMaybeMessage< ::tensorflow::SequenceExample >(Arena* arena) {
  return Arena::CreateMessageInternal< ::tensorflow::SequenceExample >(arena);
}
PROTOBUF_NAMESPACE_CLOSE

// @@protoc_insertion_point(global_scope)
#include <google/protobuf/port_undef.inc>
