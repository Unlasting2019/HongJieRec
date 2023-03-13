#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>

#include "example.pb.h"
#include "feature.pb.h"

#include <fmt/include/fmt/format.h>

namespace tfrecord_parse_dump{

inline void dump_data(
    const char* fieldName,
    const int32_t& v,
    tensorflow::SequenceExample& seq_exp
){
    tensorflow::Features* context_fs = seq_exp.mutable_context();
    tensorflow::Feature& val = (*context_fs->mutable_feature())[fieldName];
    tensorflow::Int64List* int64_v = val.mutable_int64_list();
    int64_v->add_value(v);
}

inline void dump_data(
    const char* fieldName,
    const size_t& v,
    tensorflow::SequenceExample& seq_exp
){
    tensorflow::Features* context_fs = seq_exp.mutable_context();
    tensorflow::Feature& val = (*context_fs->mutable_feature())[fieldName];
    tensorflow::Int64List* int64_v = val.mutable_int64_list();
    int64_v->add_value(v);
}

inline void dump_data(
    const char*  fieldName,
    const int64_t& v,
    tensorflow::SequenceExample& seq_exp
){
    tensorflow::Features* context_fs = seq_exp.mutable_context();
    tensorflow::Feature& val = (*context_fs->mutable_feature())[fieldName];
    tensorflow::Int64List* int64_v = val.mutable_int64_list();
    int64_v->add_value(v);
}

inline void dump_data(
    const char* fieldName,
    const float& v,
    tensorflow::SequenceExample& seq_exp
){
    tensorflow::Features* context_fs = seq_exp.mutable_context();
    tensorflow::Feature& val = (*context_fs->mutable_feature())[fieldName];
    tensorflow::FloatList* float32_v = val.mutable_float_list();
    float32_v->add_value(v);
}

inline void dump_data(
    const char* fieldName,
    const double& v,
    tensorflow::SequenceExample& seq_exp
){
    tensorflow::Features* context_fs = seq_exp.mutable_context();
    tensorflow::Feature& val = (*context_fs->mutable_feature())[fieldName];
    tensorflow::FloatList* float32_v = val.mutable_float_list();
    float32_v->add_value(v);
}


template<typename T>
inline void dump_data(
    const char* fieldName,
    auto rv,
    tensorflow::SequenceExample& seq_exp
){
    tensorflow::FeatureLists* sequence_fs = seq_exp.mutable_feature_lists();
    tensorflow::FeatureList& val = (*sequence_fs->mutable_feature_list())[fieldName];

    if constexpr(std::is_same_v<T, float> || std::is_same_v<T, double>){
        tensorflow::FloatList* fList = val.add_feature()->mutable_float_list();
        for(const auto& v : rv) fList->add_value(v);
        if(fList->value_size() == 0) fList->add_value(0.f);
    } else {
        tensorflow::Int64List* fList = val.add_feature()->mutable_int64_list();
        for(const auto& v : rv) fList->add_value(v);
        if(fList->value_size() == 0) fList->add_value(1);
    }
}

};
