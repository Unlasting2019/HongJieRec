#pragma once
#include <vector>
#include <cmath>
#include <unordered_map>
#include <string>
#include <ranges>

#include <cpp_tfrecord/dump_tfrecord.h>
#include <fmt/core.h>

#include "src/FeatureStruct.h"


using namespace std;
using namespace std::ranges;

using namespace NewsRec;
namespace FeatureOps{

namespace ProductOp{

inline int64_t product_hash(auto f1, auto f2){
    int64_t hash_ = 0;
    while(f1 > 0) hash_ = hash_ * 131 + (f1 % 10), f1 /= 10;
    while(f2 > 0) hash_ = hash_ * 131 + (f2 % 10), f2 /= 10;
    return hash_;
}

inline void doc_city_product_op(
    const auto& ctx,
    const auto& user_pf,
    auto& tf_exp
){
    int64_t product_val = product_hash(ctx.doc_id, user_pf->get(ctx.user_id).user_city);
    tfrecord_parse_dump::dump_data("doc_city_product", product_val, tf_exp);
}

inline void doc_province_product_op(
    const auto& ctx,
    const auto& user_pf,
    auto& tf_exp
){
    int64_t product_val = product_hash(ctx.doc_id, user_pf->get(ctx.user_id).user_province);
    tfrecord_parse_dump::dump_data("doc_province_product", product_val, tf_exp);
}

inline void doc_device_product_op(
    const auto& ctx,
    const auto& user_pf,
    auto& tf_exp
){
    int64_t product_val = product_hash(ctx.doc_id, user_pf->get(ctx.user_id).user_device);
    tfrecord_parse_dump::dump_data("doc_device_product", product_val, tf_exp);
}

inline void user_title_product_op(
    const auto& ctx,
    const auto& doc_pf,
    auto& tf_exp
){
    auto rv = doc_pf->get(ctx.doc_id).doc_title
        | views::keys 
        | views::transform([&](const auto& t){
            return product_hash(ctx.user_id, t);
        });

    tfrecord_parse_dump::dump_data<int64_t>("user_title_product", rv, tf_exp);
}

template<size_t seq_type, size_t min_watch_time>
inline void sequence_doc_product_op(
    const auto& ctx,
    const auto& seq_pf,
    const auto& doc_pf,
    auto& tf_exp
){
    auto rv = seq_pf->get(ctx.user_id).seq_st 
        | views::filter([](const auto& st){return std::get<2>(st) > min_watch_time;})
        | views::elements<0>
        | views::transform([&](const auto& seq_doc){
            return product_hash(ctx.doc_id, seq_doc);
        });

    if constexpr(seq_type == clk_seq)
        tfrecord_parse_dump::dump_data<int64_t>("clk_doc_product", rv, tf_exp);
    else
        tfrecord_parse_dump::dump_data<int64_t>("exp_doc_product", rv, tf_exp);
}

inline void user_cate1_product_op(
    const auto& ctx,
    const auto& doc_pf,
    auto& tf_exp
){
    int64_t product_val = product_hash(ctx.user_id, doc_pf->get(ctx.doc_id).doc_cate1);
    tfrecord_parse_dump::dump_data("user_cate1_product", product_val, tf_exp);
}

inline void user_cate2_product_op(
    const auto& ctx,
    const auto& doc_pf,
    auto& tf_exp
){
    int64_t product_val = product_hash(ctx.user_id, doc_pf->get(ctx.doc_id).doc_cate2);
    tfrecord_parse_dump::dump_data("user_cate2_product", product_val, tf_exp);
}

}; // namespace ProductOp

}; // namespace FeatureOps
