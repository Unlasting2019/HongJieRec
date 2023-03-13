#pragma once
#include <vector>
#include <math.h>
#include <cmath>
#include <unordered_map>
#include <algorithm>
#include <numeric>
#include <string>
#include <ranges>

#include <cpp_tfrecord/dump_tfrecord.h>
#include <fmt/core.h>

#include "src/FeatureStruct.h"
#include "src/FeatureOps/StatOp.h"

using namespace NewsRec;

namespace FeatureOps{

namespace TimeOp{

template<size_t seq_type>
inline void user_time_diff_op(
    const auto& ctx,
    const auto& seq_pf,
    auto& tf_exp
){
    auto rv = seq_pf->get(ctx.user_id).seq_st 
        | views::elements<1>
        | views::transform([](const auto& e){ return e / 1000; });
    const auto& diff_mean = StatOp::numeric_diff_mean_op(rv);
    const auto& diff_std = StatOp::numeric_diff_std_op(rv);
    const auto& diff_mean_std = diff_mean / (diff_std+1e-9);

    if constexpr(seq_type == clk_seq){
        tfrecord_parse_dump::dump_data("clk_user_time_diff_mean", diff_mean, tf_exp);
        tfrecord_parse_dump::dump_data("clk_user_time_diff_std", diff_std, tf_exp);
        tfrecord_parse_dump::dump_data("clk_user_time_diff_mean_std", diff_mean_std, tf_exp);
    } else {
        tfrecord_parse_dump::dump_data("exp_user_time_diff_mean", diff_mean, tf_exp);
        tfrecord_parse_dump::dump_data("exp_user_time_diff_std", diff_std, tf_exp);
        tfrecord_parse_dump::dump_data("exp_user_time_diff_mean_std", diff_mean_std, tf_exp);
    }
}

template<size_t seq_type>
inline void doc_time_diff_op(
    const auto& ctx,
    const auto& seq_pf,
    auto& tf_exp
){
    auto rv = seq_pf->get(ctx.doc_id).seq_st 
        | views::elements<1>
        | views::transform([](const auto& e){ return e / 1000; });
    const auto& diff_mean = StatOp::numeric_diff_mean_op(rv);
    const auto& diff_std = StatOp::numeric_diff_std_op(rv);
    const auto& diff_mean_std = diff_mean / (diff_std+1e-9);

    if constexpr(seq_type == clk_seq){
        tfrecord_parse_dump::dump_data("clk_doc_time_diff_mean", diff_mean, tf_exp);
        tfrecord_parse_dump::dump_data("clk_doc_time_diff_std", diff_std, tf_exp);
        tfrecord_parse_dump::dump_data("clk_doc_time_diff_mean_std", diff_mean_std, tf_exp);
    } else {
        tfrecord_parse_dump::dump_data("exp_doc_time_diff_mean", diff_mean, tf_exp);
        tfrecord_parse_dump::dump_data("exp_doc_time_diff_std", diff_std, tf_exp);
        tfrecord_parse_dump::dump_data("exp_doc_time_diff_mean_std", diff_mean_std, tf_exp);
    }
}

template<size_t seq_type>
inline void user_title_time_diff_op(
    const auto& ctx,
    const auto& seq_pf,
    const auto& doc_pf,
    auto& tf_exp
){
    std::unordered_map<titleT, std::tuple<double, double, double>> desc_map;
    for(const auto&[k, v] : doc_pf->get(ctx.doc_id).doc_title) desc_map[k] = {};

    for(auto&[k, v] : desc_map){
        auto rv = seq_pf->get(ctx.user_id).seq_st
            | views::filter([&](const auto& st){
                return doc_pf->get(std::get<0>(st)).doc_title.contains(k);
            })
            | views::elements<1>
            | views::transform([](const auto& e){ return e / 1000; });

        const auto& diff_mean = StatOp::numeric_diff_mean_op(rv);
        const auto& diff_std = StatOp::numeric_diff_std_op(rv);
        const auto& diff_mean_std = diff_mean / (diff_std+1e-9);
        v = {diff_mean, diff_std, diff_mean_std};
    }

    if constexpr(seq_type == clk_seq){
        tfrecord_parse_dump::dump_data<double>("clk_user_title_time_diff_mean", desc_map | views::values | views::elements<0>, tf_exp);
        tfrecord_parse_dump::dump_data<double>("clk_user_title_time_diff_std", desc_map | views::values | views::elements<1>, tf_exp);
        tfrecord_parse_dump::dump_data<double>("clk_user_title_time_diff_mean_std", desc_map | views::values | views::elements<2>, tf_exp);
    } else {
        tfrecord_parse_dump::dump_data<double>("exp_user_title_time_diff_mean", desc_map | views::values | views::elements<0>, tf_exp);
        tfrecord_parse_dump::dump_data<double>("exp_user_title_time_diff_std", desc_map | views::values | views::elements<1>, tf_exp);
        tfrecord_parse_dump::dump_data<double>("exp_user_title_time_diff_mean_std", desc_map | views::values | views::elements<2>, tf_exp);
    }
}

template<size_t seq_type>
inline void user_cate2_time_diff_op(
    const auto& ctx,
    const auto& seq_pf,
    const auto& doc_pf,
    auto& tf_exp
){
    auto rv = seq_pf->get(ctx.user_id).seq_st 
        | views::filter([&](const auto& st){return doc_pf->get(std::get<0>(st)).doc_cate2 == doc_pf->get(ctx.doc_id).doc_cate2;}) 
        | views::elements<1>
        | views::transform([](const auto& e){ return e / 1000; });
    const auto& diff_mean = StatOp::numeric_diff_mean_op(rv);
    const auto& diff_std = StatOp::numeric_diff_std_op(rv);
    const auto& diff_mean_std = diff_mean / (diff_std+1e-9);

    if constexpr(seq_type == clk_seq){
        tfrecord_parse_dump::dump_data("clk_user_cate2_time_diff_mean", diff_mean, tf_exp);
        tfrecord_parse_dump::dump_data("clk_user_cate2_time_diff_std", diff_std, tf_exp);
        tfrecord_parse_dump::dump_data("clk_user_cate2_time_diff_mean_std", diff_mean_std, tf_exp);
    } else {
        tfrecord_parse_dump::dump_data("exp_user_cate2_time_diff_mean", diff_mean, tf_exp);
        tfrecord_parse_dump::dump_data("exp_user_cate2_time_diff_std", diff_std, tf_exp);
        tfrecord_parse_dump::dump_data("exp_user_cate2_time_diff_mean_std", diff_mean_std, tf_exp);
    }
}

template<size_t seq_type>
inline void user_cate1_time_diff_op(
    const auto& ctx,
    const auto& seq_pf,
    const auto& doc_pf,
    auto& tf_exp
){
    auto rv = seq_pf->get(ctx.user_id).seq_st 
        | views::filter([&](const auto& st){return doc_pf->get(std::get<0>(st)).doc_cate1 == doc_pf->get(ctx.doc_id).doc_cate1;}) 
        | views::elements<1>
        | views::transform([](const auto& e){ return e / 1000; });
    const auto& diff_mean = StatOp::numeric_diff_mean_op(rv);
    const auto& diff_std = StatOp::numeric_diff_std_op(rv);
    const auto& diff_mean_std = diff_mean / (diff_std+1e-9);

    if constexpr(seq_type == clk_seq){
        tfrecord_parse_dump::dump_data("clk_user_cate1_time_diff_mean", diff_mean, tf_exp);
        tfrecord_parse_dump::dump_data("clk_user_cate1_time_diff_std", diff_std, tf_exp);
        tfrecord_parse_dump::dump_data("clk_user_cate1_time_diff_mean_std", diff_mean_std, tf_exp);
    } else {
        tfrecord_parse_dump::dump_data("exp_user_cate1_time_diff_mean", diff_mean, tf_exp);
        tfrecord_parse_dump::dump_data("exp_user_cate1_time_diff_std", diff_std, tf_exp);
        tfrecord_parse_dump::dump_data("exp_user_cate1_time_diff_mean_std", diff_mean_std, tf_exp);
    }
}

template<size_t seq_type, typename ctxT, typename seq_pfT, typename expT>
inline void doc_time_since_op(
    const ctxT& ctx,
    const seq_pfT& seq_pf,
    expT& tf_exp
){
    auto rv = seq_pf->get(ctx.doc_id).seq_st 
        | views::elements<1>
        | views::transform([](const auto& e){ return e / 1000; });
    const auto& ctx_exposedTime = ctx.ctx_exposedTime / 1000;

    const auto& time_since_mean = StatOp::time_since_mean_op(ctx_exposedTime, rv);
    const auto& time_since_mean_std = StatOp::time_since_mean_std_op(ctx_exposedTime, rv);
    const auto& time_since_first = StatOp::time_since_first_op(ctx_exposedTime, rv);
    const auto& time_since_last = StatOp::time_since_last_op(ctx_exposedTime, rv);
    if constexpr(seq_type == clk_seq){
        tfrecord_parse_dump::dump_data("clk_doc_time_since_first", time_since_first, tf_exp);
        tfrecord_parse_dump::dump_data("clk_doc_time_since_last", time_since_last, tf_exp);
        tfrecord_parse_dump::dump_data("clk_doc_time_since_mean", time_since_mean, tf_exp);
        tfrecord_parse_dump::dump_data("clk_doc_time_since_mean_std", time_since_mean_std, tf_exp);
    } else {
        tfrecord_parse_dump::dump_data("exp_doc_time_since_first", time_since_first, tf_exp);
        tfrecord_parse_dump::dump_data("exp_doc_time_since_last", time_since_last, tf_exp);
        tfrecord_parse_dump::dump_data("exp_doc_time_since_mean", time_since_mean, tf_exp);
        tfrecord_parse_dump::dump_data("exp_doc_time_since_mean_std", time_since_mean_std, tf_exp);
    }
}

template<size_t seq_type, typename ctxT, typename seq_pfT, typename expT>
inline void user_time_since_op(
    const ctxT& ctx,
    const seq_pfT& seq_pf,
    expT& tf_exp
){
    auto rv = seq_pf->get(ctx.user_id).seq_st 
        | views::elements<1>
        | views::transform([](const auto& e){ return e / 1000; });
    const auto& ctx_exposedTime = ctx.ctx_exposedTime / 1000;

    const auto& time_since_mean = StatOp::time_since_mean_op(ctx_exposedTime, rv);
    const auto& time_since_mean_std = StatOp::time_since_mean_std_op(ctx_exposedTime, rv);
    const auto& time_since_first = StatOp::time_since_first_op(ctx_exposedTime, rv);
    const auto& time_since_last = StatOp::time_since_last_op(ctx_exposedTime, rv);
    if constexpr(seq_type == clk_seq){
        tfrecord_parse_dump::dump_data("clk_user_time_since_first", time_since_first, tf_exp);
        tfrecord_parse_dump::dump_data("clk_user_time_since_last", time_since_last, tf_exp);
        tfrecord_parse_dump::dump_data("clk_user_time_since_mean", time_since_mean, tf_exp);
        tfrecord_parse_dump::dump_data("clk_user_time_since_mean_std", time_since_mean_std, tf_exp);
    } else {
        tfrecord_parse_dump::dump_data("exp_user_time_since_first", time_since_first, tf_exp);
        tfrecord_parse_dump::dump_data("exp_user_time_since_last", time_since_last, tf_exp);
        tfrecord_parse_dump::dump_data("exp_user_time_since_mean", time_since_mean, tf_exp);
        tfrecord_parse_dump::dump_data("exp_user_time_since_mean_std", time_since_mean_std, tf_exp);
    }
}

template<size_t seq_type>
inline void user_cate1_time_since_op(
    const auto& ctx,
    const auto& seq_pf,
    const auto& doc_pf,
    auto& tf_exp
){
    auto rv = seq_pf->get(ctx.user_id).seq_st 
        | views::filter([&](const auto& s){return doc_pf->get(get<0>(s)).doc_cate1 == doc_pf->get(ctx.doc_id).doc_cate1;}) 
        | views::elements<1>
        | views::transform([](const auto& e){ return e / 1000; });
    const auto& ctx_exposedTime = ctx.ctx_exposedTime / 1000;

    const auto& time_since_mean = StatOp::time_since_mean_op(ctx_exposedTime, rv);
    const auto& time_since_mean_std = StatOp::time_since_mean_std_op(ctx_exposedTime, rv);
    const auto& time_since_first = StatOp::time_since_first_op(ctx_exposedTime, rv);
    const auto& time_since_last = StatOp::time_since_last_op(ctx_exposedTime, rv);
    if constexpr(seq_type == clk_seq){
        tfrecord_parse_dump::dump_data("clk_user_cate1_time_since_first", time_since_first, tf_exp);
        tfrecord_parse_dump::dump_data("clk_user_cate1_time_since_last", time_since_last, tf_exp);
        tfrecord_parse_dump::dump_data("clk_user_cate1_time_since_mean", time_since_mean, tf_exp);
        tfrecord_parse_dump::dump_data("clk_user_cate1_time_since_mean_std", time_since_mean_std, tf_exp);
    } else {
        tfrecord_parse_dump::dump_data("exp_user_cate1_time_since_first", time_since_first, tf_exp);
        tfrecord_parse_dump::dump_data("exp_user_cate1_time_since_last", time_since_last, tf_exp);
        tfrecord_parse_dump::dump_data("exp_user_cate1_time_since_mean", time_since_mean, tf_exp);
        tfrecord_parse_dump::dump_data("exp_user_cate1_time_since_mean_std", time_since_mean_std, tf_exp);
    }
}

template<size_t seq_type>
inline void user_cate2_time_since_op(
    const auto& ctx,
    const auto& seq_pf,
    const auto& doc_pf,
    auto& tf_exp
){
    auto rv = seq_pf->get(ctx.user_id).seq_st 
        | views::filter([&](const auto& s){return doc_pf->get(get<0>(s)).doc_cate2 == doc_pf->get(ctx.doc_id).doc_cate2;}) 
        | views::elements<1>
        | views::transform([](const auto& e){ return e / 1000; });
    const auto& ctx_exposedTime = ctx.ctx_exposedTime / 1000;

    const auto& time_since_mean = StatOp::time_since_mean_op(ctx_exposedTime, rv);
    const auto& time_since_mean_std = StatOp::time_since_mean_std_op(ctx_exposedTime, rv);
    const auto& time_since_first = StatOp::time_since_first_op(ctx_exposedTime, rv);
    const auto& time_since_last = StatOp::time_since_last_op(ctx_exposedTime, rv);

    if constexpr(seq_type == clk_seq){
        tfrecord_parse_dump::dump_data("clk_user_cate2_time_since_first", time_since_first, tf_exp);
        tfrecord_parse_dump::dump_data("clk_user_cate2_time_since_last", time_since_last, tf_exp);
        tfrecord_parse_dump::dump_data("clk_user_cate2_time_since_mean", time_since_mean, tf_exp);
        tfrecord_parse_dump::dump_data("clk_user_cate2_time_since_mean_std", time_since_mean_std, tf_exp);
    } else {
        tfrecord_parse_dump::dump_data("exp_user_cate2_time_since_first", time_since_first, tf_exp);
        tfrecord_parse_dump::dump_data("exp_user_cate2_time_since_last", time_since_last, tf_exp);
        tfrecord_parse_dump::dump_data("exp_user_cate2_time_since_mean", time_since_mean, tf_exp);
        tfrecord_parse_dump::dump_data("exp_user_cate2_time_since_mean_std", time_since_mean_std, tf_exp);
    }
}

template<size_t seq_type>
inline void user_title_time_since_op(
    const auto& ctx,
    const auto& seq_pf,
    const auto& doc_pf,
    auto& tf_exp
){
    /* time_since_first, time_since_last, time_since_mean, time_since_mean_std */
    std::unordered_map<titleT, std::tuple<double, double, double, double>> desc_map;
    for(const auto&[k, v] : doc_pf->get(ctx.doc_id).doc_title) desc_map[k] = {};

    for(auto&[k, v] : desc_map){
        auto rv = seq_pf->get(ctx.user_id).seq_st
            | views::filter([&](const auto& st){
                return doc_pf->get(std::get<0>(st)).doc_title.contains(k);
            })
            | views::elements<1>
            | views::transform([](const auto& e){ return e / 1000; });
        const auto& ctx_exposedTime = ctx.ctx_exposedTime / 1000;

        const auto& time_since_mean = StatOp::time_since_mean_op(ctx_exposedTime, rv);
        const auto& time_since_mean_std = StatOp::time_since_mean_std_op(ctx_exposedTime, rv);
        const auto& time_since_first = StatOp::time_since_first_op(ctx_exposedTime, rv);
        const auto& time_since_last = StatOp::time_since_last_op(ctx_exposedTime, rv);

        v = {time_since_first, time_since_last, time_since_mean, time_since_mean_std};
    }

    if constexpr(seq_type == clk_seq){
        tfrecord_parse_dump::dump_data<double>("clk_user_title_time_since_first", desc_map | views::values | views::elements<0>, tf_exp);
        tfrecord_parse_dump::dump_data<double>("clk_user_title_time_since_last", desc_map | views::values | views::elements<1>, tf_exp);
        tfrecord_parse_dump::dump_data<double>("clk_user_title_time_since_mean", desc_map | views::values | views::elements<2>, tf_exp);
        tfrecord_parse_dump::dump_data<double>("clk_user_title_time_since_mean_std", desc_map | views::values | views::elements<3>, tf_exp);
    } else {
        tfrecord_parse_dump::dump_data<double>("exp_user_title_time_since_first", desc_map | views::values | views::elements<0>, tf_exp);
        tfrecord_parse_dump::dump_data<double>("exp_user_title_time_since_last", desc_map | views::values | views::elements<1>, tf_exp);
        tfrecord_parse_dump::dump_data<double>("exp_user_title_time_since_mean", desc_map | views::values | views::elements<2>, tf_exp);
        tfrecord_parse_dump::dump_data<double>("exp_user_title_time_since_mean_std", desc_map | views::values | views::elements<3>, tf_exp);
    }
}

}; // namespace TimeOp


}; // namespace FeatureOps
