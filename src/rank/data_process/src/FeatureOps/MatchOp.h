#pragma once
#include <numeric>
#include <algorithm>
#include <vector>
#include <cmath>
#include <unordered_map>
#include <string>
#include <ranges>

#include <cpp_tfrecord/dump_tfrecord.h>
#include <fmt/core.h>

#include "src/FeatureStruct.h"
#include "src/FeatureOps/StatOp.h"

using namespace NewsRec;

namespace FeatureOps{

namespace MatchOp{

inline void doc_watch_op(
    const auto& ctx,
    const auto& clk_seq_pf,
    const auto& watch_prior,
    auto& tf_exp
){
    const auto& clk_pf = clk_seq_pf->get(ctx.doc_id).seq_st;

    const auto& cmp = [](const auto& st){ return std::get<2>(st) >= 1;};
    const auto& cnt = ranges::count_if(clk_pf, cmp);
    const auto& sum = StatOp::numeric_sum_op(clk_pf | views::filter(cmp) | views::elements<2>);
    const auto& mean = StatOp::watch_desc_op(cnt, sum, watch_prior);

    tfrecord_parse_dump::dump_data("doc_watch_sum", sum,  tf_exp);
    tfrecord_parse_dump::dump_data("doc_watch_mean", mean,  tf_exp);
}

inline void doc_refreshTimes_watch_op(
    const auto& ctx,
    const auto& clk_seq_pf,
    const auto& doc_pf,
    const auto& watch_prior,
    auto& tf_exp
){
    const auto& clk_pf = clk_seq_pf->get(ctx.doc_id).seq_st;
    // base
    const auto& cmp = [](const auto& st){ return std::get<2>(st) >= 1;};
    const auto& cnt = ranges::count_if(clk_pf, cmp);
    const auto& sum = StatOp::numeric_sum_op(clk_pf | views::filter(cmp) | views::elements<2>);
    const auto& mean = StatOp::watch_desc_op(cnt, sum, watch_prior);
    // doc_refreshTimes
    const auto& refreshTimes_cmp = [&](const auto& st){ return std::get<2>(st) >= 1 && std::get<3>(st) == ctx.ctx_refreshTimes;};
    const auto& refreshTimes_cnt = ranges::count_if(clk_pf, refreshTimes_cmp);
    const auto& refreshTimes_sum = StatOp::numeric_sum_op(clk_pf | views::filter(refreshTimes_cmp) | views::elements<2>);
    const auto& refreshTimes_mean = StatOp::watch_desc_op(refreshTimes_cnt, refreshTimes_sum, watch_prior);
    const auto& refreshTimes_sum_ratio = sum == 0 ? 0 : refreshTimes_sum / sum;
    const auto& refreshTimes_mean_ratio = sum == 0 ? 0 : refreshTimes_mean / mean;

    //fmt::print("refreshTimes_cnt:{}\nrefreshTimes_sum:{}\nrefreshTimes_mean:{}\nrefreshTimes_sum_ratio:{}\nrefreshTimes_mean_ratio:{}\n----\n", refreshTimes_cnt, refreshTimes_sum, refreshTimes_mean, refreshTimes_sum_ratio, refreshTimes_mean_ratio);

    tfrecord_parse_dump::dump_data("doc_refreshTimes_watch_sum", refreshTimes_sum,  tf_exp);
    tfrecord_parse_dump::dump_data("doc_refreshTimes_watch_mean", refreshTimes_mean,  tf_exp);
    tfrecord_parse_dump::dump_data("doc_refreshTimes_watch_sum_ratio", refreshTimes_sum_ratio,  tf_exp);
    tfrecord_parse_dump::dump_data("doc_refreshTimes_watch_mean_ratio", refreshTimes_mean_ratio,  tf_exp);
}

inline void user_watch_op(
    const auto& ctx,
    const auto& clk_seq_pf,
    const auto& watch_prior,
    auto& tf_exp
){
    const auto& clk_pf = clk_seq_pf->get(ctx.user_id).seq_st;

    const auto& cmp = [](const auto& st){ return std::get<2>(st) >= 1;};
    const auto& cnt = ranges::count_if(clk_pf, cmp);
    const auto& sum = StatOp::numeric_sum_op(clk_pf | views::filter(cmp) | views::elements<2>);
    const auto& mean = StatOp::watch_desc_op(cnt, sum, watch_prior);

    tfrecord_parse_dump::dump_data("user_watch_sum", sum,  tf_exp);
    tfrecord_parse_dump::dump_data("user_watch_mean", mean,  tf_exp);
}

inline void user_cate1_watch_op(
    const auto& ctx,
    const auto& clk_seq_pf,
    const auto& doc_pf,
    const auto& watch_prior,
    auto& tf_exp
){
    const auto& clk_pf = clk_seq_pf->get(ctx.user_id).seq_st;
    // base
    const auto& cmp = [](const auto& st){ return std::get<2>(st) >= 1;};
    const auto& cnt = ranges::count_if(clk_pf, cmp);
    const auto& sum = StatOp::numeric_sum_op(clk_pf | views::filter(cmp) | views::elements<2>);
    const auto& mean = StatOp::watch_desc_op(cnt, sum, watch_prior);
    // user_cate1
    const auto& cate1_cmp = [&](const auto& st){ return std::get<2>(st) >= 1 && doc_pf->get(std::get<0>(st)).doc_cate1 == doc_pf->get(ctx.doc_id).doc_cate1;};
    const auto& cate1_cnt = ranges::count_if(clk_pf, cate1_cmp);
    const auto& cate1_sum = StatOp::numeric_sum_op(clk_pf | views::filter(cate1_cmp) | views::elements<2>);
    const auto& cate1_mean = StatOp::watch_desc_op(cate1_cnt, cate1_sum, watch_prior);
    const auto& cate1_sum_ratio = sum == 0 ? 0 : cate1_sum / sum;
    const auto& cate1_mean_ratio = sum == 0 ? 0 : cate1_mean / mean;

    //fmt::print("cate1_cnt:{}\ncate1_sum:{}\ncate1_mean:{}\ncate1_sum_ratio:{}\ncate1_mean_ratio:{}\n----\n", cate1_cnt, cate1_sum, cate1_mean, cate1_sum_ratio, cate1_mean_ratio);

    tfrecord_parse_dump::dump_data("user_cate1_watch_sum", cate1_sum,  tf_exp);
    tfrecord_parse_dump::dump_data("user_cate1_watch_mean", cate1_mean,  tf_exp);
    tfrecord_parse_dump::dump_data("user_cate1_watch_sum_ratio", cate1_sum_ratio,  tf_exp);
    tfrecord_parse_dump::dump_data("user_cate1_watch_mean_ratio", cate1_mean_ratio,  tf_exp);
}

inline void user_cate2_watch_op(
    const auto& ctx,
    const auto& clk_seq_pf,
    const auto& doc_pf,
    const auto& watch_prior,
    auto& tf_exp
){
    const auto& clk_pf = clk_seq_pf->get(ctx.user_id).seq_st;
    // base
    const auto& cmp = [](const auto& st){ return std::get<2>(st) >= 1;};
    const auto& cnt = ranges::count_if(clk_pf, cmp);
    const auto& sum = StatOp::numeric_sum_op(clk_pf | views::filter(cmp) | views::elements<2>);
    const auto& mean = StatOp::watch_desc_op(cnt, sum, watch_prior);
    // user_cate2
    const auto& cate2_cmp = [&](const auto& st){ return std::get<2>(st) >= 1 && doc_pf->get(std::get<0>(st)).doc_cate2 == doc_pf->get(ctx.doc_id).doc_cate2;};
    const auto& cate2_cnt = ranges::count_if(clk_pf, cate2_cmp);
    const auto& cate2_sum = StatOp::numeric_sum_op(clk_pf | views::filter(cate2_cmp) | views::elements<2>);
    const auto& cate2_mean = StatOp::watch_desc_op(cate2_cnt, cate2_sum, watch_prior);
    const auto& cate2_sum_ratio = sum == 0 ? 0 : cate2_sum / sum;
    const auto& cate2_mean_ratio = sum == 0 ? 0 : cate2_mean / mean;

    //fmt::print("cate2_cnt:{}\ncate2_sum:{}\ncate2_mean:{}\ncate2_sum_ratio:{}\ncate2_mean_ratio:{}\n----\n", cate2_cnt, cate2_sum, cate2_mean, cate2_sum_ratio, cate2_mean_ratio);

    tfrecord_parse_dump::dump_data("user_cate2_watch_sum", cate2_sum,  tf_exp);
    tfrecord_parse_dump::dump_data("user_cate2_watch_mean", cate2_mean,  tf_exp);
    tfrecord_parse_dump::dump_data("user_cate2_watch_sum_ratio", cate2_sum_ratio,  tf_exp);
    tfrecord_parse_dump::dump_data("user_cate2_watch_mean_ratio", cate2_mean_ratio,  tf_exp);
}

inline void user_title_watch_op(
    const auto& ctx,
    const auto& clk_seq_pf,
    const auto& doc_pf,
    const auto& watch_prior,
    auto& tf_exp
){
    // base
    const auto& clk_pf = clk_seq_pf->get(ctx.user_id).seq_st;
    const auto& cmp = [](const auto& st){ return std::get<2>(st) >= 1;};
    const auto& cnt = ranges::count_if(clk_pf, cmp);
    const auto& sum = StatOp::numeric_sum_op(clk_pf | views::filter(cmp) | views::elements<2>);
    const auto& mean = StatOp::watch_desc_op(cnt, sum, watch_prior);
    // user_title
    std::unordered_map<titleT, std::tuple<double, double, double, double>> desc_map;
    for(const auto&[k, v]: doc_pf->get(ctx.doc_id).doc_title) desc_map[k] = {};
    for(auto&[k, v] : desc_map){
        const auto& title_cmp = [&](const auto& st){ return std::get<2>(st) >= 1 && doc_pf->get(std::get<0>(st)).doc_title.contains(k);};
        const auto& title_cnt = ranges::count_if(clk_pf, title_cmp);
        const auto& title_sum = StatOp::numeric_sum_op(clk_pf | views::filter(title_cmp) | views::elements<2>);
        const auto& title_mean = StatOp::watch_desc_op(title_cnt, title_sum, watch_prior);
        const auto& title_sum_ratio = sum == 0 ? 0 : title_sum / sum;
        const auto& title_mean_ratio = sum == 0 ? 0 : title_mean / mean;
        //fmt::print("title_cnt:{}\ntitle_sum:{}\ntitle_mean:{}\ntitle_sum_ratio:{}\ntitle_mean_ratio:{}\n----\n", title_cnt, title_sum, title_mean, title_sum_ratio, title_mean_ratio);

        v = {title_sum, title_mean, title_sum_ratio, title_mean_ratio};
    }

    tfrecord_parse_dump::dump_data<double>("user_title_watch_sum", desc_map | views::values | views::elements<0>, tf_exp);
    tfrecord_parse_dump::dump_data<double>("user_title_watch_mean", desc_map | views::values | views::elements<1>, tf_exp);
    tfrecord_parse_dump::dump_data<double>("user_title_watch_sum_ratio", desc_map | views::values | views::elements<2>, tf_exp);
    tfrecord_parse_dump::dump_data<double>("user_title_watch_mean_ratio", desc_map | views::values | views::elements<3>, tf_exp);
}
inline void doc_click_op(
    const auto& ctx,
    const auto& clk_seq_pf,
    const auto& exp_seq_pf,
    const auto& clk_prior,
    auto& tf_exp
){
    const auto& clk_cnt = clk_seq_pf->get(ctx.doc_id).seq_st.size();
    const auto& exp_cnt = exp_seq_pf->get(ctx.doc_id).seq_st.size();
    const auto& clk_mean = StatOp::ctr_desc_op<0>(exp_cnt, clk_cnt, clk_prior);
    const auto& clk_mode = StatOp::ctr_desc_op<1>(exp_cnt, clk_cnt, clk_prior);
    const auto& clk_median = StatOp::ctr_desc_op<2>(exp_cnt, clk_cnt, clk_prior);
    const auto& clk_var = StatOp::ctr_desc_op<3>(exp_cnt, clk_cnt, clk_prior);
    const auto& clk_skew = StatOp::ctr_desc_op<4>(exp_cnt, clk_cnt, clk_prior);
    const auto& clk_kurt = StatOp::ctr_desc_op<5>(exp_cnt, clk_cnt, clk_prior);

    tfrecord_parse_dump::dump_data("doc_clk_cnt", (double)clk_cnt, tf_exp);
    tfrecord_parse_dump::dump_data("doc_exp_cnt", (double)exp_cnt, tf_exp);
    tfrecord_parse_dump::dump_data("doc_clk_mean", clk_mean, tf_exp);
    tfrecord_parse_dump::dump_data("doc_clk_mode", clk_mode, tf_exp);
    tfrecord_parse_dump::dump_data("doc_clk_median", clk_median, tf_exp);
    tfrecord_parse_dump::dump_data("doc_clk_var", clk_var, tf_exp);
    tfrecord_parse_dump::dump_data("doc_clk_skew", clk_skew, tf_exp);
    tfrecord_parse_dump::dump_data("doc_clk_kurt", clk_kurt, tf_exp);
}

inline void user_click_op(
    const auto& ctx,
    const auto& clk_seq_pf,
    const auto& exp_seq_pf,
    const auto& clk_prior,
    auto& tf_exp
){
    const auto& clk_cnt = clk_seq_pf->get(ctx.user_id).seq_st.size();
    const auto& exp_cnt = exp_seq_pf->get(ctx.user_id).seq_st.size();
    const auto& clk_mean = StatOp::ctr_desc_op<0>(exp_cnt, clk_cnt, clk_prior);
    const auto& clk_mode = StatOp::ctr_desc_op<1>(exp_cnt, clk_cnt, clk_prior);
    const auto& clk_median = StatOp::ctr_desc_op<2>(exp_cnt, clk_cnt, clk_prior);
    const auto& clk_var = StatOp::ctr_desc_op<3>(exp_cnt, clk_cnt, clk_prior);
    const auto& clk_skew = StatOp::ctr_desc_op<4>(exp_cnt, clk_cnt, clk_prior);
    const auto& clk_kurt = StatOp::ctr_desc_op<5>(exp_cnt, clk_cnt, clk_prior);

    tfrecord_parse_dump::dump_data("user_clk_cnt", (double)clk_cnt, tf_exp);
    tfrecord_parse_dump::dump_data("user_exp_cnt", (double)exp_cnt, tf_exp);
    tfrecord_parse_dump::dump_data("user_clk_mean", clk_mean, tf_exp);
    tfrecord_parse_dump::dump_data("user_clk_mode", clk_mode, tf_exp);
    tfrecord_parse_dump::dump_data("user_clk_median", clk_median, tf_exp);
    tfrecord_parse_dump::dump_data("user_clk_var", clk_var, tf_exp);
    tfrecord_parse_dump::dump_data("user_clk_skew", clk_skew, tf_exp);
    tfrecord_parse_dump::dump_data("user_clk_kurt", clk_kurt, tf_exp);
}


inline void user_title_click_op(
    const auto& ctx,
    const auto& clk_seq_pf,
    const auto& exp_seq_pf,
    const auto& doc_pf,
    const auto& clk_prior,
    auto& tf_exp
){
    const auto& clk_spf = clk_seq_pf->get(ctx.user_id).seq_st;
    const auto& exp_spf = exp_seq_pf->get(ctx.user_id).seq_st;
    const auto& base = StatOp::ctr_desc_op<0>(exp_spf.size(), clk_spf.size(), clk_prior);

    std::unordered_map<titleT, std::tuple<double, double, double, double, double, double, double, double, double, double, double>> desc_map;
    for(const auto&[k, v]: doc_pf->get(ctx.doc_id).doc_title) desc_map[k] = {};
    
    for(auto&[k, v] : desc_map){
        const auto& exp_cnt = ranges::count_if(exp_spf | views::elements<0>, [&](const auto& d){ return doc_pf->get(d).doc_title.contains(k);});
        const auto& clk_cnt = ranges::count_if(clk_spf | views::elements<0>, [&](const auto& d){ return doc_pf->get(d).doc_title.contains(k);});
        const auto& exp_ratio = exp_spf.empty() ? 0 : 1.0 * exp_cnt / exp_spf.size();
        const auto& clk_ratio = clk_spf.empty() ? 0 : 1.0 * clk_cnt / clk_spf.size();
        const auto& clk_mean = StatOp::ctr_desc_op<0>(exp_cnt, clk_cnt, clk_prior);
        const auto& clk_mode = StatOp::ctr_desc_op<1>(exp_cnt, clk_cnt, clk_prior);
        const auto& clk_median = StatOp::ctr_desc_op<2>(exp_cnt, clk_cnt, clk_prior);
        const auto& clk_var = StatOp::ctr_desc_op<3>(exp_cnt, clk_cnt, clk_prior);
        const auto& clk_skew = StatOp::ctr_desc_op<4>(exp_cnt, clk_cnt, clk_prior);
        const auto& clk_kurt = StatOp::ctr_desc_op<5>(exp_cnt, clk_cnt, clk_prior);
        const auto& clk_mean_ratio = clk_spf.empty() ? 0 : clk_mean / base;
        v = {exp_cnt, clk_cnt, exp_ratio, clk_ratio, clk_mean, clk_mode, clk_median,  clk_var, clk_skew, clk_kurt, clk_mean_ratio};
    }

    tfrecord_parse_dump::dump_data<double>("user_title_exp_cnt", desc_map | views::values | views::elements<0>, tf_exp);
    tfrecord_parse_dump::dump_data<double>("user_title_clk_cnt", desc_map | views::values | views::elements<1>, tf_exp);
    tfrecord_parse_dump::dump_data<double>("user_title_exp_ratio", desc_map | views::values | views::elements<2>, tf_exp);
    tfrecord_parse_dump::dump_data<double>("user_title_clk_ratio", desc_map | views::values | views::elements<3>, tf_exp);
    tfrecord_parse_dump::dump_data<double>("user_title_clk_mean", desc_map | views::values | views::elements<4>, tf_exp);
    tfrecord_parse_dump::dump_data<double>("user_title_clk_mode", desc_map | views::values | views::elements<5>, tf_exp);
    tfrecord_parse_dump::dump_data<double>("user_title_clk_median", desc_map | views::values | views::elements<6>, tf_exp);
    tfrecord_parse_dump::dump_data<double>("user_title_clk_var", desc_map | views::values | views::elements<7>, tf_exp);
    tfrecord_parse_dump::dump_data<double>("user_title_clk_skew", desc_map | views::values | views::elements<8>, tf_exp);
    tfrecord_parse_dump::dump_data<double>("user_title_clk_kurt", desc_map | views::values | views::elements<9>, tf_exp);
    tfrecord_parse_dump::dump_data<double>("user_title_clk_mean_ratio", desc_map | views::values | views::elements<10>, tf_exp);
}

inline void user_hour_click_op(
    const auto& ctx,
    const auto& clk_seq_pf,
    const auto& exp_seq_pf,
    const auto& doc_pf,
    const auto& clk_prior,
    auto& tf_exp
){
    const auto& clk_spf = clk_seq_pf->get(ctx.user_id).seq_st;
    const auto& exp_spf = exp_seq_pf->get(ctx.user_id).seq_st;
    int32_t key = (int32_t)(ctx.ctx_exposedTime / 1000 / 3600 + 8) % 24;

    const auto& exp_cnt = ranges::count_if(exp_spf, [&](const auto& st){ return (int32_t)(std::get<1>(st) / 1000 / 3600 + 8) % 24 == key;});
    const auto& clk_cnt = ranges::count_if(clk_spf, [&](const auto& st){ return (int32_t)(std::get<1>(st) / 1000 / 3600 + 8) % 24 == key;});
    const auto& exp_ratio = exp_spf.empty() ? 0 : (double)exp_cnt / exp_spf.size();
    const auto& clk_ratio = clk_spf.empty() ? 0 : (double)clk_cnt / clk_spf.size();
    
    const auto& clk_mean = StatOp::ctr_desc_op<0>(exp_cnt, clk_cnt, clk_prior);
    const auto& clk_mode = StatOp::ctr_desc_op<1>(exp_cnt, clk_cnt, clk_prior);
    const auto& clk_median = StatOp::ctr_desc_op<2>(exp_cnt, clk_cnt, clk_prior);
    const auto& clk_var = StatOp::ctr_desc_op<3>(exp_cnt, clk_cnt, clk_prior);
    const auto& clk_skew = StatOp::ctr_desc_op<4>(exp_cnt, clk_cnt, clk_prior);
    const auto& clk_kurt = StatOp::ctr_desc_op<5>(exp_cnt, clk_cnt, clk_prior);

    const auto& base = StatOp::ctr_desc_op<0>(exp_spf.size(), clk_spf.size(), clk_prior);
    const auto& clk_mean_ratio = clk_spf.empty() ? 0 : clk_mean / base;

    tfrecord_parse_dump::dump_data("user_hour_clk_cnt", (double)clk_cnt, tf_exp);
    tfrecord_parse_dump::dump_data("user_hour_clk_ratio", clk_ratio, tf_exp);

    tfrecord_parse_dump::dump_data("user_hour_exp_cnt", (double)exp_cnt, tf_exp);
    tfrecord_parse_dump::dump_data("user_hour_exp_ratio", exp_ratio, tf_exp);

    tfrecord_parse_dump::dump_data("user_hour_clk_mean", clk_mean, tf_exp);
    tfrecord_parse_dump::dump_data("user_hour_clk_mode", clk_mode, tf_exp);
    tfrecord_parse_dump::dump_data("user_hour_clk_median", clk_median, tf_exp);
    tfrecord_parse_dump::dump_data("user_hour_clk_var", clk_var, tf_exp);
    tfrecord_parse_dump::dump_data("user_hour_clk_skew", clk_skew, tf_exp);
    tfrecord_parse_dump::dump_data("user_hour_clk_kurt", clk_kurt, tf_exp);
    tfrecord_parse_dump::dump_data("user_hour_clk_mean_ratio", clk_mean_ratio, tf_exp);
}

inline void user_refreshTimes_click_op(
    const auto& ctx,
    const auto& clk_seq_pf,
    const auto& exp_seq_pf,
    const auto& doc_pf,
    const auto& clk_prior,
    auto& tf_exp
){
    const auto& clk_spf = clk_seq_pf->get(ctx.user_id).seq_st;
    const auto& exp_spf = exp_seq_pf->get(ctx.user_id).seq_st;

    const auto& exp_cnt = ranges::count_if(exp_spf, [&](const auto& st){ return std::get<3>(st) == ctx.ctx_refreshTimes;});
    const auto& clk_cnt = ranges::count_if(clk_spf, [&](const auto& st){ return std::get<3>(st)  == ctx.ctx_refreshTimes;});
    const auto& exp_ratio = exp_spf.empty() ? 0 : (double)exp_cnt / exp_spf.size();
    const auto& clk_ratio = clk_spf.empty() ? 0 : (double)clk_cnt / clk_spf.size();

    
    const auto& clk_mean = StatOp::ctr_desc_op<0>(exp_cnt, clk_cnt, clk_prior);
    const auto& clk_mode = StatOp::ctr_desc_op<1>(exp_cnt, clk_cnt, clk_prior);
    const auto& clk_median = StatOp::ctr_desc_op<2>(exp_cnt, clk_cnt, clk_prior);
    const auto& clk_var = StatOp::ctr_desc_op<3>(exp_cnt, clk_cnt, clk_prior);
    const auto& clk_skew = StatOp::ctr_desc_op<4>(exp_cnt, clk_cnt, clk_prior);
    const auto& clk_kurt = StatOp::ctr_desc_op<5>(exp_cnt, clk_cnt, clk_prior);

    const auto& base = StatOp::ctr_desc_op<0>(exp_spf.size(), clk_spf.size(), clk_prior);
    const auto& clk_mean_ratio = clk_spf.empty() ? 0 : clk_mean / base;

    tfrecord_parse_dump::dump_data("user_refreshTimes_clk_cnt", (double)clk_cnt, tf_exp);
    tfrecord_parse_dump::dump_data("user_refreshTimes_clk_ratio", clk_ratio, tf_exp);

    tfrecord_parse_dump::dump_data("user_refreshTimes_exp_cnt", (double)exp_cnt, tf_exp);
    tfrecord_parse_dump::dump_data("user_refreshTimes_exp_ratio", exp_ratio, tf_exp);

    tfrecord_parse_dump::dump_data("user_refreshTimes_clk_mean", clk_mean, tf_exp);
    tfrecord_parse_dump::dump_data("user_refreshTimes_clk_mode", clk_mode, tf_exp);
    tfrecord_parse_dump::dump_data("user_refreshTimes_clk_median", clk_median, tf_exp);
    tfrecord_parse_dump::dump_data("user_refreshTimes_clk_var", clk_var, tf_exp);
    tfrecord_parse_dump::dump_data("user_refreshTimes_clk_skew", clk_skew, tf_exp);
    tfrecord_parse_dump::dump_data("user_refreshTimes_clk_kurt", clk_kurt, tf_exp);
    tfrecord_parse_dump::dump_data("user_refreshTimes_clk_mean_ratio", clk_mean_ratio, tf_exp);
}

inline void user_cate2_click_op(
    const auto& ctx,
    const auto& clk_seq_pf,
    const auto& exp_seq_pf,
    const auto& doc_pf,
    const auto& clk_prior,
    auto& tf_exp
){
    const auto& clk_spf = clk_seq_pf->get(ctx.user_id).seq_st;
    const auto& exp_spf = exp_seq_pf->get(ctx.user_id).seq_st;
    cate2T key = doc_pf->get(ctx.doc_id).doc_cate2;

    const auto& exp_cnt = std::ranges::count_if(exp_spf, [&](const auto& st){ return doc_pf->get(std::get<0>(st)).doc_cate2 == key;});
    const auto& clk_cnt = std::ranges::count_if(clk_spf, [&](const auto& st){ return doc_pf->get(std::get<0>(st)).doc_cate2 == key;});
    const auto& exp_ratio = exp_spf.empty() ? 0 : (double)exp_cnt / exp_spf.size();
    const auto& clk_ratio = clk_spf.empty() ? 0 : (double)clk_cnt / clk_spf.size();
    
    const auto& clk_mean = StatOp::ctr_desc_op<0>(exp_cnt, clk_cnt, clk_prior);
    const auto& clk_mode = StatOp::ctr_desc_op<1>(exp_cnt, clk_cnt, clk_prior);
    const auto& clk_median = StatOp::ctr_desc_op<2>(exp_cnt, clk_cnt, clk_prior);
    const auto& clk_var = StatOp::ctr_desc_op<3>(exp_cnt, clk_cnt, clk_prior);
    const auto& clk_skew = StatOp::ctr_desc_op<4>(exp_cnt, clk_cnt, clk_prior);
    const auto& clk_kurt = StatOp::ctr_desc_op<5>(exp_cnt, clk_cnt, clk_prior);

    const auto& base = StatOp::ctr_desc_op<0>(exp_spf.size(), clk_spf.size(), clk_prior);
    const auto& clk_mean_ratio = clk_spf.empty() ? 0 : clk_mean / base;

    tfrecord_parse_dump::dump_data("user_cate2_clk_cnt", (double)clk_cnt, tf_exp);
    tfrecord_parse_dump::dump_data("user_cate2_clk_ratio", clk_ratio, tf_exp);

    tfrecord_parse_dump::dump_data("user_cate2_exp_cnt", (double)exp_cnt, tf_exp);
    tfrecord_parse_dump::dump_data("user_cate2_exp_ratio", exp_ratio, tf_exp);

    tfrecord_parse_dump::dump_data("user_cate2_clk_mean", clk_mean, tf_exp);
    tfrecord_parse_dump::dump_data("user_cate2_clk_mode", clk_mode, tf_exp);
    tfrecord_parse_dump::dump_data("user_cate2_clk_median", clk_median, tf_exp);
    tfrecord_parse_dump::dump_data("user_cate2_clk_var", clk_var, tf_exp);
    tfrecord_parse_dump::dump_data("user_cate2_clk_skew", clk_skew, tf_exp);
    tfrecord_parse_dump::dump_data("user_cate2_clk_kurt", clk_kurt, tf_exp);
    tfrecord_parse_dump::dump_data("user_cate2_clk_mean_ratio", clk_mean_ratio, tf_exp);
}

inline void user_cate1_click_op(
    const auto& ctx,
    const auto& clk_seq_pf,
    const auto& exp_seq_pf,
    const auto& doc_pf,
    const auto& clk_prior,
    auto& tf_exp
){
    const auto& clk_spf = clk_seq_pf->get(ctx.user_id).seq_st;
    const auto& exp_spf = exp_seq_pf->get(ctx.user_id).seq_st;
    cate1T key = doc_pf->get(ctx.doc_id).doc_cate1;

    const auto& exp_cnt = std::ranges::count_if(exp_spf, [&](const auto& st){ return doc_pf->get(std::get<0>(st)).doc_cate1 == key;});
    const auto& clk_cnt = std::ranges::count_if(clk_spf, [&](const auto& st){ return doc_pf->get(std::get<0>(st)).doc_cate1 == key;});
    const auto& exp_ratio = exp_spf.empty() ? 0. : (double)exp_cnt / exp_spf.size();
    const auto& clk_ratio = clk_spf.empty() ? 0. : (double)clk_cnt / clk_spf.size();

    const auto& clk_mean = StatOp::ctr_desc_op<0>(exp_cnt, clk_cnt, clk_prior);
    const auto& clk_mode = StatOp::ctr_desc_op<1>(exp_cnt, clk_cnt, clk_prior);
    const auto& clk_median = StatOp::ctr_desc_op<2>(exp_cnt, clk_cnt, clk_prior);
    const auto& clk_var = StatOp::ctr_desc_op<3>(exp_cnt, clk_cnt, clk_prior);
    const auto& clk_skew = StatOp::ctr_desc_op<4>(exp_cnt, clk_cnt, clk_prior);
    const auto& clk_kurt = StatOp::ctr_desc_op<5>(exp_cnt, clk_cnt, clk_prior);

    const auto& base = StatOp::ctr_desc_op<0>(exp_spf.size(), clk_spf.size(), clk_prior);
    const auto& clk_mean_ratio = clk_spf.empty() ? 0 : clk_mean / base;

    tfrecord_parse_dump::dump_data("user_cate1_clk_cnt", (double)clk_cnt, tf_exp);
    tfrecord_parse_dump::dump_data("user_cate1_clk_ratio", clk_ratio, tf_exp);

    tfrecord_parse_dump::dump_data("user_cate1_exp_cnt", (double)exp_cnt, tf_exp);
    tfrecord_parse_dump::dump_data("user_cate1_exp_ratio", exp_ratio, tf_exp);

    tfrecord_parse_dump::dump_data("user_cate1_clk_mean", clk_mean, tf_exp);
    tfrecord_parse_dump::dump_data("user_cate1_clk_mode", clk_mode, tf_exp);
    tfrecord_parse_dump::dump_data("user_cate1_clk_median", clk_median, tf_exp);
    tfrecord_parse_dump::dump_data("user_cate1_clk_var", clk_var, tf_exp);
    tfrecord_parse_dump::dump_data("user_cate1_clk_skew", clk_skew, tf_exp);
    tfrecord_parse_dump::dump_data("user_cate1_clk_kurt", clk_kurt, tf_exp);

    tfrecord_parse_dump::dump_data("user_cate1_clk_mean_ratio", clk_mean_ratio, tf_exp);
}

inline void doc_device_click_op(
    const auto& ctx,
    const auto& clk_seq_pf,
    const auto& exp_seq_pf,
    const auto& user_pf,
    const auto& clk_prior,
    auto& tf_exp
){
    const auto& clk_spf = clk_seq_pf->get(ctx.user_id).seq_st;
    const auto& exp_spf = exp_seq_pf->get(ctx.user_id).seq_st;
    deviceT key = user_pf->get(ctx.user_id).user_device;

    const auto& exp_cnt = std::ranges::count_if(exp_spf, [&](const auto& st){ return user_pf->get(std::get<0>(st)).user_device == key;});
    const auto& clk_cnt = std::ranges::count_if(clk_spf, [&](const auto& st){ return user_pf->get(std::get<0>(st)).user_device == key;});
    const auto& exp_ratio = exp_spf.empty() ? 0 : (double)exp_cnt / exp_spf.size();
    const auto& clk_ratio = clk_spf.empty() ? 0 : (double)clk_cnt / clk_spf.size();

    const auto& clk_mean = StatOp::ctr_desc_op<0>(exp_cnt, clk_cnt, clk_prior);
    const auto& clk_mode = StatOp::ctr_desc_op<1>(exp_cnt, clk_cnt, clk_prior);
    const auto& clk_median = StatOp::ctr_desc_op<2>(exp_cnt, clk_cnt, clk_prior);
    const auto& clk_var = StatOp::ctr_desc_op<3>(exp_cnt, clk_cnt, clk_prior);
    const auto& clk_skew = StatOp::ctr_desc_op<4>(exp_cnt, clk_cnt, clk_prior);
    const auto& clk_kurt = StatOp::ctr_desc_op<5>(exp_cnt, clk_cnt, clk_prior);

    const auto& base = StatOp::ctr_desc_op<0>(exp_spf.size(), clk_spf.size(), clk_prior);
    const auto& clk_mean_ratio = clk_spf.empty() ? 0 : clk_mean / base;

    tfrecord_parse_dump::dump_data("doc_device_clk_cnt", (double)clk_cnt, tf_exp);
    tfrecord_parse_dump::dump_data("doc_device_clk_ratio", clk_ratio, tf_exp);

    tfrecord_parse_dump::dump_data("doc_device_exp_cnt", (double)exp_cnt, tf_exp);
    tfrecord_parse_dump::dump_data("doc_device_exp_ratio", exp_ratio, tf_exp);

    tfrecord_parse_dump::dump_data("doc_device_clk_mean", clk_mean, tf_exp);
    tfrecord_parse_dump::dump_data("doc_device_clk_mode", clk_mode, tf_exp);
    tfrecord_parse_dump::dump_data("doc_device_clk_median", clk_median, tf_exp);
    tfrecord_parse_dump::dump_data("doc_device_clk_var", clk_var, tf_exp);
    tfrecord_parse_dump::dump_data("doc_device_clk_skew", clk_skew, tf_exp);
    tfrecord_parse_dump::dump_data("doc_device_clk_kurt", clk_kurt, tf_exp);

    tfrecord_parse_dump::dump_data("doc_device_clk_mean_ratio", clk_mean_ratio, tf_exp);
}

inline void doc_province_click_op(
    const auto& ctx,
    const auto& clk_seq_pf,
    const auto& exp_seq_pf,
    const auto& user_pf,
    const auto& clk_prior,
    auto& tf_exp
){
    const auto& clk_spf = clk_seq_pf->get(ctx.user_id).seq_st;
    const auto& exp_spf = exp_seq_pf->get(ctx.user_id).seq_st;
    provinceT key = user_pf->get(ctx.user_id).user_province;

    const auto& exp_cnt = std::ranges::count_if(exp_spf, [&](const auto& st){ return user_pf->get(std::get<0>(st)).user_province == key;});
    const auto& clk_cnt = std::ranges::count_if(clk_spf, [&](const auto& st){ return user_pf->get(std::get<0>(st)).user_province == key;});
    const auto& exp_ratio = exp_spf.empty() ? 0 : (double)exp_cnt / exp_spf.size();
    const auto& clk_ratio = clk_spf.empty() ? 0 : (double)clk_cnt / clk_spf.size();

    const auto& clk_mean = StatOp::ctr_desc_op<0>(exp_cnt, clk_cnt, clk_prior);
    const auto& clk_mode = StatOp::ctr_desc_op<1>(exp_cnt, clk_cnt, clk_prior);
    const auto& clk_median = StatOp::ctr_desc_op<2>(exp_cnt, clk_cnt, clk_prior);
    const auto& clk_var = StatOp::ctr_desc_op<3>(exp_cnt, clk_cnt, clk_prior);
    const auto& clk_skew = StatOp::ctr_desc_op<4>(exp_cnt, clk_cnt, clk_prior);
    const auto& clk_kurt = StatOp::ctr_desc_op<5>(exp_cnt, clk_cnt, clk_prior);

    const auto& base = StatOp::ctr_desc_op<0>(exp_spf.size(), clk_spf.size(), clk_prior);
    const auto& clk_mean_ratio = clk_spf.size() ? 0 : clk_mean / base;

    tfrecord_parse_dump::dump_data("doc_province_clk_cnt", (double)clk_cnt, tf_exp);
    tfrecord_parse_dump::dump_data("doc_province_clk_ratio", clk_ratio, tf_exp);

    tfrecord_parse_dump::dump_data("doc_province_exp_cnt", (double)exp_cnt, tf_exp);
    tfrecord_parse_dump::dump_data("doc_province_exp_ratio", exp_ratio, tf_exp);

    tfrecord_parse_dump::dump_data("doc_province_clk_mean", clk_mean, tf_exp);
    tfrecord_parse_dump::dump_data("doc_province_clk_mode", clk_mode, tf_exp);
    tfrecord_parse_dump::dump_data("doc_province_clk_median", clk_median, tf_exp);
    tfrecord_parse_dump::dump_data("doc_province_clk_var", clk_var, tf_exp);
    tfrecord_parse_dump::dump_data("doc_province_clk_skew", clk_skew, tf_exp);
    tfrecord_parse_dump::dump_data("doc_province_clk_kurt", clk_kurt, tf_exp);

    tfrecord_parse_dump::dump_data("doc_province_clk_mean_ratio", clk_mean_ratio, tf_exp);
}

inline void doc_city_click_op(
    const auto& ctx,
    const auto& clk_seq_pf,
    const auto& exp_seq_pf,
    const auto& user_pf,
    const auto& clk_prior,
    auto& tf_exp
){
    const auto& clk_spf = clk_seq_pf->get(ctx.user_id).seq_st;
    const auto& exp_spf = exp_seq_pf->get(ctx.user_id).seq_st;
    cityT key = user_pf->get(ctx.user_id).user_city;

    const auto& exp_cnt = std::ranges::count_if(exp_spf, [&](const auto& st){ return user_pf->get(std::get<0>(st)).user_city == key;});
    const auto& clk_cnt = std::ranges::count_if(clk_spf, [&](const auto& st){ return user_pf->get(std::get<0>(st)).user_city == key;});
    const auto& exp_ratio = exp_spf.empty() ? 0 : (double)exp_cnt / exp_spf.size();
    const auto& clk_ratio = clk_spf.empty() ? 0 : (double)clk_cnt / clk_spf.size();

    const auto& clk_mean = StatOp::ctr_desc_op<0>(exp_cnt, clk_cnt, clk_prior);
    const auto& clk_mode = StatOp::ctr_desc_op<1>(exp_cnt, clk_cnt, clk_prior);
    const auto& clk_median = StatOp::ctr_desc_op<2>(exp_cnt, clk_cnt, clk_prior);
    const auto& clk_var = StatOp::ctr_desc_op<3>(exp_cnt, clk_cnt, clk_prior);
    const auto& clk_skew = StatOp::ctr_desc_op<4>(exp_cnt, clk_cnt, clk_prior);
    const auto& clk_kurt = StatOp::ctr_desc_op<5>(exp_cnt, clk_cnt, clk_prior);

    const auto& base = StatOp::ctr_desc_op<0>(exp_spf.size(), clk_spf.size(), clk_prior);
    const auto& clk_mean_ratio = clk_spf.empty() ? 0 : clk_mean / base;

    tfrecord_parse_dump::dump_data("doc_city_clk_cnt", (double)clk_cnt, tf_exp);
    tfrecord_parse_dump::dump_data("doc_city_clk_ratio", clk_ratio, tf_exp);

    tfrecord_parse_dump::dump_data("doc_city_exp_cnt", (double)exp_cnt, tf_exp);
    tfrecord_parse_dump::dump_data("doc_city_exp_ratio", exp_ratio, tf_exp);

    tfrecord_parse_dump::dump_data("doc_city_clk_mean", clk_mean, tf_exp);
    tfrecord_parse_dump::dump_data("doc_city_clk_mode", clk_mode, tf_exp);
    tfrecord_parse_dump::dump_data("doc_city_clk_median", clk_median, tf_exp);
    tfrecord_parse_dump::dump_data("doc_city_clk_var", clk_var, tf_exp);
    tfrecord_parse_dump::dump_data("doc_city_clk_skew", clk_skew, tf_exp);
    tfrecord_parse_dump::dump_data("doc_city_clk_kurt", clk_kurt, tf_exp);

    tfrecord_parse_dump::dump_data("doc_city_clk_mean_ratio", clk_mean_ratio, tf_exp);
}

inline void doc_hour_click_op(
    const auto& ctx,
    const auto& clk_seq_pf,
    const auto& exp_seq_pf,
    const auto& user_pf,
    const auto& clk_prior,
    auto& tf_exp
){
    const auto& clk_spf = clk_seq_pf->get(ctx.user_id).seq_st;
    const auto& exp_spf = exp_seq_pf->get(ctx.user_id).seq_st;
    int32_t key = (int32_t)(ctx.ctx_exposedTime / 1000 / 3600 + 8) % 24;

    const auto& exp_cnt = ranges::count_if(exp_spf, [&](const auto& st){ return (int32_t)(std::get<1>(st) / 1000 / 3600 + 8) % 24 == key;});
    const auto& clk_cnt = ranges::count_if(clk_spf, [&](const auto& st){ return (int32_t)(std::get<1>(st) / 1000 / 3600 + 8) % 24 == key;});
    const auto& exp_ratio = exp_spf.empty() ? 0 : (double)exp_cnt / exp_spf.size();
    const auto& clk_ratio = clk_spf.empty() ? 0 : (double)clk_cnt / clk_spf.size();
    
    const auto& clk_mean = StatOp::ctr_desc_op<0>(exp_cnt, clk_cnt, clk_prior);
    const auto& clk_mode = StatOp::ctr_desc_op<1>(exp_cnt, clk_cnt, clk_prior);
    const auto& clk_median = StatOp::ctr_desc_op<2>(exp_cnt, clk_cnt, clk_prior);
    const auto& clk_var = StatOp::ctr_desc_op<3>(exp_cnt, clk_cnt, clk_prior);
    const auto& clk_skew = StatOp::ctr_desc_op<4>(exp_cnt, clk_cnt, clk_prior);
    const auto& clk_kurt = StatOp::ctr_desc_op<5>(exp_cnt, clk_cnt, clk_prior);

    const auto& base = StatOp::ctr_desc_op<0>(exp_spf.size(), clk_spf.size(), clk_prior);
    const auto& clk_mean_ratio = clk_spf.empty() ? 0 : clk_mean / base;

    tfrecord_parse_dump::dump_data("doc_hour_clk_cnt", (double)clk_cnt, tf_exp);
    tfrecord_parse_dump::dump_data("doc_hour_clk_ratio", clk_ratio, tf_exp);

    tfrecord_parse_dump::dump_data("doc_hour_exp_cnt", (double)exp_cnt, tf_exp);
    tfrecord_parse_dump::dump_data("doc_hour_exp_ratio", exp_ratio, tf_exp);

    tfrecord_parse_dump::dump_data("doc_hour_clk_mean", clk_mean, tf_exp);
    tfrecord_parse_dump::dump_data("doc_hour_clk_mode", clk_mode, tf_exp);
    tfrecord_parse_dump::dump_data("doc_hour_clk_median", clk_median, tf_exp);
    tfrecord_parse_dump::dump_data("doc_hour_clk_var", clk_var, tf_exp);
    tfrecord_parse_dump::dump_data("doc_hour_clk_skew", clk_skew, tf_exp);
    tfrecord_parse_dump::dump_data("doc_hour_clk_kurt", clk_kurt, tf_exp);
    tfrecord_parse_dump::dump_data("doc_hour_clk_mean_ratio", clk_mean_ratio, tf_exp);
}

inline void doc_refreshTimes_click_op(
    const auto& ctx,
    const auto& clk_seq_pf,
    const auto& exp_seq_pf,
    const auto& user_pf,
    const auto& clk_prior,
    auto& tf_exp
){
    const auto& clk_spf = clk_seq_pf->get(ctx.doc_id).seq_st;
    const auto& exp_spf = exp_seq_pf->get(ctx.doc_id).seq_st;

    const auto& exp_cnt = ranges::count_if(exp_spf, [&](const auto& st){ return std::get<3>(st) == ctx.ctx_refreshTimes;});
    const auto& clk_cnt = ranges::count_if(clk_spf, [&](const auto& st){ return std::get<3>(st)  == ctx.ctx_refreshTimes;});
    const auto& exp_ratio = exp_spf.empty() ? 0 : (double)exp_cnt / exp_spf.size();
    const auto& clk_ratio = clk_spf.empty() ? 0 : (double)clk_cnt / clk_spf.size();

    
    const auto& clk_mean = StatOp::ctr_desc_op<0>(exp_cnt, clk_cnt, clk_prior);
    const auto& clk_mode = StatOp::ctr_desc_op<1>(exp_cnt, clk_cnt, clk_prior);
    const auto& clk_median = StatOp::ctr_desc_op<2>(exp_cnt, clk_cnt, clk_prior);
    const auto& clk_var = StatOp::ctr_desc_op<3>(exp_cnt, clk_cnt, clk_prior);
    const auto& clk_skew = StatOp::ctr_desc_op<4>(exp_cnt, clk_cnt, clk_prior);
    const auto& clk_kurt = StatOp::ctr_desc_op<5>(exp_cnt, clk_cnt, clk_prior);

    const auto& base = StatOp::ctr_desc_op<0>(exp_spf.size(), clk_spf.size(), clk_prior);
    const auto& clk_mean_ratio = clk_spf.empty() ? 0 : clk_mean / base;

    tfrecord_parse_dump::dump_data("doc_refreshTimes_clk_cnt", (double)clk_cnt, tf_exp);
    tfrecord_parse_dump::dump_data("doc_refreshTimes_clk_ratio", clk_ratio, tf_exp);

    tfrecord_parse_dump::dump_data("doc_refreshTimes_exp_cnt", (double)exp_cnt, tf_exp);
    tfrecord_parse_dump::dump_data("doc_refreshTimes_exp_ratio", exp_ratio, tf_exp);

    tfrecord_parse_dump::dump_data("doc_refreshTimes_clk_mean", clk_mean, tf_exp);
    tfrecord_parse_dump::dump_data("doc_refreshTimes_clk_mode", clk_mode, tf_exp);
    tfrecord_parse_dump::dump_data("doc_refreshTimes_clk_median", clk_median, tf_exp);
    tfrecord_parse_dump::dump_data("doc_refreshTimes_clk_var", clk_var, tf_exp);
    tfrecord_parse_dump::dump_data("doc_refreshTimes_clk_skew", clk_skew, tf_exp);
    tfrecord_parse_dump::dump_data("doc_refreshTimes_clk_kurt", clk_kurt, tf_exp);
    tfrecord_parse_dump::dump_data("doc_refreshTimes_clk_mean_ratio", clk_mean_ratio, tf_exp);
}

};

};
