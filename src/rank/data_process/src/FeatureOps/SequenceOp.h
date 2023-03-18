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

namespace SequenceOp{

template<size_t seq_type, int maxlen, size_t min_watch_time>
inline void sequence_hour_op(
    const auto& ctx,
    const auto& seq_pf,
    const auto& doc_pf,
    auto& tf_exp
){
    // 从current开始截断序列 
    const auto& spf = seq_pf->get(ctx.user_id);
    const int diff = std::max((int)spf.seq_st.size() - maxlen, 0);
    auto rv = ranges::subrange(spf.seq_st.cbegin()+diff, spf.seq_st.cend()) 
        | views::filter([](const auto& st){return std::get<2>(st) > min_watch_time;})
        | views::elements<1> 
        | views::transform([](const auto& et){return (int32_t)(et / 1000 / 3600 + 8) % 24 + 2;})
        | views::take(maxlen);

    if constexpr(seq_type == clk_seq)
        tfrecord_parse_dump::dump_data<int32_t>("clk_hour", rv, tf_exp);
    else 
        tfrecord_parse_dump::dump_data<int32_t>("exp_hour", rv, tf_exp);
}

template<size_t seq_type, int maxlen, size_t min_watch_time>
inline void sequence_refreshTimes_op(
    const auto& ctx,
    const auto& seq_pf,
    const auto& doc_pf,
    auto& tf_exp
){
    // 从current开始截断序列 
    const auto& spf = seq_pf->get(ctx.user_id);
    const int diff = std::max((int)spf.seq_st.size() - maxlen, 0);
    auto rv = ranges::subrange(spf.seq_st.cbegin()+diff, spf.seq_st.cend()) 
        | views::filter([](const auto& st){return std::get<2>(st) > min_watch_time;})
        | views::elements<3> 
        | views::transform([](const auto& rt){ return rt + 2;})
        | views::take(maxlen);

    if constexpr(seq_type == clk_seq)
        tfrecord_parse_dump::dump_data<refreshTimesT>("clk_refreshTimes", rv, tf_exp);
    else 
        tfrecord_parse_dump::dump_data<refreshTimesT>("exp_refreshTimes", rv, tf_exp);
}

template<size_t seq_type, int maxlen, size_t min_watch_time>
inline void sequence_watchTime_op(
    const auto& ctx,
    const auto& seq_pf,
    const auto& doc_pf,
    auto& tf_exp
){
    // 从current开始截断序列 
    const auto& spf = seq_pf->get(ctx.user_id);
    const int diff = std::max((int)spf.seq_st.size() - maxlen, 0);
    auto rv = ranges::subrange(spf.seq_st.cbegin()+diff, spf.seq_st.cend()) 
        | views::filter([](const auto& st){return std::get<2>(st) > min_watch_time;})
        | views::elements<2> 
        | views::take(maxlen);

    if constexpr(seq_type == clk_seq)
        tfrecord_parse_dump::dump_data<watchTimeT>("clk_watchTime", rv, tf_exp);
    else 
        tfrecord_parse_dump::dump_data<watchTimeT>("exp_watchTime", rv, tf_exp);
}

template<size_t seq_type, int maxlen, size_t min_watch_time>
inline void sequence_doc_op(
    const auto& ctx,
    const auto& seq_pf,
    const auto& doc_pf,
    auto& tf_exp
){
    // 从current开始截断序列 
    const auto& spf = seq_pf->get(ctx.user_id);
    const int diff = std::max((int)spf.seq_st.size() - maxlen, 0);
    auto rv = ranges::subrange(spf.seq_st.cbegin()+diff, spf.seq_st.cend()) 
        | views::filter([](const auto& st){return std::get<2>(st) > min_watch_time;})
        | views::elements<0> 
        | views::take(maxlen);

    if constexpr(seq_type == clk_seq)
        tfrecord_parse_dump::dump_data<docT>("clk_doc_id", rv, tf_exp);
    else 
        tfrecord_parse_dump::dump_data<docT>("exp_doc_id", rv, tf_exp);
}

template<size_t seq_type, int maxlen, size_t min_watch_time>
inline void sequence_cate1_op(
    const auto& ctx,
    const auto& seq_pf,
    const auto& doc_pf,
    auto& tf_exp
){
    // 从current开始截断序列 
    const auto& spf = seq_pf->get(ctx.user_id);
    const int diff = std::max((int)spf.seq_st.size() - maxlen, 0);
    auto rv = ranges::subrange(spf.seq_st.cbegin()+diff, spf.seq_st.cend()) 
        | views::filter([](const auto& st){return std::get<2>(st) > min_watch_time;})
        | views::elements<0> 
        | views::transform([&](const auto& d){return doc_pf->get(d).doc_cate1;})
        | views::take(maxlen);

    if constexpr(seq_type == clk_seq)
        tfrecord_parse_dump::dump_data<cate1T>("clk_doc_cate1", rv, tf_exp);
    else 
        tfrecord_parse_dump::dump_data<cate1T>("exp_doc_cate1", rv, tf_exp);
}
template<size_t seq_type, int maxlen, size_t min_watch_time>
inline void sequence_cate2_op(
    const auto& ctx,
    const auto& seq_pf,
    const auto& doc_pf,
    auto& tf_exp
){
    // 从current开始截断序列 
    const auto& spf = seq_pf->get(ctx.user_id);
    const int diff = std::max((int)spf.seq_st.size() - maxlen, 0);
    auto rv = ranges::subrange(spf.seq_st.cbegin()+diff, spf.seq_st.cend()) 
        | views::filter([](const auto& st){return std::get<2>(st) > min_watch_time;})
        | views::elements<0> 
        | views::transform([&](const auto& d){return doc_pf->get(d).doc_cate2;})
        | views::take(maxlen);

    if constexpr(seq_type == clk_seq)
        tfrecord_parse_dump::dump_data<cate2T>("clk_doc_cate2", rv, tf_exp);
    else 
        tfrecord_parse_dump::dump_data<cate2T>("exp_doc_cate2", rv, tf_exp);
}

template<size_t seq_type, int maxlen, size_t min_watch_time>
inline void sequence_cate1_cnt_op(
    const auto& ctx,
    const auto& seq_pf,
    const auto& doc_pf,
    auto& tf_exp
){
    const auto& spf = seq_pf->get(ctx.user_id);
    // 生成cnt map
    std::unordered_map<cate1T, float> cate1_cnt;
    for(const auto& cate1 : spf.seq_st 
            | views::elements<0>
            | views::transform([&](const auto& d){return doc_pf->get(d).doc_cate1;}))
        cate1_cnt[cate1] += 1;
    // 从current开始截断序列 
    const int diff = std::max((int)spf.seq_st.size() - maxlen, 0);
    auto rv = ranges::subrange(spf.seq_st.cbegin()+diff, spf.seq_st.cend()) 
        | views::filter([](const auto& st){return std::get<2>(st) > min_watch_time;})
        | views::elements<0> 
        | views::transform([&](const auto& d){return doc_pf->get(d).doc_cate1;})
        | views::transform([&](const auto& d){return cate1_cnt[d];})
        | views::take(maxlen);

    if constexpr(seq_type == clk_seq)
        tfrecord_parse_dump::dump_data<float>("clk_doc_cate1_cnt", rv, tf_exp);
    else 
        tfrecord_parse_dump::dump_data<float>("exp_doc_cate1_cnt", rv, tf_exp);
}

template<size_t seq_type, int maxlen, size_t min_watch_time>
inline void sequence_cate2_cnt_op(
    const auto& ctx,
    const auto& seq_pf,
    const auto& doc_pf,
    auto& tf_exp
){
    const auto& spf = seq_pf->get(ctx.user_id);
    // 生成cnt map
    std::unordered_map<cate2T, float> cate2_cnt;
    for(const auto& cate2 : spf.seq_st 
            | views::elements<0>
            | views::transform([&](const auto& d){return doc_pf->get(d).doc_cate2;}))
        cate2_cnt[cate2] += 1;
    // 从current开始截断序列 
    const int diff = std::max((int)spf.seq_st.size() - maxlen, 0);
    auto rv = ranges::subrange(spf.seq_st.cbegin()+diff, spf.seq_st.cend()) 
        | views::filter([](const auto& st){return std::get<2>(st) > min_watch_time;})
        | views::elements<0> 
        | views::transform([&](const auto& d){return doc_pf->get(d).doc_cate2;})
        | views::transform([&](const auto& d){return cate2_cnt[d];})
        | views::take(maxlen);

    if constexpr(seq_type == clk_seq)
        tfrecord_parse_dump::dump_data<float>("clk_doc_cate2_cnt", rv, tf_exp);
    else 
        tfrecord_parse_dump::dump_data<float>("exp_doc_cate2_cnt", rv, tf_exp);
}

template<size_t seq_type, int maxlen, size_t min_watch_time>
inline void sequence_doc_cnt_op(
    const auto& ctx,
    const auto& seq_pf,
    const auto& doc_pf,
    auto& tf_exp
){
    const auto& spf = seq_pf->get(ctx.user_id);
    // 生成cnt map
    std::unordered_map<docT, float> doc_id_cnt;
    for(const auto& doc_id : spf.seq_st 
            | views::elements<0>)
        doc_id_cnt[doc_id] += 1;
    // 从current开始截断序列 
    const int diff = std::max((int)spf.seq_st.size() - maxlen, 0);
    auto rv = ranges::subrange(spf.seq_st.cbegin()+diff, spf.seq_st.cend()) 
        | views::filter([](const auto& st){return std::get<2>(st) > min_watch_time;})
        | views::elements<0> 
        | views::transform([&](const auto& d){return doc_id_cnt[d];})
        | views::take(maxlen);

    if constexpr(seq_type == clk_seq)
        tfrecord_parse_dump::dump_data<float>("clk_doc_id_cnt", rv, tf_exp);
    else 
        tfrecord_parse_dump::dump_data<float>("exp_doc_id_cnt", rv, tf_exp);
}

template<size_t seq_type, int maxlen, size_t min_watch_time, size_t doc_num>
inline void sequence_doc_neg_op(
    const auto& ctx,
    const auto& seq_pf,
    const auto& doc_pf,
    auto& tf_exp
){
    const auto& spf = seq_pf->get(ctx.user_id);
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> dist(0, doc_num);
    // 从current开始截断序列 
    const int diff = std::max((int)spf.seq_st.size() - maxlen, 0);
    auto rv = ranges::subrange(spf.seq_st.cbegin()+diff, spf.seq_st.cend()) 
        | views::filter([](const auto& st){return std::get<2>(st) > min_watch_time;})
        | views::elements<0> 
        | views::transform([&](const auto& d){return dist(rng);})
        | views::take(maxlen);

    if constexpr(seq_type == clk_seq)
        tfrecord_parse_dump::dump_data<docT>("clk_doc_neg", rv, tf_exp);
    else 
        tfrecord_parse_dump::dump_data<docT>("exp_doc_neg", rv, tf_exp);
}

}; // namespace SequenceOp

}; // FeatureOps
