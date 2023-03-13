#pragma once
#include <vector>
#include <ranges>
#include <cmath>
#include <unordered_map>
#include <string>

#include <cpp_tfrecord/dump_tfrecord.h>
#include <fmt/core.h>

#include "src/FeatureStruct.h"

using namespace NewsRec;

namespace FeatureOps{

// 序列类型
constexpr size_t exp_seq = 0;
constexpr int exp_seq_len = 2000;
constexpr size_t clk_seq = 1;
constexpr int clk_seq_len = 200;
// TimeOps
constexpr exposedTimeT max_interval  = 7 * 24 * 60 * 60;
constexpr exposedTimeT diff_pre_default = 1624464000.f;
// CTR
std::unordered_map<int32_t, double> clk_prior {
    {29, 0.14572840658873992},
    {28, 0.14772575305968433},
    {27, 0.1493655317279772},
    {26, 0.14629837155901632},
    {25, 0.15909779621521647},
    {24, 0.f},
};
std::unordered_map<int32_t, double> watch_prior {
    {29, 148.85053108914897},
    {28, 148.7646795739862},
    {27, 147.88873642131276},
    {26, 147.031405672395},
    {25, 169.69424021628038},
    {24, 0.f},
};
constexpr size_t min_cnt = 1000;


template<typename pfT, typename expT>
void dump_upf(
    const pfT& upf,
    expT& tf_exp
){
    tfrecord_parse_dump::dump_data("user_device", upf.user_device, tf_exp);
    tfrecord_parse_dump::dump_data("user_os", upf.user_os, tf_exp);
    tfrecord_parse_dump::dump_data("user_province", upf.user_province, tf_exp);
    tfrecord_parse_dump::dump_data("user_city", upf.user_city, tf_exp);
    tfrecord_parse_dump::dump_data<ageT>("user_age_key", upf.user_age | views::keys, tf_exp);
    tfrecord_parse_dump::dump_data<double>("user_age_val", upf.user_age | views::values, tf_exp);
    tfrecord_parse_dump::dump_data<genderT>("user_gender_key", upf.user_gender | views::keys, tf_exp);
    tfrecord_parse_dump::dump_data<double>("user_gender_val", upf.user_gender | views::values, tf_exp);
}

template<typename pfT, typename expT>
void dump_dpf(
    const pfT& dpf,
    expT& tf_exp
){
    tfrecord_parse_dump::dump_data<titleT>("doc_title_key", dpf.doc_title | views::keys, tf_exp);
    tfrecord_parse_dump::dump_data<double>("doc_title_val", dpf.doc_title | views::values, tf_exp);
    tfrecord_parse_dump::dump_data("doc_picNum", dpf.doc_picNum, tf_exp);
    tfrecord_parse_dump::dump_data("doc_cate1", dpf.doc_cate1, tf_exp);
    tfrecord_parse_dump::dump_data("doc_cate2", dpf.doc_cate2, tf_exp);
    tfrecord_parse_dump::dump_data<keywordsT>("doc_keywords_key", dpf.doc_keywords | views::keys, tf_exp);
    tfrecord_parse_dump::dump_data<double>("doc_keywords_val", dpf.doc_keywords | views::values, tf_exp);
}

template<typename pfT, typename expT>
void dump_cpf(
    const pfT& cpf,
    expT& tf_exp
){
    tfrecord_parse_dump::dump_data("user_id", cpf.user_id, tf_exp);
    tfrecord_parse_dump::dump_data("doc_id", cpf.doc_id, tf_exp);
    tfrecord_parse_dump::dump_data("ctx_network", cpf.ctx_network, tf_exp);
    tfrecord_parse_dump::dump_data("ctx_refreshTimes", cpf.ctx_refreshTimes, tf_exp);
    tfrecord_parse_dump::dump_data("is_click", cpf.is_click, tf_exp);
    tfrecord_parse_dump::dump_data("watch_time", cpf.watch_time == -1. ? 0 : cpf.watch_time, tf_exp);

    tfrecord_parse_dump::dump_data("ctx_hour", (int32_t)(cpf.ctx_exposedTime / 1000 / 3600 + 8)%24, tf_exp);
}

};
