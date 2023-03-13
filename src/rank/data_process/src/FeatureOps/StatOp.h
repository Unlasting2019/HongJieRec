#pragma once
#include <vector>
#include <cmath>
#include <unordered_map>
#include <string>
#include <ranges>
#include <iostream>
#include <algorithm>
#include <numeric>

#include <cpp_tfrecord/dump_tfrecord.h>
#include <fmt/core.h>

#include "src/FeatureOps/FeatureOps.h"

using namespace NewsRec;

namespace FeatureOps{


namespace StatOp{

inline double watch_desc_op(
    const size_t& cnt,
    const double& watch_sum,
    const double& watch_prior
){
    return cnt == 0 ? watch_prior : (watch_sum + watch_prior * 100) / (100 + cnt);
}

inline double numeric_sum_op(
    auto rv
){
    double sum = 0;
    for(const auto& num : rv) sum += num;
    return sum;
}

template<size_t desc>
inline double ctr_desc_op(
    const size_t& exp_cnt,
    const size_t& clk_cnt,
    const double& clk_prior
){
    double N_prior = std::max(min_cnt-exp_cnt, (size_t)0);
    double alpha_prior = clk_prior * N_prior, beta_prior = (1-clk_prior) * N_prior;
    double alpha = alpha_prior + clk_cnt, beta = beta_prior + exp_cnt - clk_cnt;

    // 0 - mean, 1- mode, 2 - median, 3 - var, 4 - skew, 5 - kurt
    double num = 0, dem = 0;
    if constexpr(desc == 0){
        num = alpha;
        dem = alpha + beta;
    }
    else if constexpr(desc == 1){
        num = alpha - 1;
        dem = alpha + beta - 2;
    }
    else if constexpr(desc == 2){
        num = alpha - 1/3;
        dem = alpha + beta - 2/3;
    }
    else if constexpr(desc == 3){
        num = alpha * beta;
        dem = pow(alpha + beta, 2) * (alpha + beta + 1);
    }
    else if constexpr(desc == 4){
        num = 2*(beta-alpha)*(alpha+beta+1);
        dem = (alpha+beta+2)*sqrt(alpha*beta);
    }
    else{
        num = 6*pow(alpha-beta,2)*(alpha+beta+1) - alpha*beta*(alpha+beta+2);
        dem = alpha*beta*(alpha+beta+2)*(alpha+beta+3);
    }

    return dem == 0 ? 0 :  num / dem;
}

template<double default_=max_interval>
inline double time_since_first_op(
    const auto& ctx,
    auto rv
){
    double min_ = DBL_MAX;
    size_t cnt_ = 0;
    for(const auto& n : rv) min_ = std::min(n, min_), ++cnt_;
    return cnt_ == 0 ? default_ : ctx - min_;
}

template<double default_=max_interval>
inline double time_since_last_op(
    const auto& ctx,
    auto rv
){
    double max_ = DBL_MIN;
    size_t cnt_ = 0;
    for(const auto& n : rv) max_ = std::max(n, max_), ++cnt_;
    return cnt_ == 0 ? default_ : ctx - max_;
}

template<double default_=max_interval>
inline double time_since_mean_op(
    const auto& ctx,
    auto rv
){
    double sum_ = 0;
    size_t cnt_ = 0;
    for(const auto& n : rv) sum_ += n, cnt_++;
    return cnt_ == 0 ? default_: ctx - sum_ / cnt_;
}

template<double default_=max_interval>
inline double time_since_mean_std_op(
    const auto& ctx,
    auto rv
){
    double sum_ = 0;
    size_t cnt_ = 0;
    for(const auto& n : rv) sum_ += n, cnt_++;
    if(cnt_ == 0)
        return default_;
    else{
        sum_ = 0;
        double mean_ = sum_ / cnt_;
        size_t cnt_ = 0;
        for(const auto& n : rv) sum_ += std::pow(n-mean_, 2), cnt_++;
        return ctx - std::sqrt(sum_ / cnt_);
    }
}

template<double pre=diff_pre_default, double default_=max_interval>
inline double numeric_diff_mean_op(
    auto rv
){
    double pre_ = pre, sum_ = 0;
    size_t cnt_ = 0;
    for(const auto& n : rv)  sum_ += (n-pre_), cnt_++, pre_ = n;
    return cnt_ <= 1 ? default_: sum_ / cnt_;
}

template<double pre=diff_pre_default, double default_=max_interval>
inline double numeric_diff_std_op(
    auto rv
){
    double pre_ = pre, sum_ = 0, mean_ = numeric_diff_mean_op(rv);
    size_t cnt_ = 0;
    for(const auto& n : rv) sum_ += std::pow( (n-pre_)-mean_, 2), cnt_++, pre_ = n;
    return cnt_ <= 1 ? default_: std::sqrt(sum_ / cnt_);
}

}; // namespace Statop

}; // namespace FeatureOps
