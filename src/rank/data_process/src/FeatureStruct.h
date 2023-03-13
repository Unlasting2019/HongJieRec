#pragma once
#include <tuple>
#include <iostream>
#include <stdio.h>
#include <unordered_map>
#include <vector>

#include <fmt/core.h>
#include <fmt/color.h>
#include <cpp_reflect/static_reflect.hpp>

using namespace std;

namespace NewsRec{

// user
using userT = int32_t;
using deviceT = int32_t;
using osT = int32_t;
using cityT = int32_t;
using provinceT = int32_t;
using ageT = int32_t;
using genderT = int32_t;
// doc
using docT = int32_t;
using cate1T = int32_t;
using cate2T = int32_t;
using titleT = int32_t;
using keywordsT = int32_t;
using postTimeT = double;
using picNumT = double;
// ctx
using exposedTimeT = double;
using networkT = int32_t;
using refreshTimesT = int32_t;
using exposedPosT = int32_t;
// label
using watchTimeT = float;
using isClickT = int32_t;
// seqT
using seqT = std::tuple<int32_t, exposedTimeT, watchTimeT, refreshTimesT>;


DEFINE_STRUCT(user_pf,
    (deviceT) user_device,
    (osT) user_os,
    (provinceT) user_province,
    (cityT) user_city,
    (unordered_map<ageT, float>) user_age,
    (unordered_map<genderT, float>) user_gender
);

DEFINE_STRUCT(doc_pf,
    (unordered_map<titleT, float>) doc_title,
    (postTimeT) doc_postTime,
    (picNumT) doc_picNum,
    (cate1T) doc_cate1,
    (cate2T) doc_cate2,
    (unordered_map<keywordsT, float>) doc_keywords
);

DEFINE_STRUCT(ctx_pf,
    (userT) user_id,
    (docT) doc_id,
    (exposedTimeT) ctx_exposedTime,
    (networkT) ctx_network,
    (refreshTimesT) ctx_refreshTimes,
    (exposedPosT) ctx_exposedPos,
    (isClickT) is_click,
    (watchTimeT) watch_time
);

DEFINE_STRUCT(seq_pf,
    (std::vector<seqT>) seq_st
);

DEFINE_STRUCT(data_option,
    (std::string) data_dir,
    (std::string) record_dir,
    (int32_t) record_size,
    (int32_t) thread_num,
    (int32_t) merge_day
);

void parse_option(data_option& opt, int argc, char** argv)
{
    for(int i=1; i<argc; ++i)
    {
        if(std::strcmp(argv[i], "--data_dir") == 0)
            opt.data_dir = std::string(argv[++i]);
        else if(std::strcmp(argv[i], "--record_dir") == 0)
            opt.record_dir = std::string(argv[++i]);
        else if(std::strcmp(argv[i], "--record_size") == 0)
            opt.record_size = atoi(argv[++i]);
        else if(std::strcmp(argv[i], "--thread_num") == 0)
            opt.thread_num = atoi(argv[++i]);
        else if(std::strcmp(argv[i], "--merge_day") == 0)
            opt.merge_day = atoi(argv[++i]);
    }

    forEach(opt, [&](const auto& fieldName, const auto& val){
        fmt::print("{}:{}\n",fieldName, val);
    });
};
};
