#include <time.h>
#include <iostream>
#include <unordered_map>
#include <vector>

#include <fmt/core.h>
#include <fmt/format.h>
#include <fmt/ranges.h>

#include "src/FeatureStruct.h"
#include "src/FeatureProfile.h"

using namespace std;
using namespace NewsRec;

int main(int argc, char** argv){
    fmt::print(fg(fmt::color::steel_blue) | fmt::emphasis::italic, "\n------------ start parse_option -----------\n");
    data_option opt;
    parse_option(opt, argc, argv);
    /* 定义 */
    FeatureFrame<user_pf, userT>                              user_f;
    FeatureFrame<doc_pf,  docT>                               doc_f;
    FeatureFrame<seq_pf, userT>                               user_clk_f;
    FeatureFrame<seq_pf, userT>                               user_exp_f;
    FeatureFrame<seq_pf, docT>                                doc_clk_f;
    FeatureFrame<seq_pf, docT>                                doc_exp_f;
    DataFrame<ctx_pf>                                         log_data;
    FeatureProfile<1024 * 1024 * 10> Fp(opt, &user_f, &doc_f, &user_clk_f, &user_exp_f, &doc_clk_f, &doc_exp_f);
    fmt::print(fg(fmt::color::steel_blue) | fmt::emphasis::italic, "------------ start read_feature - merge_day:{}-----------\n", opt.merge_day);
    Fp.read_static(user_f, "processed_user_info.csv");
    Fp.read_static(doc_f,  "processed_doc_info.csv");
    Fp.read_dynamic(user_clk_f, "clk_user_seq.csv");
    Fp.read_dynamic(user_exp_f, "exp_user_seq.csv");
    Fp.read_dynamic(doc_clk_f, "clk_doc_seq.csv");
    Fp.read_dynamic(doc_exp_f, "exp_doc_seq.csv");
    Fp.read_dynamic(log_data, "processed_train_info.csv");
    /* 预处理 */
    if(opt.merge_day != 29)
        log_data.shuffle_data();
    /* serializeObj */
    fmt::print(fg(fmt::color::steel_blue) | fmt::emphasis::italic, "------------ start dump_nn_data - merge_day:{}-----------\n", opt.merge_day);
    Fp.dump_tfrecord(log_data);
}
