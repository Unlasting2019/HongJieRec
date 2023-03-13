#pragma once
#include <thread>
#include <algorithm>
#include <string>
#include <unordered_map>
#include <time.h>
#include <dirent.h>
#include <ranges>

#include <fmt/core.h>
#include <fmt/ranges.h>
#include <fmt/format.h>
#include <fmt/color.h>
#include <cpp_tfrecord/example.pb.h>
#include <cpp_tfrecord/feature.pb.h>
#include <cpp_tfrecord/record_writer.h>
#include <cpp_frame/feature_frame.h>

#include "src/FeatureStruct.h"
#include "src/FeatureOps/FeatureOps.h"
#include "src/FeatureOps/TimeOp.h"
#include "src/FeatureOps/MatchOp.h"
#include "src/FeatureOps/StatOp.h"
#include "src/FeatureOps/SequenceOp.h"
#include "src/FeatureOps/ProductOp.h"

using namespace std;
using namespace std::chrono;
using namespace FeatureOps;

namespace NewsRec{


template<size_t buffer_size>
class FeatureProfile
{
public:
    FeatureProfile(
        data_option opt,
        const FeatureFrame<user_pf, userT>* user_f_,
        const FeatureFrame<doc_pf, userT>* doc_f_,
        const FeatureFrame<seq_pf, userT>* user_clk_seq_f_,
        const FeatureFrame<seq_pf, userT>* user_exp_seq_f_,
        const FeatureFrame<seq_pf, docT>*  doc_clk_seq_f_,
        const FeatureFrame<seq_pf, docT>*  doc_exp_seq_f_
    ) : 
        user_pf_(user_f_), 
        doc_pf_(doc_f_), 
        user_clk_pf_(user_clk_seq_f_), 
        user_exp_pf_(user_exp_seq_f_),
        doc_clk_pf_(doc_clk_seq_f_), 
        doc_exp_pf_(doc_exp_seq_f_)
    {
        data_dir = opt.data_dir;
        record_dir = opt.record_dir;
        record_size = opt.record_size;
        merge_day = opt.merge_day;
        thread_num = opt.thread_num;
        clk_prior = FeatureOps::clk_prior[merge_day];
        watch_prior = FeatureOps::watch_prior[merge_day];
    }

    template<typename pfT>
    void read_static(pfT& pf, const char* f_name){
	    auto beg_t = system_clock::now();       //开始时间
        auto fs = fmt::format("{}/{}", this->data_dir, f_name);
        FILE* f_ptr = fopen(fs.c_str(), "r");
        if(!f_ptr){
            fmt::print(fg(fmt::color::crimson) | fmt::emphasis::bold, "read_feature:{}\t - fs nullptr\n", fs);
            return;
        }
        
        pf.read_data(f_ptr, io_buffer, buffer_size);
        fmt::print("read_feature:{} - end size:{} cost:{}s\n", fs, pf.size(), static_cast<duration<double>>(system_clock::now() - beg_t).count());
    };

    template<typename pfT>
    void read_dynamic(pfT& pf, const char* f_name){
	    auto beg_t = system_clock::now();       //开始时间
        auto fs = fmt::format("{}/{}/day={}", data_dir, f_name, merge_day);
        DIR* dir = opendir(fs.c_str());
        struct dirent *ent;
        if(!dir){
            fmt::print(fg(fmt::color::crimson) | fmt::emphasis::bold, "read_feature:{}\t - fs nullptr\n", fs);
            return;
        }
        
        while((ent = readdir(dir)) != nullptr){
            if(!strcmp(ent->d_name,".") \
                ||!strcmp(ent->d_name,"..") \
                || ent->d_name[0] == '.') continue;

            string fs_ = fmt::format("{}/{}", fs, ent->d_name);
            FILE* f_ptr = fopen(fs_.c_str(), "r");
            pf.read_data(f_ptr, io_buffer, buffer_size);
            fclose(f_ptr);
        }

        closedir(dir);
        fmt::print("read_feature:{} - end size:{} cost:{}s\n", fs, pf.size(), static_cast<duration<double>>(system_clock::now() - beg_t).count());
    }

    template<typename ctxT, typename expT>
    inline void FeatureOps(
        const ctxT& ctx,
        expT& tf_exp
    ){
        // BaseDump
        FeatureOps::dump_upf(user_pf_->get(ctx.user_id), tf_exp);
        FeatureOps::dump_dpf(doc_pf_->get(ctx.doc_id), tf_exp);
        FeatureOps::dump_cpf(ctx, tf_exp);
        // SequenceOp
        SequenceOp::sequence_doc_op<clk_seq, clk_seq_len, 1>(ctx, user_clk_pf_, doc_pf_, tf_exp);
        SequenceOp::sequence_cate1_op<clk_seq, clk_seq_len, 1>(ctx, user_clk_pf_, doc_pf_, tf_exp);
        SequenceOp::sequence_cate2_op<clk_seq, clk_seq_len, 1>(ctx, user_clk_pf_, doc_pf_, tf_exp);
        SequenceOp::sequence_watchTime_op<clk_seq, clk_seq_len, 1>(ctx, user_clk_pf_, doc_pf_, tf_exp);
        SequenceOp::sequence_refreshTimes_op<clk_seq, clk_seq_len, 1>(ctx, user_clk_pf_, doc_pf_, tf_exp);
        SequenceOp::sequence_doc_cnt_op<clk_seq, clk_seq_len, 1>(ctx, user_clk_pf_, doc_pf_, tf_exp);
        SequenceOp::sequence_cate1_cnt_op<clk_seq, clk_seq_len, 1>(ctx, user_clk_pf_, doc_pf_, tf_exp);
        SequenceOp::sequence_cate2_cnt_op<clk_seq, clk_seq_len, 1>(ctx, user_clk_pf_, doc_pf_, tf_exp);
        // MatchOp
        MatchOp::user_click_op(ctx, user_clk_pf_, user_exp_pf_, clk_prior, tf_exp);
        MatchOp::user_watch_op(ctx, user_clk_pf_, watch_prior, tf_exp);
        MatchOp::user_cate1_click_op(ctx, user_clk_pf_, user_exp_pf_, doc_pf_, clk_prior, tf_exp);
        MatchOp::user_cate1_watch_op(ctx, user_clk_pf_, doc_pf_, watch_prior, tf_exp);
        MatchOp::user_cate2_click_op(ctx, user_clk_pf_, user_exp_pf_, doc_pf_, clk_prior, tf_exp);
        MatchOp::user_cate2_watch_op(ctx, user_clk_pf_, doc_pf_, watch_prior, tf_exp);
        MatchOp::user_title_click_op(ctx, user_clk_pf_, user_exp_pf_, doc_pf_, clk_prior, tf_exp);
        MatchOp::user_title_watch_op(ctx, user_clk_pf_, doc_pf_, watch_prior, tf_exp);
        MatchOp::doc_click_op(ctx, doc_clk_pf_, doc_exp_pf_, clk_prior, tf_exp);
        MatchOp::doc_watch_op(ctx, doc_clk_pf_, watch_prior, tf_exp);
        MatchOp::doc_refreshTimes_click_op(ctx, doc_clk_pf_, doc_exp_pf_, user_pf_, clk_prior, tf_exp);
        MatchOp::doc_refreshTimes_watch_op(ctx, doc_clk_pf_, user_pf_, clk_prior, tf_exp);
        // TimeOp - time_since
        TimeOp::user_time_since_op<clk_seq>(ctx, user_clk_pf_, tf_exp);
        TimeOp::user_cate1_time_since_op<clk_seq>(ctx, user_clk_pf_, doc_pf_, tf_exp);
        TimeOp::user_cate2_time_since_op<clk_seq>(ctx, user_clk_pf_, doc_pf_, tf_exp);
        TimeOp::doc_time_since_op<clk_seq>(ctx, doc_clk_pf_, tf_exp);
        // TimeOp - time_diff
        TimeOp::user_time_diff_op<clk_seq>(ctx, user_clk_pf_, tf_exp);
        TimeOp::user_cate1_time_diff_op<clk_seq>(ctx, user_clk_pf_, doc_pf_, tf_exp);
        TimeOp::user_cate2_time_diff_op<clk_seq>(ctx, user_clk_pf_, doc_pf_, tf_exp);
        TimeOp::doc_time_diff_op<clk_seq>(ctx, doc_clk_pf_, tf_exp);
        // ProductOp 
        ProductOp::sequence_doc_product_op<clk_seq, 1>(ctx, user_clk_pf_, doc_pf_, tf_exp);
        ProductOp::user_cate1_product_op(ctx, doc_pf_, tf_exp);
        ProductOp::user_cate2_product_op(ctx, doc_pf_, tf_exp);
    }

    template<typename ctxT> 
    inline void dump_tfrecord_(
        const ctxT& ctx_data,
        const size_t start,
        const size_t end
    ){
        tensorflow::SequenceExample seq_exp;
        size_t sz=0, n=ctx_data.size();

        std::string fs, str;
        tfrecord::RecordWriter rw;
        for(int i=start; i<end && i<n; ++i){
            const auto& ctx = ctx_data.get(i);
            this->FeatureOps(ctx, seq_exp);

            if(++sz % record_size == 0){
                fs = fmt::format("{}/{}_{}", record_dir, merge_day, i);
                seq_exp.SerializeToString(&str);
                rw.WriteRecord(fs, str.data(), str.size());
                seq_exp.clear_context();
                seq_exp.clear_feature_lists();
            }
        }

        if(sz % record_size != 0){
            for(int i=start; sz++%record_size != 0; ++i){
                const auto& ctx = ctx_data.get(i);
                this->FeatureOps(ctx, seq_exp);
            }
            fs = fmt::format("{}/{}_{}", record_dir, merge_day, end);
            seq_exp.SerializeToString(&str);
            rw.WriteRecord(fs, str.data(), str.size());
        }
    }

    template<typename ctxT> 
    inline void dump_tfrecord(
        const ctxT& ctx_data
    ){
        auto beg_t = system_clock::now();
        std::vector<std::thread> thread_pool;
        thread_pool.reserve(thread_num);

        int block_size = ctx_data.size() / thread_num;
        int diff_block_size = block_size + record_size - block_size % record_size;
        for(int i=0; i<thread_num; ++i){
            thread_pool.emplace_back(std::thread{
                &FeatureProfile::dump_tfrecord_<ctxT>,
                this,
                std::cref(ctx_data), 
                i*diff_block_size, (i+1)*diff_block_size
            });
        }

        for(int i=0; i<thread_pool.size(); ++i)
            thread_pool[i].join();
        
        fmt::print("dump_tfrecord - end cost:{}s\n", static_cast<duration<double>>(system_clock::now() - beg_t).count());
    }

private:
    /* 参数配置 */
    std::string data_dir, record_dir;
    int32_t record_size, thread_num, merge_day;
    char* io_buffer = new char[buffer_size];
    float clk_prior, watch_prior;
    /* 特征Profile */
    const FeatureFrame<user_pf, userT>* user_pf_;
    const FeatureFrame<doc_pf, userT>* doc_pf_;
    const FeatureFrame<seq_pf, userT>* user_clk_pf_;
    const FeatureFrame<seq_pf, userT>* user_exp_pf_;
    const FeatureFrame<seq_pf, docT>*  doc_clk_pf_;
    const FeatureFrame<seq_pf, docT>*  doc_exp_pf_;
};

};
