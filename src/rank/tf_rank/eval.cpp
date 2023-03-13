#include <iostream>
#include <algorithm>
#include <vector>
#include <unordered_map>
#include <stdio.h>

using namespace std;

inline 
void shrink_block_to_fit(
    char* block,
    FILE* f_ptr,
    size_t* ret
){
    size_t index = *ret-1;
    while(block[index] != '\n') index--;
    fseek(f_ptr, index-*ret+1, SEEK_CUR);
    *ret = index+1;
}

inline
size_t read_block_from_disk(
    char* block_buffer,
    uint32_t bf_size,
    FILE* f_ptr
){
    size_t ret = fread(block_buffer, 1, bf_size, f_ptr);
    if(ret == bf_size)
        shrink_block_to_fit(block_buffer, f_ptr, &ret);
    return ret;
}

struct y_true{
    int32_t user_id;
    int32_t doc_id;
    int32_t is_click;
    int32_t watch_time;
    y_true() : user_id(0), doc_id(0), is_click(0), watch_time(0) {}
};

struct y_pred{
    double is_click;
};

template<typename T>
void read_y_true(
    const char* fs,
    std::vector<T>& vec
){
    T data;
    FILE* f_ptr = fopen(fs, "r");
    char io_buffer[1<<20] = {'\0'};

    while(fgetc(f_ptr) != '\n');
    while(1){
        size_t ret = read_block_from_disk(io_buffer, 1<<20, f_ptr);
        if(ret == 0) break;

        size_t sz = 0; 
        for(char *p1=io_buffer, *p2=p1; ret>0; ret--, p1++){
            if(*p1 == ',' || *p1 == '\n'){
                if(sz % 4 == 0) data.user_id = strtol(p2, &p1, 0);
                else if(sz % 4 == 1) data.doc_id = strtol(p2, &p1, 0);
                else if(sz % 4 == 2) data.is_click = strtod(p2, &p1);
                else data.watch_time = strtod(p2, &p1);

                if(++sz % 4 == 0 )
                    vec.push_back(data);
                p2 = p1 +1;
            }
        } // read
    }
    printf("read y_true end - size:%ld\n", vec.size());
}

template<typename T>
void read_y_pred(
    const char* fs,
    std::vector<T>& vec
){
    T data;
    FILE* f_ptr = fopen(fs, "r");
    char io_buffer[1<<20] = {'\0'};

    while(fgetc(f_ptr) != '\n');
    while(1){
        size_t ret = read_block_from_disk(io_buffer, 1<<20, f_ptr);
        if(ret == 0) break;

        size_t sz = 0; 
        for(char *p1=io_buffer, *p2=p1; ret>0; ret--, p1++){
            if(*p1 == ',' || *p1 == '\n'){
                data.is_click = strtod(p2, &p1);
                vec.push_back(data);
                p2 = p1 +1;
            }
        } // read
    }
    printf("read y_pred end - size:%ld\n", vec.size());
}


template<typename T1, typename T2>
double cal_auc(
    const std::vector<T1>& y_true,
    const std::vector<T2>& y_pred,
    const size_t size
){
    std::vector<tuple<int32_t, double>> index_vec;
    for(int i=0; i<size; ++i)
        index_vec.emplace_back(y_true[i].is_click, y_pred[i].is_click);

    std::sort(index_vec.begin(), index_vec.end(), [](const auto& t1, const auto& t2){
        return std::get<1>(t1)  < std::get<1>(t2);
    });

    long pos_rank = 0;
    long pos_num = 0;
    for(int i=0; i<size; ++i)
        if(std::get<0>(index_vec[i]) == 1)
            pos_rank += (i+1), pos_num += 1;

    return (pos_rank -  1.f * pos_num * (pos_num+1) / 2)  / (pos_num * (size - pos_num));
}


template<typename T1, typename T2>
double cal_ugauc(
    const std::vector<T1>& y_true,
    const std::vector<T2>& y_pred,
    const size_t size
){

    std::unordered_map<int32_t, std::vector<int64_t>> groups;
    for(int i=0; i<size; ++i)
        groups[y_true[i].user_id].push_back(i);

    const auto& check_fn = [&](const std::vector<int64_t>& vec){
        bool one = false, zero = false;
        for(const auto& i : vec)
            if(y_true[i].is_click == 1) one = true;
            else zero = true;

        return one && zero;
    };


    double auc = 0;
    long cnt = 0;
    for(const auto&[k, v] : groups)
        if(check_fn(v)){
            std::vector<T1> y_true_;
            std::vector<T2> y_pred_;
            for(const auto& i : v)
                y_true_.push_back(y_true[i]), y_pred_.push_back(y_pred[i]);

            auc += cal_auc(y_true_, y_pred_, y_true_.size());
            cnt += 1;
        }

    return cnt == 0 ? 0.5 : auc / cnt;
}

template<typename T1, typename T2>
double cal_dgauc(
    const std::vector<T1>& y_true,
    const std::vector<T2>& y_pred,
    const size_t size
){

    std::unordered_map<int32_t, std::vector<int64_t>> groups;
    for(int i=0; i<size; ++i)
        groups[y_true[i].doc_id].push_back(i);

    const auto& check_fn = [&](const std::vector<int64_t>& vec){
        bool one = false, zero = false;
        for(const auto& i : vec)
            if(y_true[i].is_click == 1) one = true;
            else zero = true;

        return one && zero;
    };


    double auc = 0;
    long cnt = 0;
    for(const auto&[k, v] : groups)
        if(check_fn(v)){
            std::vector<T1> y_true_;
            std::vector<T2> y_pred_;
            for(const auto& i : v)
                y_true_.push_back(y_true[i]), y_pred_.push_back(y_pred[i]);

            auc += cal_auc(y_true_, y_pred_, y_true_.size());
            cnt += 1;
        }

    return cnt == 0 ? 0.5 : auc / cnt;
}

int main(int argv, char **argc){
    std::vector<y_true> y_true_vec;
    std::vector<y_pred> y_pred_vec;
    read_y_true(argc[1], y_true_vec);
    read_y_pred(argc[2], y_pred_vec);
    size_t sz = min(y_true_vec.size(), y_pred_vec.size());
    // auc & u_gauc & d_gauc
    printf("%s_auc:%.15lf\n", argc[3],cal_auc(y_true_vec, y_pred_vec, sz));
    printf("%s_ugauc:%.15lf\n", argc[3], cal_ugauc(y_true_vec, y_pred_vec, sz));
    printf("%s_dgauc:%.15lf\n", argc[3], cal_dgauc(y_true_vec, y_pred_vec, sz));
}
