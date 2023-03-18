#pragma once
#include <cstdint>
#include <algorithm>
#include <random>
#include <unordered_map>
#include <vector>

#include <fmt/format.h>
#include <fmt/ranges.h>
#include <fmt/core.h>

#include <cpp_str/parse_str.h>

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

template<typename DataT, typename IndexT>
class FeatureFrame{
public:
    inline void read_data(FILE* f_ptr, char* io_buffer, const size_t bf_size);
    inline const DataT& get(const IndexT& key) const;
    inline size_t size() const {return this->index_data.size();}

public:
    std::unordered_map<IndexT, DataT>::iterator begin() { return this->index_data.begin(); }
    std::unordered_map<IndexT, DataT>::iterator end() { return this->index_data.end(); }

    std::unordered_map<IndexT, DataT>::const_iterator begin() const { return this->index_data.cbegin(); }
    std::unordered_map<IndexT, DataT>::const_iterator end() const { return this->index_data.cend(); }

private:
    std::unordered_map<IndexT, DataT> index_data;
    DataT unk_data;
};

template<typename DataT, typename IndexT>
inline void FeatureFrame<DataT, IndexT>::read_data(
    FILE* f_ptr,
    char* io_buffer,
    const size_t bf_size
){
    IndexT index; DataT data;

    while(fgetc(f_ptr) != '\n');
    while(1){
        size_t ret = read_block_from_disk(io_buffer, bf_size, f_ptr);
        if(ret == 0) break;
        
        std::vector<std::pair<const char*, const char*>> field_ptr;
        for(char *p1=io_buffer, *p2=p1; ret>0; ret--, p1++){
            if(*p1 == '\t' || *p1 == '\n'){
                field_ptr.emplace_back(p2, p1);
                p2=p1+1;
            }
        } // read
        const char* p1=nullptr, *p2 = p1; 
        for(int i=0; i<field_ptr.size();){
            std::tie(p1, p2) = field_ptr[i++];
            str_parse_dump::parse_str(p1, p2, index);

            forEach(data, [&](const auto& fieldName, auto& value){
                std::tie(p1, p2) = field_ptr[i++];
                str_parse_dump::parse_str(p1, p2, value);
            });

            this->index_data[index] = data;
        } //  parse
    }
    /*
    forEach(data, [&](const auto& fieldName, auto& value){
        fmt::print("{}:{}\n",fieldName, value);
    });
    */
    // init unk_data
    forEach(this->unk_data, [&](const auto& fieldName, auto& value){
        str_parse_dump::parse_str(nullptr, nullptr, value);
    });
}

template<typename DataT, typename IndexT>
inline const DataT& FeatureFrame<DataT, IndexT>::get(
    const IndexT& key
) const {
    auto f_iter = this->index_data.find(key);
    return f_iter == this->index_data.end() ? this->unk_data : f_iter->second;
}

template<typename dataT>
class DataFrame{
public:
    vector<dataT>::iterator begin() { return frame_data.begin(); }
    vector<dataT>::iterator end() { return frame_data.end(); }
    vector<dataT>::const_iterator begin() const { return frame_data.cbegin(); }
    vector<dataT>::const_iterator end() const { return frame_data.cend(); }
    void shuffle_data() {
            std::shuffle(frame_data.begin(), frame_data.end(), std::default_random_engine {});
        }

public:
    inline void read_data(FILE* f_ptr, char* io_buffer, const size_t bf_size);
    inline size_t size() const {return this->frame_data.size();}

    const dataT& get(const size_t i) const {return frame_data[i];}

private:
    std::vector<dataT> frame_data;
};

template<typename dataT>
inline void DataFrame<dataT>::read_data(
    FILE* f_ptr, 
    char* io_buffer, 
    const size_t bf_size
){
    dataT data;

    while(fgetc(f_ptr) != '\n');
    while(1){
        size_t ret = read_block_from_disk(io_buffer, bf_size, f_ptr);
        if(ret == 0) break;
        
        std::vector<std::pair<const char*, const char*>> field_ptr;
        for(char *p1=io_buffer, *p2=p1; ret>0; ret--, p1++){
            if(*p1 == '\t' || *p1 == '\n'){
                field_ptr.emplace_back(p2, p1);
                p2=p1+1;
            }
        } // read
        
        const char* p1=nullptr, *p2 = p1; 
        for(int i=0; i<field_ptr.size();){
            forEach(data, [&](const auto& fieldName, auto& value){
                std::tie(p1, p2) = field_ptr[i++];
                str_parse_dump::parse_str(p1, p2, value);
            });
            this->frame_data.push_back(data);
        } //  parse
    }
}
