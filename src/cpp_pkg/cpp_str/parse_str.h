#pragma once
#include <CharConv/CharConv.h>
#include <fast_float/fast_float.h>

#include <tuple>
#include <unordered_map>
#include <vector>
#include <iostream>


namespace str_parse_dump{

inline 
void parse_str(const char* start, const char* end, int32_t& value){
    if(start == end)
        value = 1;
    else
        fast_int::from_chars(start, end, value);
}


inline 
void parse_str(const char* start, const char* end, int64_t& value){
    if(start == end)
        value = 1;
    else
        fast_int::from_chars(start, end, value);
}

inline 
void parse_str(const char* start, const char* end, float& value){
    if(start == end)
        value = 0.f;
    else
        fast_float::from_chars(start, end, value);
}

inline
void parse_str(const char* start, const char* end, double& value){
    if(start == end)
        value = 0.f;
    else
        fast_float::from_chars(start, end, value);
}


template<typename T, size_t N, size_t fieldNum>
inline void parse_tp(
    const char* start,
    const char* end,
    T& tp
){
    if constexpr(N != 0){
        for(const char* p = start; ;++p){
            if(p == end || *p == '#'){
                parse_str(start, p, std::get<fieldNum-N>(tp));
                parse_tp<T, N-1, fieldNum>(p+1, end, tp);
                return;
            } // if
        } // for
    }// if
}


template<typename... Args>
inline void parse_str(
    const char* start,
    const char* end,
    std::tuple<Args...>& tp
){
    if(start == end){
        tp = {};
        return;
    }

    parse_tp<decltype(tp), sizeof...(Args), sizeof...(Args)>(start, end, tp);
}


template<typename T> 
inline void parse_str(
    const char* start, 
    const char* end, 
    std::vector<T>& val_vec,
    const char sep='|'
){
    if(start == end){
        std::vector<T>().swap(val_vec);
        return;
    }

    val_vec.clear();
    T val;
    for(const char *p1=start, *p2=p1; p1<=end; ++p1){
        if(p1 == end || *p1 == sep){
            parse_str(p2, p1, val);
            p2 = p1 + 1;
            val_vec.push_back(val);
        }
    }
    val_vec.shrink_to_fit();
}

template<typename T1, typename T2>
inline void parse_str(
    const char* start, 
    const char* end, 
    std::unordered_map<T1, T2>& T_map,
    const char kv_sep=':',
    const char v_sep=','
){
    if(start == end){
        std::unordered_map<T1, T2>().swap(T_map);
        return;
    }
    
    T_map.clear();
    T1 key;
    T2 val;
    for(const char *p1=start, *p2=p1; p1 <= end; ++p1){
        if(p1 == end || *p1 == v_sep){
            parse_str(p2, p1, val);
            p2 = p1 + 1;
            T_map.emplace(key, val);
        } else if(*p1 == kv_sep){
            parse_str(p2, p1, key);
            p2 = p1+1;
        }
    }

}

};

