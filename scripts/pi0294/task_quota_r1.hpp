// Resource admission only. Does not change actuation, physics, pause or safety.
#pragma once
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

namespace gr00t_quota_r1 {
inline bool uint_text(const std::string& s, uint64_t& value) {
 if(s.empty() || s.size()>20)return false;
 value=0;
 for(char c:s) {
  if(c<'0' || c>'9')return false;
  unsigned digit=static_cast<unsigned>(c-'0');
  if(value>(std::numeric_limits<uint64_t>::max()-digit)/10)return false;
  value=value*10+digit;
 }
 return true;
}
inline bool env_equal(const char* key,const std::string& value) {
 const char* env=std::getenv(key);return env && value==env && !value.empty();
}
inline bool read_bound(uint64_t& bound) {
 const char* file=std::getenv("GR00T_TASK_QUOTA_BINDING");
 if(!file)return false;
 std::ifstream f(file);std::vector<std::string> lines;std::string line;
 while(std::getline(f,line))lines.push_back(line);
 if(!f.eof() || lines.size()!=7 || lines[0]!="GRTQUOTA2"
   || !env_equal("GR00T_TASK_RUN_UUID",lines[1])
   || !env_equal("GR00T_TASK_CARD_SHA256",lines[2])
   || !env_equal("GR00T_TASK_RESERVATION_ID",lines[3]))return false;
 uint64_t warmup,task;
 if(!uint_text(lines[4],bound) || !uint_text(lines[5],warmup) || !uint_text(lines[6],task)
   || warmup>5 || !(task==0 || task==32 || task==64 || task==96 || task==128)
   || bound==0 || bound>66503 || bound>(warmup+task)*500+3)return false;
 return true;
}
inline bool read_quota(const char* file,uint64_t& value) {
 uint64_t bound=0;if(!file || !read_bound(bound))return false;
 std::ifstream f(file);std::string token,extra;
 if(!(f>>token) || (f>>extra) || !uint_text(token,value))return false;
 return value>0 && value<=bound;
}
}
