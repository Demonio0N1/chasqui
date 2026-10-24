// Chasqui MHD — minimal INI-style parameter file reader with command-line overrides.
//
//   [block]
//   key = value      # comment
//
// Command line:  chasqui -i file.in [block.key=value ...]
#pragma once
#include <map>
#include <string>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <iostream>
#include <algorithm>
#include <cctype>

class Params {
 public:
  void ReadFile(const std::string &fname) {
    std::ifstream f(fname);
    if (!f) throw std::runtime_error("cannot open input file: " + fname);
    std::string line, block;
    while (std::getline(f, line)) {
      auto hash = line.find('#');
      if (hash != std::string::npos) line = line.substr(0, hash);
      line = Trim(line);
      if (line.empty()) continue;
      if (line.front() == '[' && line.back() == ']') {
        block = Trim(line.substr(1, line.size() - 2));
        continue;
      }
      auto eq = line.find('=');
      if (eq == std::string::npos) continue;
      std::string key = Trim(line.substr(0, eq));
      std::string val = Trim(line.substr(eq + 1));
      table_[block + "." + key] = val;
    }
  }
  // "block.key=value"
  void Override(const std::string &arg) {
    auto eq = arg.find('=');
    if (eq == std::string::npos) throw std::runtime_error("bad override: " + arg);
    table_[Trim(arg.substr(0, eq))] = Trim(arg.substr(eq + 1));
  }
  bool Has(const std::string &block, const std::string &key) const {
    return table_.count(block + "." + key) > 0;
  }
  std::string GetString(const std::string &b, const std::string &k) const {
    auto it = table_.find(b + "." + k);
    if (it == table_.end()) throw std::runtime_error("missing parameter " + b + "." + k);
    return it->second;
  }
  std::string GetOrAddString(const std::string &b, const std::string &k, const std::string &d) {
    if (!Has(b, k)) table_[b + "." + k] = d;
    return table_[b + "." + k];
  }
  double GetReal(const std::string &b, const std::string &k) const { return std::stod(GetString(b, k)); }
  double GetOrAddReal(const std::string &b, const std::string &k, double d) {
    if (!Has(b, k)) table_[b + "." + k] = ToStr(d);
    return std::stod(table_[b + "." + k]);
  }
  int GetInt(const std::string &b, const std::string &k) const { return std::stoi(GetString(b, k)); }
  int GetOrAddInt(const std::string &b, const std::string &k, int d) {
    if (!Has(b, k)) table_[b + "." + k] = std::to_string(d);
    return std::stoi(table_[b + "." + k]);
  }
  bool GetOrAddBool(const std::string &b, const std::string &k, bool d) {
    if (!Has(b, k)) table_[b + "." + k] = d ? "true" : "false";
    std::string v = table_[b + "." + k];
    std::transform(v.begin(), v.end(), v.begin(), ::tolower);
    return (v == "true" || v == "1" || v == "yes");
  }
  void Print(std::ostream &os) const {
    std::string last;
    for (auto &kv : table_) {
      auto dot = kv.first.find('.');
      std::string b = kv.first.substr(0, dot), k = kv.first.substr(dot + 1);
      if (b != last) { os << "[" << b << "]\n"; last = b; }
      os << "  " << k << " = " << kv.second << "\n";
    }
  }

 private:
  static std::string Trim(const std::string &s) {
    size_t a = 0, b = s.size();
    while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
    while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
    return s.substr(a, b - a);
  }
  static std::string ToStr(double d) { std::ostringstream o; o.precision(17); o << d; return o.str(); }
  std::map<std::string, std::string> table_;
};
