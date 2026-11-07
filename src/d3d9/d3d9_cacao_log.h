#pragma once

#include <fstream>
#include <string>
#include <ctime>
#include <chrono>
#include <unordered_map>

namespace dxvk {

  /**
   * Simple append-mode logger that writes to "cacao.log" in the game's CWD.
   * Not thread-safe; only called from the D3D9 device thread.
   *
   * Production guards (the render thread does the I/O — runaway spam directly
   * costs fps; a 275 MB log was produced overnight by alternating projections
   * defeating a memcmp-only change gate):
   *  - per-tag rate limit (one line per tag per 500 ms);
   *  - hard size cap: past it logging disables itself.
   */
  class CacaoLogger {
  public:
    static CacaoLogger& get() {
      static CacaoLogger instance;
      return instance;
    }

    void open() {
      if (m_open)
        return;
      m_stream.open("cacao.log", std::ios::out | std::ios::app);
      m_open = m_stream.good();
      if (m_open)
        log("[INIT] CacaoLogger opened");
    }

    void log(const std::string& msg) {
      if (!m_open)
        return;

      // Hard size cap: self-disable past the limit.
      if (m_bytes > kMaxLogBytes) {
        if (!m_capped) {
          m_capped = true;
          std::time_t t = std::time(nullptr);
          std::tm* tm = std::localtime(&t);
          char buf[10];
          std::strftime(buf, sizeof(buf), "%H:%M:%S", tm);
          m_stream << "[" << buf << "] [INIT] log size cap reached ("
                   << (kMaxLogBytes >> 20) << " MB) - logging disabled\n";
          m_stream.flush();
        }
        return;
      }

      // Per-tag rate limit: one line per tag per 500 ms. L2 alternates
      // projections hundreds of times per second; a memcmp-only gate on
      // [SETTRANSFORM] logs on every alternation.
      std::chrono::steady_clock::time_point now =
        std::chrono::steady_clock::now();
      size_t close = msg.find(']');
      std::string tag = (close != std::string::npos)
        ? msg.substr(0, close + 1) : msg;

      auto it = m_tagLast.find(tag);
      if (it != m_tagLast.end()
       && now - it->second < std::chrono::milliseconds(kTagIntervalMs))
        return;
      m_tagLast[tag] = now;

      std::time_t t = std::time(nullptr);
      std::tm* tm = std::localtime(&t);
      char buf[10];
      std::strftime(buf, sizeof(buf), "%H:%M:%S", tm);
      m_stream << "[" << buf << "] " << msg << "\n";
      m_stream.flush();
      m_bytes += msg.size() + 12u;
    }

    void close() {
      if (m_open) {
        m_stream.close();
        m_open = false;
      }
    }

  private:
    CacaoLogger() = default;
    ~CacaoLogger() { close(); }

    static constexpr size_t   kMaxLogBytes   = 64u << 20; // 64 MB
    static constexpr uint32_t kTagIntervalMs = 500u;

    std::ofstream m_stream;
    bool          m_open  = false;
    bool          m_capped = false;
    size_t        m_bytes = 0u;
    std::unordered_map<std::string, std::chrono::steady_clock::time_point>
                  m_tagLast;
  };

}
