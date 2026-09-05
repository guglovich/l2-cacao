#pragma once

#include <fstream>
#include <string>
#include <ctime>

namespace dxvk {

  /**
   * Simple append-mode logger that writes to "cacao.log" in the game's CWD.
   * Not thread-safe; only called from the D3D9 device thread.
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
      // HH:MM:SS timestamp
      std::time_t t = std::time(nullptr);
      std::tm* tm = std::localtime(&t);
      char buf[10];
      std::strftime(buf, sizeof(buf), "%H:%M:%S", tm);
      m_stream << "[" << buf << "] " << msg << "\n";
      m_stream.flush();
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

    std::ofstream m_stream;
    bool          m_open = false;
  };

}
