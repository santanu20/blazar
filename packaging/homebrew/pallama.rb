# Homebrew formula template for pallama.
#
# Publishing: a tap repo (e.g. <you>/homebrew-pallama) carries this file.
# Fill VERSION + SHA256 per release (the workflow publishes
# pallama-{tag}-{triple}.tar.gz with the exact shape below); the
# install.sh / engine-update digest chain is the same asset set.
class Pallama < Formula
  desc "Local-first llama.cpp orchestrator: OpenAI + ollama APIs, zero-fork engine management"
  homepage "https://github.com/OWNER/pallama"
  url "https://github.com/OWNER/pallama/releases/download/VERSION/pallama-VERSION-x86_64-unknown-linux-gnu.tar.gz"
  # sha256 "FILL_PER_RELEASE"
  version "VERSION"

  def install
    bin.install "pallama"
  end

  def post_install
    ohai "Run `pallama doctor` to verify config, engine, and hardware."
  end

  service do
    run [opt_bin/"pallama", "serve"]
    keep_alive true
    log_path var/"log/pallama.log"
    error_log_path var/"log/pallama.err.log"
  end

  test do
    assert_match "pallama", shell_output("#{bin}/pallama --version")
  end
end
