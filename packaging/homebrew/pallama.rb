# Homebrew formula template for pallama.
#
# Publishing: a tap repo (e.g. <you>/homebrew-pallama) carries this file.
# Fill VERSION + per-arch SHA256 per release (the workflow publishes
# pallama-{tag}-{triple}.tar.gz with the exact shapes below); the
# install.sh / engine-update digest chain is the same asset set.
class Pallama < Formula
  desc "Multi-engine local inference server: OpenAI, ollama and Anthropic APIs"
  homepage "https://github.com/OWNER/pallama"
  version "VERSION"

  on_macos do
    if Hardware::CPU.intel?
      url "https://github.com/OWNER/pallama/releases/download/VERSION/pallama-VERSION-x86_64-apple-darwin.tar.gz"
      sha256 "FILL_PER_RELEASE"
    end
    if Hardware::CPU.arm?
      url "https://github.com/OWNER/pallama/releases/download/VERSION/pallama-VERSION-aarch64-apple-darwin.tar.gz"
      sha256 "FILL_PER_RELEASE"
    end
  end

  on_linux do
    if Hardware::CPU.intel?
      url "https://github.com/OWNER/pallama/releases/download/VERSION/pallama-VERSION-x86_64-unknown-linux-gnu.tar.gz"
      sha256 "FILL_PER_RELEASE"
    end
    if Hardware::CPU.arm? && Hardware::CPU.is_64_bit?
      url "https://github.com/OWNER/pallama/releases/download/VERSION/pallama-VERSION-aarch64-unknown-linux-gnu.tar.gz"
      sha256 "FILL_PER_RELEASE"
    end
  end

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
