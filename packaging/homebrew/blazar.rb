# Homebrew formula template for blazar.
#
# Publishing: a tap repo (e.g. <you>/homebrew-blazar) carries this file.
# Fill VERSION + per-arch SHA256 per release (the workflow publishes
# blazar-{tag}-{triple}.tar.gz with the exact shapes below); the
# install.sh / engine-update digest chain is the same asset set.
class Blazar < Formula
  desc "Multi-engine local inference server: OpenAI, ollama and Anthropic APIs"
  homepage "https://github.com/santanu20/blazar"
  version "VERSION"

  on_macos do
    if Hardware::CPU.intel?
      url "https://github.com/santanu20/blazar/releases/download/VERSION/blazar-VERSION-x86_64-apple-darwin.tar.gz"
      sha256 "FILL_PER_RELEASE"
    end
    if Hardware::CPU.arm?
      url "https://github.com/santanu20/blazar/releases/download/VERSION/blazar-VERSION-aarch64-apple-darwin.tar.gz"
      sha256 "FILL_PER_RELEASE"
    end
  end

  on_linux do
    if Hardware::CPU.intel?
      url "https://github.com/santanu20/blazar/releases/download/VERSION/blazar-VERSION-x86_64-unknown-linux-gnu.tar.gz"
      sha256 "FILL_PER_RELEASE"
    end
    if Hardware::CPU.arm? && Hardware::CPU.is_64_bit?
      url "https://github.com/santanu20/blazar/releases/download/VERSION/blazar-VERSION-aarch64-unknown-linux-gnu.tar.gz"
      sha256 "FILL_PER_RELEASE"
    end
  end

  def install
    bin.install "blazar"
  end

  def post_install
    ohai "Run `blazar doctor` to verify config, engine, and hardware."
  end

  service do
    run [opt_bin/"blazar", "serve"]
    keep_alive true
    log_path var/"log/blazar.log"
    error_log_path var/"log/blazar.err.log"
  end

  test do
    assert_match "blazar", shell_output("#{bin}/blazar --version")
  end
end
