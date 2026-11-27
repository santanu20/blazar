//! Log-sink liveness watch.
//!
//! Issue 012 (silent log freeze, 2026-11-22): for about ten minutes three
//! sinks stayed dark while ~248 requests completed, and nothing in the
//! daemon could say afterwards whether events were never emitted or were
//! written and lost. This module keeps the arithmetic that answers that
//! question on every scrape: it compares two [`SinkStats`] snapshots one
//! window apart and names a sink as stalled when the process kept
//! producing enabled events while that sink wrote nothing.
//!
//! The arithmetic is pure so it can be pinned by unit tests; the detector
//! task in `serve()` feeds it and turns a verdict into a metric and a
//! marker line that bypasses tracing (a tracing event about a dead tracing
//! sink would itself be lost).

use std::collections::HashMap;

use blazar_core::telemetry::SinkStats;

/// Consecutive dark windows before a sink counts as stalled. Two windows
/// tolerate one coalesced write; a longer streak is a real outage.
pub const DARK_WINDOWS_BEFORE_STALL: u32 = 2;

/// Window interval of the detector task that drives [`LogSinkWatch`].
pub const WATCH_INTERVAL: std::time::Duration = std::time::Duration::from_secs(20);

pub const STDERR_SINK: &str = "stderr";
pub const FILE_SINK: &str = "file";

/// Rolling state of the per-sink liveness detector.
#[derive(Debug, Default)]
pub struct LogSinkWatch {
    prev: Option<SinkStats>,
    dark: HashMap<&'static str, u32>,
    stalls: HashMap<&'static str, u64>,
    reported: HashMap<&'static str, bool>,
}

impl LogSinkWatch {
    /// Advance the detector by one window.
    ///
    /// The first call only records the baseline. A sink is dark in a window
    /// when at least one event was enabled and the sink wrote zero lines;
    /// a window without events leaves the current streak untouched (an idle
    /// daemon is not a stalled one). After `DARK_WINDOWS_BEFORE_STALL`
    /// consecutive dark windows the sink's stall counter increments and the
    /// sink is returned exactly once per incident; a later successful write
    /// clears the incident.
    pub fn tick(&mut self, now: &SinkStats, file_armed: bool) -> Vec<&'static str> {
        let Some(prev) = self.prev else {
            self.prev = Some(*now);
            return Vec::new();
        };
        let events = now.events.saturating_sub(prev.events);
        let mut stalled = Vec::new();
        let mut observe = |sink: &'static str, writes: u64| {
            let streak = self.dark.entry(sink).or_insert(0);
            if events > 0 && writes == 0 {
                *streak = streak.saturating_add(1);
                let already = *self.reported.get(sink).unwrap_or(&false);
                if *streak == DARK_WINDOWS_BEFORE_STALL && !already {
                    *self.stalls.entry(sink).or_insert(0) += 1;
                    self.reported.insert(sink, true);
                    stalled.push(sink);
                }
            } else if writes > 0 {
                *streak = 0;
                self.reported.insert(sink, false);
            }
        };
        observe(
            STDERR_SINK,
            now.stderr_writes.saturating_sub(prev.stderr_writes),
        );
        if file_armed {
            observe(FILE_SINK, now.file_writes.saturating_sub(prev.file_writes));
        }
        self.prev = Some(*now);
        stalled
    }

    /// Total stalls observed per sink since process start.
    #[must_use]
    pub fn stall_counts(&self) -> Vec<(&'static str, u64)> {
        let mut counts: Vec<(&'static str, u64)> =
            self.stalls.iter().map(|(k, v)| (*k, *v)).collect();
        counts.sort_by_key(|(k, _)| *k);
        counts
    }
}

/// One detector step: re-check that the durable log still names the file
/// being written, then advance the window arithmetic with the live
/// counters. Returns the sinks that just started stalling. The whole
/// wiring sits here, not in the task, so a test can drive it without the
/// 20 s interval.
pub fn advance(watch: &mut LogSinkWatch) -> Vec<&'static str> {
    blazar_core::telemetry::reopen_file_if_replaced();
    let now = blazar_core::telemetry::sink_stats();
    watch.tick(&now, blazar_core::telemetry::file_sink_armed())
}

/// `/metrics` text for the stall counters. Both sinks are always listed so
/// scrape output stays stable across restarts.
#[must_use]
pub fn stall_metrics(stalls: &LogSinkWatch) -> String {
    let counts = stalls.stall_counts();
    let mut out = String::new();
    for sink in [STDERR_SINK, FILE_SINK] {
        let count = counts
            .iter()
            .find(|(name, _)| *name == sink)
            .map_or(0, |(_, count)| *count);
        out.push_str("blazar_log_sink_stall_total{sink=\"");
        out.push_str(sink);
        out.push_str("\"} ");
        out.push_str(&count.to_string());
        out.push('\n');
    }
    out
}

#[cfg(test)]
mod tests {
    #![allow(non_snake_case)]

    use super::*;

    fn stats(events: u64, stderr_writes: u64, file_writes: u64) -> SinkStats {
        SinkStats {
            events,
            stderr_writes,
            stderr_failures: 0,
            file_writes,
            file_failures: 0,
            file_reopened: 0,
        }
    }

    #[test]
    fn unit__log_sink_watch__first_tick_is_a_baseline_not_a_verdict() {
        let mut watch = LogSinkWatch::default();
        assert!(watch.tick(&stats(5, 0, 0), true).is_empty());
    }

    #[test]
    fn unit__log_sink_watch__stalls_after_two_dark_windows_once_per_incident() {
        let mut watch = LogSinkWatch::default();
        watch.tick(&stats(0, 0, 0), true);
        // Window 1: events flowed, stderr wrote nothing.
        assert!(watch.tick(&stats(4, 0, 4), true).is_empty());
        // Window 2: still dark - the verdict lands exactly here.
        assert_eq!(watch.tick(&stats(9, 0, 9), true), vec![STDERR_SINK]);
        // Window 3: the outage continues but the incident is already reported.
        assert!(watch.tick(&stats(14, 0, 14), true).is_empty());
        assert_eq!(watch.stall_counts(), vec![(STDERR_SINK, 1)]);
    }

    #[test]
    fn unit__log_sink_watch__a_write_recovers_the_sink_and_rearms_the_report() {
        let mut watch = LogSinkWatch::default();
        watch.tick(&stats(0, 0, 0), true);
        watch.tick(&stats(4, 0, 4), true);
        // A single dark window followed by traffic on the sink: no stall.
        assert!(watch.tick(&stats(9, 3, 9), true).is_empty());
        assert!(watch.tick(&stats(12, 0, 12), true).is_empty());
        assert_eq!(watch.tick(&stats(16, 0, 16), true), vec![STDERR_SINK]);
        // Recovery, then a fresh incident counts again.
        watch.tick(&stats(20, 2, 20), true);
        watch.tick(&stats(24, 0, 24), true);
        assert_eq!(watch.tick(&stats(28, 0, 28), true), vec![STDERR_SINK]);
        assert_eq!(watch.stall_counts(), vec![(STDERR_SINK, 2)]);
    }

    #[test]
    fn unit__log_sink_watch__idle_window_is_not_dark() {
        let mut watch = LogSinkWatch::default();
        watch.tick(&stats(0, 0, 0), true);
        assert!(watch.tick(&stats(0, 0, 0), true).is_empty());
        assert!(watch.tick(&stats(0, 0, 0), true).is_empty());
        assert!(watch.tick(&stats(0, 0, 0), true).is_empty());
        assert!(watch.stall_counts().is_empty());
    }

    #[test]
    fn unit__log_sink_watch__unarmed_file_sink_is_ignored() {
        let mut watch = LogSinkWatch::default();
        watch.tick(&stats(0, 0, 0), false);
        assert!(watch.tick(&stats(4, 4, 0), false).is_empty());
        assert!(watch.tick(&stats(9, 9, 0), false).is_empty());
        assert_eq!(watch.stall_counts(), Vec::<(&'static str, u64)>::new());
    }

    #[test]
    fn unit__log_sink_watch__file_sink_stalls_independently_of_stderr() {
        let mut watch = LogSinkWatch::default();
        watch.tick(&stats(0, 0, 0), true);
        assert!(watch.tick(&stats(4, 4, 0), true).is_empty());
        assert_eq!(watch.tick(&stats(9, 9, 0), true), vec![FILE_SINK]);
        assert_eq!(watch.stall_counts(), vec![(FILE_SINK, 1)]);
    }

    #[test]
    fn unit__stall_metrics__lists_both_sinks_with_zeroes() {
        let mut watch = LogSinkWatch::default();
        watch.tick(&stats(0, 0, 0), true);
        watch.tick(&stats(4, 0, 4), true);
        watch.tick(&stats(9, 0, 9), true);
        let text = stall_metrics(&watch);
        assert!(text.contains("blazar_log_sink_stall_total{sink=\"stderr\"} 1\n"));
        assert!(text.contains("blazar_log_sink_stall_total{sink=\"file\"} 0\n"));
    }
}
