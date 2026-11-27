#![cfg(target_os = "linux")]

//! End-to-end pin for the log-sink stall detector. It installs the real
//! tracing subscriber with its durable-log arm on a full device, so the
//! detector sees the counters the daemon sees: stderr healthy, the file
//! arm failing on every write. Each test binary is its own process, so
//! installing the global subscriber here affects nothing else.

use blazar_gateway::log_health::{self, FILE_SINK, LogSinkWatch};

fn emit(line: &str) {
    tracing::info!(target: "blazar::log_health_probe", "{line}");
}

#[test]
#[allow(non_snake_case)]
fn integration__detector__names_a_file_sink_that_is_dark_while_events_flow() {
    let dir = tempfile::tempdir().expect("temp dir");
    let log = dir.path().join("daemon.log");
    std::os::unix::fs::symlink("/dev/full", &log).expect("symlink onto /dev/full");
    blazar_core::telemetry::init_tracing(0, Some("blazar=info"), Some(&log));
    assert!(
        blazar_core::telemetry::file_sink_armed(),
        "file arm not armed"
    );

    let mut watch = LogSinkWatch::default();
    emit("baseline");
    assert!(log_health::advance(&mut watch).is_empty());
    let before = blazar_core::telemetry::sink_stats();

    emit("window one");
    assert!(
        log_health::advance(&mut watch).is_empty(),
        "one dark window is not a stall yet"
    );
    emit("window two");
    assert_eq!(log_health::advance(&mut watch), vec![FILE_SINK]);

    let after = blazar_core::telemetry::sink_stats();
    assert!(after.events > before.events, "events were not counted");
    assert_eq!(
        after.file_writes, before.file_writes,
        "a full device must reject every write"
    );
    assert!(after.file_failures > before.file_failures);
    assert!(
        after.stderr_writes > before.stderr_writes,
        "stderr must stay healthy"
    );
    assert_eq!(
        after.file_reopened, 0,
        "a full device is the same file, not a replaced one"
    );
    let metrics = log_health::stall_metrics(&watch);
    assert!(metrics.contains("blazar_log_sink_stall_total{sink=\"file\"} 1\n"));
    assert!(metrics.contains("blazar_log_sink_stall_total{sink=\"stderr\"} 0\n"));
}
