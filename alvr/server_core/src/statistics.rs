use crate::FILESYSTEM_LAYOUT;
use alvr_common::{DeviceMotion, SlidingWindowAverage, HEAD_ID};
use alvr_events::{BitrateDirectives, EventType, GraphStatistics, StatisticsSummary};
use alvr_packets::{eye_pose_status_pico, ClientStatistics, EyeTrackingDataPICO};
use alvr_session::{settings_schema::Switch, Settings};
use csv;
use serde::Serialize;
use serde_json;
use std::{
    collections::{HashMap, VecDeque},
    fs::{self, File},
    net::{ToSocketAddrs, UdpSocket},
    path::PathBuf,
    sync::mpsc::{sync_channel, SyncSender},
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

const FULL_REPORT_INTERVAL: Duration = Duration::from_millis(500);
const EPS_INTERVAL: Duration = Duration::from_micros(1);
const DATA_COLLECTION_QUEUE_LEN: usize = 32;

const DATASET_TARGET_HZ: f32 = 90.0;

const DEFAULT_TELEMETRY_ENDPOINT: &str = "127.0.0.1:49152";

// This struct is used for queuing and is not flat
#[derive(Clone)]
struct TelemetryPayload {
    frame_timestamp_ns: u64,
    system_unix_time_ms: u64,
    head_motion: Option<DeviceMotion>,
    pico_eye_tracking_data: Option<EyeTrackingDataPICO>,
    // MTP decomposition: 8 layers
    game_time_latency: Duration,
    server_compositor_latency: Duration,
    encoder_latency: Duration,
    network_latency: Duration,
    decoder_latency: Duration,
    decoder_queue_latency: Duration,
    client_compositor_latency: Duration,
    vsync_queue_latency: Duration,
    // Aggregated metrics
    total_pipeline_latency: Duration,
    throughput_bps: f32,
    packets_lost_per_sec: f32,
    client_fps: f32,
    server_fps: f32,
    // Frame type flag for DRL model (I-frame vs P-frame size distribution)
    is_idr: bool,
    // Raw frame size in bytes for DRL model (critical for I/P frame size distribution analysis)
    video_packet_bytes: usize,
}

// This struct is flat and is used for CSV serialization
#[derive(Serialize)]
struct TelemetryCsvRecord {
    frame_timestamp_ns: u64,
    system_unix_time_ms: u64,
    head_position_x: f32,
    head_position_y: f32,
    head_position_z: f32,
    head_orientation_x: f32,
    head_orientation_y: f32,
    head_orientation_z: f32,
    head_orientation_w: f32,
    left_gaze_vector_valid: bool,
    right_gaze_vector_valid: bool,
    left_eye_openness_valid: bool,
    right_eye_openness_valid: bool,
    left_pupil_dilation_valid: bool,
    right_pupil_dilation_valid: bool,
    left_gaze_vector_x: Option<f32>,
    left_gaze_vector_y: Option<f32>,
    left_gaze_vector_z: Option<f32>,
    right_gaze_vector_x: Option<f32>,
    right_gaze_vector_y: Option<f32>,
    right_gaze_vector_z: Option<f32>,
    left_eye_openness: Option<f32>,
    right_eye_openness: Option<f32>,
    left_pupil_dilation: Option<f32>,
    right_pupil_dilation: Option<f32>,
    // MTP decomposition: 8 layers (in milliseconds)
    game_time_ms: f32,
    server_compositor_ms: f32,
    encode_latency_ms: f32,
    network_latency_ms: f32,
    decode_latency_ms: f32,
    decoder_queue_ms: f32,
    client_compositor_ms: f32,
    vsync_queue_ms: f32,
    // Aggregated metrics
    total_pipeline_latency_ms: f32,
    throughput_mbps: f32,
    packets_lost_per_sec: f32,
    client_fps: f32,
    server_fps: f32,
    // Frame type flag for DRL model (I-frame vs P-frame size distribution)
    is_idr: bool,
    // Raw frame size in bytes for DRL model (critical for I/P frame size distribution analysis)
    video_packet_bytes: usize,
}

// ============================================================================
// State Log (90Hz tick sampling) - Separate from event log
// ============================================================================

/// Target sampling rate for state log (90Hz)
pub const STATE_LOG_TARGET_HZ: f32 = 90.0;

/// Internal structure for state log queue (not flat, includes metadata)
/// Public for use by 90Hz tick thread in connection.rs
#[derive(Clone)]
pub struct StateLogPayload {
    // Time indices (aligned with event log)
    pub system_unix_time_ms: u64,
    pub monotonic_time_ns: u64,
    // Head pose (sample-and-hold)
    pub head_position: [f32; 3],
    pub head_orientation: [f32; 4],
    pub head_age_ms: f32,
    pub head_valid: bool,
    // Eye tracking (sample-and-hold)
    pub left_gaze_vector: [f32; 3],
    pub right_gaze_vector: [f32; 3],
    pub left_eye_openness: f32,
    pub right_eye_openness: f32,
    pub left_pupil_dilation: f32,
    pub right_pupil_dilation: f32,
    pub left_gaze_vector_valid: bool,
    pub right_gaze_vector_valid: bool,
    pub left_eye_openness_valid: bool,
    pub right_eye_openness_valid: bool,
    pub left_pupil_dilation_valid: bool,
    pub right_pupil_dilation_valid: bool,
    pub eye_age_ms: f32,
    pub eye_source_time: u64, // Original EyeTrackingDataPICO.time
    // Server state
    pub server_fps: f32,
    pub server_fps_age_ms: f32,
    // Rendered pose (cached from last frame submission for ATW delta)
    pub rendered_head_position: [f32; 3],
    pub rendered_head_orientation: [f32; 4],
    pub rendered_head_valid: bool,
    // Pose delta (prediction error: current vs rendered)
    pub pose_delta_position: f32,
    pub pose_delta_orientation_deg: f32,
}

/// Flat structure for CSV serialization
#[derive(Serialize)]
struct StateLogCsvRecord {
    // Time indices
    system_unix_time_ms: u64,
    monotonic_time_ns: u64,
    // Head pose
    head_position_x: f32,
    head_position_y: f32,
    head_position_z: f32,
    head_orientation_x: f32,
    head_orientation_y: f32,
    head_orientation_z: f32,
    head_orientation_w: f32,
    head_age_ms: f32,
    head_valid: bool,
    // Eye tracking
    left_gaze_vector_x: f32,
    left_gaze_vector_y: f32,
    left_gaze_vector_z: f32,
    right_gaze_vector_x: f32,
    right_gaze_vector_y: f32,
    right_gaze_vector_z: f32,
    left_eye_openness: f32,
    right_eye_openness: f32,
    left_pupil_dilation: f32,
    right_pupil_dilation: f32,
    left_gaze_vector_valid: bool,
    right_gaze_vector_valid: bool,
    left_eye_openness_valid: bool,
    right_eye_openness_valid: bool,
    left_pupil_dilation_valid: bool,
    right_pupil_dilation_valid: bool,
    eye_age_ms: f32,
    eye_source_time: u64,
    // Server state
    server_fps: f32,
    server_fps_age_ms: f32,
    // Rendered pose
    rendered_head_position_x: f32,
    rendered_head_position_y: f32,
    rendered_head_position_z: f32,
    rendered_head_orientation_x: f32,
    rendered_head_orientation_y: f32,
    rendered_head_orientation_z: f32,
    rendered_head_orientation_w: f32,
    rendered_head_valid: bool,
    // Pose delta
    pose_delta_position: f32,
    pose_delta_orientation_deg: f32,
}

impl From<StateLogPayload> for StateLogCsvRecord {
    fn from(p: StateLogPayload) -> Self {
        Self {
            system_unix_time_ms: p.system_unix_time_ms,
            monotonic_time_ns: p.monotonic_time_ns,
            head_position_x: p.head_position[0],
            head_position_y: p.head_position[1],
            head_position_z: p.head_position[2],
            head_orientation_x: p.head_orientation[0],
            head_orientation_y: p.head_orientation[1],
            head_orientation_z: p.head_orientation[2],
            head_orientation_w: p.head_orientation[3],
            head_age_ms: p.head_age_ms,
            head_valid: p.head_valid,
            left_gaze_vector_x: p.left_gaze_vector[0],
            left_gaze_vector_y: p.left_gaze_vector[1],
            left_gaze_vector_z: p.left_gaze_vector[2],
            right_gaze_vector_x: p.right_gaze_vector[0],
            right_gaze_vector_y: p.right_gaze_vector[1],
            right_gaze_vector_z: p.right_gaze_vector[2],
            left_eye_openness: p.left_eye_openness,
            right_eye_openness: p.right_eye_openness,
            left_pupil_dilation: p.left_pupil_dilation,
            right_pupil_dilation: p.right_pupil_dilation,
            left_gaze_vector_valid: p.left_gaze_vector_valid,
            right_gaze_vector_valid: p.right_gaze_vector_valid,
            left_eye_openness_valid: p.left_eye_openness_valid,
            right_eye_openness_valid: p.right_eye_openness_valid,
            left_pupil_dilation_valid: p.left_pupil_dilation_valid,
            right_pupil_dilation_valid: p.right_pupil_dilation_valid,
            eye_age_ms: p.eye_age_ms,
            eye_source_time: p.eye_source_time,
            server_fps: p.server_fps,
            server_fps_age_ms: p.server_fps_age_ms,
            rendered_head_position_x: p.rendered_head_position[0],
            rendered_head_position_y: p.rendered_head_position[1],
            rendered_head_position_z: p.rendered_head_position[2],
            rendered_head_orientation_x: p.rendered_head_orientation[0],
            rendered_head_orientation_y: p.rendered_head_orientation[1],
            rendered_head_orientation_z: p.rendered_head_orientation[2],
            rendered_head_orientation_w: p.rendered_head_orientation[3],
            rendered_head_valid: p.rendered_head_valid,
            pose_delta_position: p.pose_delta_position,
            pose_delta_orientation_deg: p.pose_delta_orientation_deg,
        }
    }
}

// ============================================================================
// Event Log (existing telemetry) - Conversion implementations
// ============================================================================

impl From<TelemetryPayload> for TelemetryCsvRecord {
    fn from(payload: TelemetryPayload) -> Self {
        let (head_pos, head_ori) = if let Some(motion) = payload.head_motion {
            (
                motion.pose.position.to_array(),
                motion.pose.orientation.to_array(),
            )
        } else {
            ([0.0; 3], [0.0, 0.0, 0.0, 1.0])
        };

        let pico_eye = payload.pico_eye_tracking_data;

        let left_gaze_vector_valid = pico_eye
            .as_ref()
            .is_some_and(|d| eye_pose_status_pico::is_gaze_vector_valid(d.left_eye_pose_status));
        let right_gaze_vector_valid = pico_eye
            .as_ref()
            .is_some_and(|d| eye_pose_status_pico::is_gaze_vector_valid(d.right_eye_pose_status));
        let left_eye_openness_valid = pico_eye
            .as_ref()
            .is_some_and(|d| eye_pose_status_pico::is_eye_openness_valid(d.left_eye_pose_status));
        let right_eye_openness_valid = pico_eye
            .as_ref()
            .is_some_and(|d| eye_pose_status_pico::is_eye_openness_valid(d.right_eye_pose_status));
        let left_pupil_dilation_valid = pico_eye
            .as_ref()
            .is_some_and(|d| eye_pose_status_pico::is_pupil_dilation_valid(d.left_eye_pose_status));
        let right_pupil_dilation_valid = pico_eye.as_ref().is_some_and(|d| {
            eye_pose_status_pico::is_pupil_dilation_valid(d.right_eye_pose_status)
        });

        Self {
            frame_timestamp_ns: payload.frame_timestamp_ns,
            system_unix_time_ms: payload.system_unix_time_ms,
            head_position_x: head_pos[0],
            head_position_y: head_pos[1],
            head_position_z: head_pos[2],
            head_orientation_x: head_ori[0],
            head_orientation_y: head_ori[1],
            head_orientation_z: head_ori[2],
            head_orientation_w: head_ori[3],
            left_gaze_vector_valid,
            right_gaze_vector_valid,
            left_eye_openness_valid,
            right_eye_openness_valid,
            left_pupil_dilation_valid,
            right_pupil_dilation_valid,
            left_gaze_vector_x: pico_eye.as_ref().map(|d| d.left_eye_gaze_vector[0]),
            left_gaze_vector_y: pico_eye.as_ref().map(|d| d.left_eye_gaze_vector[1]),
            left_gaze_vector_z: pico_eye.as_ref().map(|d| d.left_eye_gaze_vector[2]),
            right_gaze_vector_x: pico_eye.as_ref().map(|d| d.right_eye_gaze_vector[0]),
            right_gaze_vector_y: pico_eye.as_ref().map(|d| d.right_eye_gaze_vector[1]),
            right_gaze_vector_z: pico_eye.as_ref().map(|d| d.right_eye_gaze_vector[2]),
            left_eye_openness: pico_eye.as_ref().map(|d| d.left_eye_openness),
            right_eye_openness: pico_eye.as_ref().map(|d| d.right_eye_openness),
            left_pupil_dilation: pico_eye.as_ref().map(|d| d.left_eye_pupil_dilation),
            right_pupil_dilation: pico_eye.as_ref().map(|d| d.right_eye_pupil_dilation),
            // MTP decomposition: 8 layers
            game_time_ms: payload.game_time_latency.as_secs_f32() * 1000.0,
            server_compositor_ms: payload.server_compositor_latency.as_secs_f32() * 1000.0,
            encode_latency_ms: payload.encoder_latency.as_secs_f32() * 1000.0,
            network_latency_ms: payload.network_latency.as_secs_f32() * 1000.0,
            decode_latency_ms: payload.decoder_latency.as_secs_f32() * 1000.0,
            decoder_queue_ms: payload.decoder_queue_latency.as_secs_f32() * 1000.0,
            client_compositor_ms: payload.client_compositor_latency.as_secs_f32() * 1000.0,
            vsync_queue_ms: payload.vsync_queue_latency.as_secs_f32() * 1000.0,
            // Aggregated metrics
            total_pipeline_latency_ms: payload.total_pipeline_latency.as_secs_f32() * 1000.0,
            throughput_mbps: payload.throughput_bps / 1_000_000.0,
            packets_lost_per_sec: payload.packets_lost_per_sec,
            client_fps: payload.client_fps,
            server_fps: payload.server_fps,
            is_idr: payload.is_idr,
            video_packet_bytes: payload.video_packet_bytes,
        }
    }
}

struct DataCollectorWorker {
    sender: SyncSender<TelemetryPayload>,
}

impl DataCollectorWorker {
    fn new(file_path: PathBuf) -> Self {
        let (sender, receiver) = sync_channel(DATA_COLLECTION_QUEUE_LEN);

        thread::spawn(move || {
            let file = match File::create(&file_path) {
                Ok(file) => file,
                Err(e) => {
                    alvr_common::error!("Failed to create DRL data log file: {e}");
                    return;
                }
            };

            let mut writer = csv::WriterBuilder::new()
                .has_headers(false)
                .from_writer(file);

            // Manually write header from TelemetryCsvRecord field names
            if writer
                .write_record(&[
                    "frame_timestamp_ns",
                    "system_unix_time_ms",
                    "head_position_x",
                    "head_position_y",
                    "head_position_z",
                    "head_orientation_x",
                    "head_orientation_y",
                    "head_orientation_z",
                    "head_orientation_w",
                    "left_gaze_vector_valid",
                    "right_gaze_vector_valid",
                    "left_eye_openness_valid",
                    "right_eye_openness_valid",
                    "left_pupil_dilation_valid",
                    "right_pupil_dilation_valid",
                    "left_gaze_vector_x",
                    "left_gaze_vector_y",
                    "left_gaze_vector_z",
                    "right_gaze_vector_x",
                    "right_gaze_vector_y",
                    "right_gaze_vector_z",
                    "left_eye_openness",
                    "right_eye_openness",
                    "left_pupil_dilation",
                    "right_pupil_dilation",
                    "game_time_ms",
                    "server_compositor_ms",
                    "encode_latency_ms",
                    "network_latency_ms",
                    "decode_latency_ms",
                    "decoder_queue_ms",
                    "client_compositor_ms",
                    "vsync_queue_ms",
                    "total_pipeline_latency_ms",
                    "throughput_mbps",
                    "packets_lost_per_sec",
                    "client_fps",
                    "server_fps",
                    "is_idr",
                    "video_packet_bytes",
                ])
                .is_err()
            {
                alvr_common::error!("Failed to write DRL data log header");
                return;
            }

            for payload in receiver {
                let record = TelemetryCsvRecord::from(payload);
                if writer.serialize(record).is_err() {
                    // Don't log error here to avoid spam. The channel will just fill up and drop data.
                }
            }
        });

        Self { sender }
    }

    fn try_send(&self, payload: TelemetryPayload) {
        let _ = self.sender.try_send(payload);
    }
}

// ============================================================================
// State Log Worker (90Hz tick sampling)
// ============================================================================

struct StateLogWorker {
    sender: SyncSender<StateLogPayload>,
}

impl StateLogWorker {
    fn new(file_path: PathBuf) -> Self {
        let (sender, receiver) = sync_channel(DATA_COLLECTION_QUEUE_LEN);

        thread::spawn(move || {
            let file = match File::create(&file_path) {
                Ok(file) => file,
                Err(e) => {
                    alvr_common::error!("Failed to create state log file: {e}");
                    return;
                }
            };

            let mut writer = csv::WriterBuilder::new()
                .has_headers(false)
                .from_writer(file);

            // Write header for StateLogCsvRecord
            if writer
                .write_record(&[
                    // Time indices
                    "system_unix_time_ms",
                    "monotonic_time_ns",
                    // Head pose
                    "head_position_x",
                    "head_position_y",
                    "head_position_z",
                    "head_orientation_x",
                    "head_orientation_y",
                    "head_orientation_z",
                    "head_orientation_w",
                    "head_age_ms",
                    "head_valid",
                    // Eye tracking
                    "left_gaze_vector_x",
                    "left_gaze_vector_y",
                    "left_gaze_vector_z",
                    "right_gaze_vector_x",
                    "right_gaze_vector_y",
                    "right_gaze_vector_z",
                    "left_eye_openness",
                    "right_eye_openness",
                    "left_pupil_dilation",
                    "right_pupil_dilation",
                    "left_gaze_vector_valid",
                    "right_gaze_vector_valid",
                    "left_eye_openness_valid",
                    "right_eye_openness_valid",
                    "left_pupil_dilation_valid",
                    "right_pupil_dilation_valid",
                    "eye_age_ms",
                    "eye_source_time",
                    // Server state
                    "server_fps",
                    "server_fps_age_ms",
                    // Rendered pose (cached from last frame submission)
                    "rendered_head_position_x",
                    "rendered_head_position_y",
                    "rendered_head_position_z",
                    "rendered_head_orientation_x",
                    "rendered_head_orientation_y",
                    "rendered_head_orientation_z",
                    "rendered_head_orientation_w",
                    "rendered_head_valid",
                    // Pose delta (prediction error)
                    "pose_delta_position",
                    "pose_delta_orientation_deg",
                ])
                .is_err()
            {
                alvr_common::error!("Failed to write state log header");
                return;
            }

            for payload in receiver {
                let record = StateLogCsvRecord::from(payload);
                if writer.serialize(record).is_err() {
                    // Don't log error to avoid spam
                }
            }
        });

        Self { sender }
    }

    fn try_send(&self, payload: StateLogPayload) {
        let _ = self.sender.try_send(payload);
    }
}

struct TelemetryUploadWorker {
    sender: SyncSender<TelemetryPayload>,
}

impl TelemetryUploadWorker {
    fn new(endpoint: &str) -> Option<Self> {
        let resolved_addr = match endpoint
            .to_socket_addrs()
            .ok()
            .and_then(|mut addrs| addrs.next())
        {
            Some(addr) => addr,
            None => {
                alvr_common::warn!(
                    "Invalid telemetry endpoint: {endpoint}. Telemetry upload disabled."
                );
                return None;
            }
        };

        let (sender, receiver) = sync_channel(DATA_COLLECTION_QUEUE_LEN);

        thread::spawn(move || {
            let socket = match UdpSocket::bind("0.0.0.0:0") {
                Ok(socket) => socket,
                Err(err) => {
                    alvr_common::warn!(
                        "Failed to bind telemetry socket: {err}. Telemetry upload disabled."
                    );
                    return;
                }
            };

            if socket.connect(resolved_addr).is_err() {
                alvr_common::warn!("Failed to connect telemetry socket to {resolved_addr}. Telemetry upload disabled.");
                return;
            }

            for payload in receiver {
                let record = TelemetryCsvRecord::from(payload);
                match serde_json::to_vec(&record) {
                    Ok(bytes) => {
                        let _ = socket.send(&bytes);
                    }
                    Err(err) => {
                        alvr_common::warn!("Failed to serialize telemetry payload: {err:?}")
                    }
                }
            }
        });

        Some(Self { sender })
    }

    fn try_send(&self, payload: TelemetryPayload) {
        let _ = self.sender.try_send(payload);
    }
}

#[derive(Clone)]
pub struct DatasetSessionMetadata {
    pub session_dir: PathBuf,
    pub session_start_ms: u64,
    pub csv_path: PathBuf,               // Event log (telemetry_event.csv)
    pub state_log_csv_path: PathBuf,     // State log (telemetry_state_90hz.csv)
    pub video_manifest_path: PathBuf,
    pub video_segment_duration: Duration,
}

pub struct HistoryFrame {
    target_timestamp: Duration,
    tracking_received: Instant,
    frame_present: Instant,
    frame_composed: Instant,
    frame_encoded: Instant,
    video_packet_bytes: usize,
    total_pipeline_latency: Duration,
    is_idr: bool,
}

impl Default for HistoryFrame {
    fn default() -> Self {
        let now = Instant::now();
        Self {
            target_timestamp: Duration::ZERO,
            tracking_received: now,
            frame_present: now,
            frame_composed: now,
            frame_encoded: now,
            video_packet_bytes: 0,
            total_pipeline_latency: Duration::ZERO,
            is_idr: false,
        }
    }
}

#[derive(Default, Clone)]
struct BatteryData {
    gauge_value: f32,
    is_plugged: bool,
}

pub struct StatisticsManager {
    history_buffer: VecDeque<HistoryFrame>,
    max_history_size: usize,
    last_full_report_instant: Instant,
    partial_stats_last_reset: Instant,
    last_frame_present_instant: Instant,
    last_frame_present_interval: Duration,
    video_packets_total: usize,
    video_packets_partial_sum: usize,
    video_bytes_total: usize,
    video_bytes_partial_sum: usize,
    packets_lost_total: usize,
    packets_lost_partial_sum: usize,
    battery_gauges: HashMap<u64, BatteryData>,
    steamvr_pipeline_latency: Duration,
    motion_to_photon_latency_average: SlidingWindowAverage<Duration>,
    last_vsync_time: Instant,
    frame_interval: Duration,
    last_throughput_directives: BitrateDirectives,
    // Event log (existing telemetry)
    data_collector_worker: Option<DataCollectorWorker>,
    telemetry_uploader: Option<TelemetryUploadWorker>,
    dataset_session: Option<DatasetSessionMetadata>,
    dataset_sample_interval: Duration,
    last_dataset_sample_instant: Instant,
    // State log (90Hz tick sampling)
    state_log_worker: Option<StateLogWorker>,
}

impl StatisticsManager {
    pub fn new(
        settings: &Settings,
        nominal_server_frame_interval: Duration,
        steamvr_pipeline_frames: f32,
    ) -> Self {
        let endpoint = settings
            .extra
            .logging
            .data_collection_endpoint
            .trim()
            .to_string();
        let telemetry_endpoint = if endpoint.is_empty() {
            DEFAULT_TELEMETRY_ENDPOINT.to_string()
        } else {
            endpoint
        };

        let telemetry_enabled = settings.extra.logging.enable_data_collection;

        let dataset_segment_duration =
            if let Switch::Enabled(config) = &settings.extra.capture.rolling_video_files {
                Duration::from_secs(config.duration_s.max(1))
            } else {
                Duration::from_secs(10)
            };

        let dataset_session = if telemetry_enabled
            && settings.extra.logging.enable_dataset_collection
        {
            if let Some(layout) = FILESYSTEM_LAYOUT.get() {
                let base_dir = if settings.extra.logging.dataset_directory.trim().is_empty() {
                    layout.log_dir.join("dataset")
                } else {
                    PathBuf::from(settings.extra.logging.dataset_directory.trim())
                };
                let session_start_ms = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_else(|_| Duration::ZERO)
                    .as_millis() as u64;
                let session_dir = base_dir.join(format!("{}", session_start_ms));
                if fs::create_dir_all(&session_dir).is_err() {
                    alvr_common::warn!(
                        "Failed to create dataset directory: {}",
                        session_dir.display()
                    );
                    None
                } else {
                    let csv_path = session_dir.join("telemetry_event.csv");
                    let state_log_csv_path = session_dir.join("telemetry_state_90hz.csv");
                    let manifest_path = session_dir.join("video_manifest.csv");
                    if fs::write(&manifest_path, "file,start_ms,duration_ms\n").is_err() {
                        alvr_common::warn!(
                            "Failed to initialize video manifest at {}",
                            manifest_path.display()
                        );
                    }
                    Some(DatasetSessionMetadata {
                        session_dir,
                        session_start_ms,
                        csv_path,
                        state_log_csv_path,
                        video_manifest_path: manifest_path,
                        video_segment_duration: dataset_segment_duration,
                    })
                }
            } else {
                alvr_common::warn!("Filesystem layout unavailable; dataset collection disabled");
                None
            }
        } else {
            None
        };

        let data_collector_worker = dataset_session
            .as_ref()
            .map(|info| DataCollectorWorker::new(info.csv_path.clone()));

        // State log worker for 90Hz tick sampling
        let state_log_worker = dataset_session
            .as_ref()
            .map(|info| StateLogWorker::new(info.state_log_csv_path.clone()));

        let telemetry_uploader = if telemetry_enabled {
            TelemetryUploadWorker::new(&telemetry_endpoint)
        } else {
            None
        };

        let dataset_sample_interval = Duration::from_secs_f32(1.0 / DATASET_TARGET_HZ);

        Self {
            history_buffer: VecDeque::new(),
            max_history_size: settings.connection.statistics_history_size,
            last_full_report_instant: Instant::now(),
            partial_stats_last_reset: Instant::now(),
            last_frame_present_instant: Instant::now(),
            last_frame_present_interval: Duration::ZERO,
            video_packets_total: 0,
            video_packets_partial_sum: 0,
            video_bytes_total: 0,
            video_bytes_partial_sum: 0,
            packets_lost_total: 0,
            packets_lost_partial_sum: 0,
            battery_gauges: HashMap::new(),
            steamvr_pipeline_latency: Duration::from_secs_f32(
                steamvr_pipeline_frames * nominal_server_frame_interval.as_secs_f32(),
            ),
            motion_to_photon_latency_average: SlidingWindowAverage::new(
                Duration::ZERO,
                settings.connection.statistics_history_size,
            ),
            last_vsync_time: Instant::now(),
            frame_interval: nominal_server_frame_interval,
            last_throughput_directives: BitrateDirectives::default(),
            data_collector_worker,
            telemetry_uploader,
            dataset_session,
            dataset_sample_interval,
            last_dataset_sample_instant: Instant::now() - dataset_sample_interval,
            state_log_worker,
        }
    }

    pub fn dataset_session_metadata(&self) -> Option<DatasetSessionMetadata> {
        self.dataset_session.clone()
    }

    /// Check if state log is enabled
    pub fn is_state_log_enabled(&self) -> bool {
        self.state_log_worker.is_some()
    }

    /// Get current server FPS (computed from last frame present interval)
    pub fn get_server_fps(&self) -> f32 {
        1.0 / Duration::max(self.last_frame_present_interval, EPS_INTERVAL).as_secs_f32()
    }

    /// Get last frame present instant for age calculation
    pub fn get_last_frame_present_instant(&self) -> Instant {
        self.last_frame_present_instant
    }

    /// Submit a state log sample (called from 90Hz tick thread)
    pub fn submit_state_log(&self, payload: StateLogPayload) {
        if let Some(worker) = &self.state_log_worker {
            worker.try_send(payload);
        }
    }

    pub fn report_tracking_received(&mut self, target_timestamp: Duration) {
        if !self
            .history_buffer
            .iter()
            .any(|frame| frame.target_timestamp == target_timestamp)
        {
            self.history_buffer.push_front(HistoryFrame {
                target_timestamp,
                tracking_received: Instant::now(),
                ..Default::default()
            });
        }

        if self.history_buffer.len() > self.max_history_size {
            self.history_buffer.pop_back();
        }
    }

    pub fn report_frame_present(&mut self, target_timestamp: Duration, offset: Duration) {
        if let Some(frame) = self
            .history_buffer
            .iter_mut()
            .find(|frame| frame.target_timestamp == target_timestamp)
        {
            let now = Instant::now() - offset;

            self.last_frame_present_interval =
                now.saturating_duration_since(self.last_frame_present_instant);
            self.last_frame_present_instant = now;

            frame.frame_present = now;
        }
    }

    pub fn report_frame_composed(&mut self, target_timestamp: Duration, offset: Duration) {
        if let Some(frame) = self
            .history_buffer
            .iter_mut()
            .find(|frame| frame.target_timestamp == target_timestamp)
        {
            frame.frame_composed = Instant::now() - offset;
        }
    }

    pub fn report_frame_encoded(
        &mut self,
        target_timestamp: Duration,
        bytes_count: usize,
        is_idr: bool,
    ) -> Duration {
        self.video_packets_total += 1;
        self.video_packets_partial_sum += 1;
        self.video_bytes_total += bytes_count;
        self.video_bytes_partial_sum += bytes_count;

        if let Some(frame) = self
            .history_buffer
            .iter_mut()
            .find(|frame| frame.target_timestamp == target_timestamp)
        {
            frame.frame_encoded = Instant::now();
            frame.video_packet_bytes = bytes_count;
            frame.is_idr = is_idr;
            frame
                .frame_encoded
                .saturating_duration_since(frame.frame_composed)
        } else {
            Duration::ZERO
        }
    }

    pub fn report_packet_loss(&mut self) {
        self.packets_lost_total += 1;
        self.packets_lost_partial_sum += 1;
    }

    pub fn report_battery(&mut self, device_id: u64, gauge_value: f32, is_plugged: bool) {
        *self.battery_gauges.entry(device_id).or_default() = BatteryData {
            gauge_value,
            is_plugged,
        };
    }

    pub fn report_throughput_stats(&mut self, stats: BitrateDirectives) {
        self.last_throughput_directives = stats;
    }

    pub fn report_statistics(
        &mut self,
        client_stats: ClientStatistics,
        head_motion: Option<DeviceMotion>,
        pico_eye_tracking_data: Option<EyeTrackingDataPICO>,
    ) -> (Duration, Duration) {
        self.motion_to_photon_latency_average
            .submit_sample(client_stats.total_pipeline_latency);

        if let Some(frame) = self
            .history_buffer
            .iter_mut()
            .find(|frame| frame.target_timestamp == client_stats.target_timestamp)
        {
            frame.total_pipeline_latency = client_stats.total_pipeline_latency;

            let game_time_latency = frame
                .frame_present
                .saturating_duration_since(frame.tracking_received);
            let server_compositor_latency = frame
                .frame_composed
                .saturating_duration_since(frame.frame_present);
            let encoder_latency = frame
                .frame_encoded
                .saturating_duration_since(frame.frame_composed);
            let network_latency = frame.total_pipeline_latency.saturating_sub(
                game_time_latency
                    + server_compositor_latency
                    + encoder_latency
                    + client_stats.video_decode
                    + client_stats.video_decoder_queue
                    + client_stats.rendering
                    + client_stats.vsync_queue,
            );
            let client_fps =
                1.0 / Duration::max(client_stats.frame_interval, EPS_INTERVAL).as_secs_f32();
            let server_fps =
                1.0 / Duration::max(self.last_frame_present_interval, EPS_INTERVAL).as_secs_f32();

            let now = Instant::now();
            let stats_window_duration = Duration::max(
                now.saturating_duration_since(self.partial_stats_last_reset),
                EPS_INTERVAL,
            );
            let packets_lost_per_sec =
                self.packets_lost_partial_sum as f32 / stats_window_duration.as_secs_f32();

            let safe_network_latency = Duration::max(network_latency, EPS_INTERVAL);
            let safe_present_interval =
                Duration::max(self.last_frame_present_interval, EPS_INTERVAL);

            let should_log_dataset = if self.data_collector_worker.is_some() {
                if now >= self.last_dataset_sample_instant + self.dataset_sample_interval {
                    self.last_dataset_sample_instant = now;
                    true
                } else {
                    false
                }
            } else {
                false
            };

            let should_upload = self.telemetry_uploader.is_some();

            if should_log_dataset || should_upload {
                let payload = TelemetryPayload {
                    frame_timestamp_ns: client_stats.target_timestamp.as_nanos() as u64,
                    system_unix_time_ms: SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap()
                        .as_millis() as u64,
                    head_motion,
                    pico_eye_tracking_data,
                    // MTP decomposition: 8 layers
                    game_time_latency,
                    server_compositor_latency,
                    encoder_latency,
                    network_latency,
                    decoder_latency: client_stats.video_decode,
                    decoder_queue_latency: client_stats.video_decoder_queue,
                    client_compositor_latency: client_stats.rendering,
                    vsync_queue_latency: client_stats.vsync_queue,
                    // Aggregated metrics
                    total_pipeline_latency: client_stats.total_pipeline_latency,
                    throughput_bps: (frame.video_packet_bytes as f32 * 8.0)
                        / safe_network_latency.as_secs_f32(),
                    packets_lost_per_sec,
                    client_fps,
                    server_fps,
                    is_idr: frame.is_idr,
                    video_packet_bytes: frame.video_packet_bytes,
                };

                if should_log_dataset {
                    if let Some(worker) = &self.data_collector_worker {
                        worker.try_send(payload.clone());
                    }
                }

                if let Some(uploader) = &self.telemetry_uploader {
                    uploader.try_send(payload);
                }
            }

            if now > self.last_full_report_instant + FULL_REPORT_INTERVAL {
                self.last_full_report_instant = now;
                let interval_secs = stats_window_duration.as_secs_f32();

                alvr_events::send_event(EventType::StatisticsSummary(StatisticsSummary {
                    video_packets_total: self.video_packets_total,
                    video_packets_per_sec: (self.video_packets_partial_sum as f32 / interval_secs)
                        as usize,
                    video_mbytes_total: (self.video_bytes_total as f32 / 1e6) as usize,
                    video_mbits_per_sec: self.video_bytes_partial_sum as f32 * 8.
                        / 1e6
                        / interval_secs,
                    total_latency_ms: client_stats.total_pipeline_latency.as_secs_f32() * 1000.,
                    network_latency_ms: network_latency.as_secs_f32() * 1000.,
                    encode_latency_ms: encoder_latency.as_secs_f32() * 1000.,
                    decode_latency_ms: client_stats.video_decode.as_secs_f32() * 1000.,
                    packets_lost_total: self.packets_lost_total,
                    packets_lost_per_sec: (self.packets_lost_partial_sum as f32 / interval_secs)
                        as usize,
                    client_fps: client_fps as u32,
                    server_fps: server_fps as u32,
                    battery_hmd: (self
                        .battery_gauges
                        .get(&HEAD_ID)
                        .cloned()
                        .unwrap_or_default()
                        .gauge_value
                        * 100.) as u32,
                    hmd_plugged: self
                        .battery_gauges
                        .get(&HEAD_ID)
                        .cloned()
                        .unwrap_or_default()
                        .is_plugged,
                }));
                self.video_packets_partial_sum = 0;
                self.video_bytes_partial_sum = 0;
                self.packets_lost_partial_sum = 0;
                self.partial_stats_last_reset = now;
            }

            let throughput_bps =
                (frame.video_packet_bytes as f32 * 8.0) / safe_network_latency.as_secs_f32();
            alvr_events::send_event(EventType::GraphStatistics(GraphStatistics {
                total_pipeline_latency_s: client_stats.total_pipeline_latency.as_secs_f32(),
                game_time_s: game_time_latency.as_secs_f32(),
                server_compositor_s: server_compositor_latency.as_secs_f32(),
                encoder_s: encoder_latency.as_secs_f32(),
                network_s: network_latency.as_secs_f32(),
                decoder_s: client_stats.video_decode.as_secs_f32(),
                decoder_queue_s: client_stats.video_decoder_queue.as_secs_f32(),
                client_compositor_s: client_stats.rendering.as_secs_f32(),
                vsync_queue_s: client_stats.vsync_queue.as_secs_f32(),
                client_fps,
                server_fps,
                bitrate_directives: self.last_throughput_directives.clone(),
                throughput_bps,
                bitrate_bps: (frame.video_packet_bytes as f32 * 8.0)
                    / safe_present_interval.as_secs_f32(),
            }));
            (network_latency, game_time_latency)
        } else {
            (Duration::ZERO, Duration::ZERO)
        }
    }

    pub fn motion_to_photon_latency_average(&self) -> Duration {
        self.motion_to_photon_latency_average.get_average()
    }

    pub fn tracker_pose_time_offset(&self) -> Duration {
        self.steamvr_pipeline_latency
    }

    pub fn duration_until_next_vsync(&mut self) -> Duration {
        let now = Instant::now();
        while self.last_vsync_time + self.frame_interval < now {
            self.last_vsync_time += self.frame_interval;
        }
        (self.last_vsync_time + self.frame_interval).saturating_duration_since(now)
    }
}
