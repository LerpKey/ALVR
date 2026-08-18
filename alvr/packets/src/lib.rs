use alvr_common::{
    anyhow::Result,
    glam::{UVec2, Vec2},
    semver::Version,
    ConnectionState, DeviceMotion, Fov, LogEntry, LogSeverity, Pose, ToAny,
};
use alvr_session::{
    ClientsidePostProcessingConfig, CodecType, PassthroughMode, SessionConfig, Settings,
};
use serde::{Deserialize, Serialize};
use serde_json as json;
use std::{
    collections::HashSet,
    fmt::{self, Debug},
    net::IpAddr,
    path::PathBuf,
    time::Duration,
};

pub const TRACKING: u16 = 0;
pub const HAPTICS: u16 = 1;
pub const AUDIO: u16 = 2;
pub const VIDEO: u16 = 3;
pub const STATISTICS: u16 = 4;

// todo: use simple string
#[derive(Serialize, Deserialize, Clone)]
pub struct VideoStreamingCapabilitiesLegacy {
    pub default_view_resolution: UVec2,
    pub supported_refresh_rates_plus_extra_data: Vec<f32>,
    pub microphone_sample_rate: u32,
}

// Note: not a network packet
#[derive(Serialize, Deserialize, Clone)]
pub struct VideoStreamingCapabilities {
    pub default_view_resolution: UVec2,
    pub supported_refresh_rates: Vec<f32>, // todo rename
    pub microphone_sample_rate: u32,
    pub supports_foveated_encoding: bool, // todo rename
    pub encoder_high_profile: bool,
    pub encoder_10_bits: bool,
    pub encoder_av1: bool,
    pub multimodal_protocol: bool,
    pub prefer_10bit: bool,
    pub prefer_full_range: bool,
    pub preferred_encoding_gamma: f32,
    pub prefer_hdr: bool,
}

// Nasty workaround to make the packet extensible, pushing the limits of protocol compatibility
// Todo: replace VideoStreamingCapabilitiesLegacy with simple json string
pub fn encode_video_streaming_capabilities(
    caps: &VideoStreamingCapabilities,
) -> Result<VideoStreamingCapabilitiesLegacy> {
    let caps_json = json::to_value(caps)?;

    let mut supported_refresh_rates_plus_extra_data = vec![];
    for rate in caps_json["supported_refresh_rates"].as_array().to_any()? {
        supported_refresh_rates_plus_extra_data.push(rate.as_f64().to_any()? as f32);
    }
    for byte in json::to_string(caps)?.as_bytes() {
        // using negative values is not going to trigger strange behavior for old servers
        supported_refresh_rates_plus_extra_data.push(-(*byte as f32));
    }

    let default_view_resolution = json::from_value(caps_json["default_view_resolution"].clone())?;
    let microphone_sample_rate = caps_json["microphone_sample_rate"].as_u64().to_any()? as u32;

    Ok(VideoStreamingCapabilitiesLegacy {
        default_view_resolution,
        supported_refresh_rates_plus_extra_data,
        microphone_sample_rate,
    })
}

pub fn decode_video_streaming_capabilities(
    legacy: &VideoStreamingCapabilitiesLegacy,
) -> Result<VideoStreamingCapabilities> {
    let mut json_bytes = vec![];
    let mut supported_refresh_rates = vec![];
    for rate in &legacy.supported_refresh_rates_plus_extra_data {
        if *rate < 0.0 {
            json_bytes.push((-*rate) as u8)
        } else {
            supported_refresh_rates.push(*rate);
        }
    }

    let caps_json =
        json::from_str::<json::Value>(&String::from_utf8(json_bytes)?).unwrap_or(json::Value::Null);

    Ok(VideoStreamingCapabilities {
        default_view_resolution: legacy.default_view_resolution,
        supported_refresh_rates,
        microphone_sample_rate: legacy.microphone_sample_rate,
        supports_foveated_encoding: caps_json["supports_foveated_encoding"]
            .as_bool()
            .unwrap_or(true),
        encoder_high_profile: caps_json["encoder_high_profile"].as_bool().unwrap_or(true),
        encoder_10_bits: caps_json["encoder_10_bits"].as_bool().unwrap_or(true),
        encoder_av1: caps_json["encoder_av1"].as_bool().unwrap_or(true),
        multimodal_protocol: caps_json["multimodal_protocol"].as_bool().unwrap_or(false),
        prefer_10bit: caps_json["prefer_10bit"].as_bool().unwrap_or(false),
        prefer_full_range: caps_json["prefer_full_range"].as_bool().unwrap_or(true),
        preferred_encoding_gamma: caps_json["preferred_encoding_gamma"]
            .as_f64()
            .unwrap_or(1.0) as f32,
        prefer_hdr: caps_json["prefer_hdr"].as_bool().unwrap_or(false),
    })
}

#[derive(Serialize, Deserialize)]
pub enum ClientConnectionResult {
    ConnectionAccepted {
        client_protocol_id: u64,
        display_name: String,
        server_ip: IpAddr,
        streaming_capabilities: Option<VideoStreamingCapabilitiesLegacy>, // todo: use String
    },
    ClientStandby,
}

// Note: not a network packet
#[derive(Serialize, Deserialize, Clone)]
pub struct NegotiatedStreamingConfig {
    pub view_resolution: UVec2,
    pub refresh_rate_hint: f32,
    pub game_audio_sample_rate: u32,
    pub enable_foveated_encoding: bool,
    // This is needed to detect when to use SteamVR hand trackers. This does NOT imply if multimodal
    // input is supported
    pub use_multimodal_protocol: bool,
    pub use_full_range: bool,
    pub encoding_gamma: f32,
    pub enable_hdr: bool,
    pub wired: bool,
}

#[derive(Serialize, Deserialize)]
pub struct StreamConfigPacket {
    pub session: String,    // JSON session that allows for extrapolation
    pub negotiated: String, // Encoded NegotiatedVideoStreamingConfig
}

pub fn encode_stream_config(
    session: &SessionConfig,
    negotiated: &NegotiatedStreamingConfig,
) -> Result<StreamConfigPacket> {
    Ok(StreamConfigPacket {
        session: json::to_string(session)?,
        negotiated: json::to_string(negotiated)?,
    })
}

#[derive(Serialize, Deserialize, Clone)]
pub struct StreamConfig {
    pub server_version: Version,
    pub settings: Settings,
    pub negotiated_config: NegotiatedStreamingConfig,
}

pub fn decode_stream_config(packet: &StreamConfigPacket) -> Result<StreamConfig> {
    let mut session_config = SessionConfig::default();
    session_config.merge_from_json(&json::from_str(&packet.session)?)?;
    let settings = session_config.to_settings();

    let negotiated_json = json::from_str::<json::Value>(&packet.negotiated)?;

    let view_resolution = json::from_value(negotiated_json["view_resolution"].clone())?;
    let refresh_rate_hint = json::from_value(negotiated_json["refresh_rate_hint"].clone())?;
    let game_audio_sample_rate =
        json::from_value(negotiated_json["game_audio_sample_rate"].clone())?;
    let enable_foveated_encoding =
        json::from_value(negotiated_json["enable_foveated_encoding"].clone())
            .unwrap_or_else(|_| settings.video.foveated_encoding.enabled());
    let use_multimodal_protocol =
        json::from_value(negotiated_json["use_multimodal_protocol"].clone()).unwrap_or(false);
    let use_full_range = json::from_value(negotiated_json["use_full_range"].clone())
        .unwrap_or(settings.video.encoder_config.use_full_range);
    let encoding_gamma = json::from_value(negotiated_json["encoding_gamma"].clone()).unwrap_or(1.0);
    let enable_hdr = json::from_value(negotiated_json["enable_hdr"].clone()).unwrap_or(false);
    let wired = json::from_value(negotiated_json["wired"].clone()).unwrap_or(false);

    Ok(StreamConfig {
        server_version: session_config.server_version,
        settings,
        negotiated_config: NegotiatedStreamingConfig {
            view_resolution,
            refresh_rate_hint,
            game_audio_sample_rate,
            enable_foveated_encoding,
            use_multimodal_protocol,
            use_full_range,
            encoding_gamma,
            enable_hdr,
            wired,
        },
    })
}

#[derive(Serialize, Deserialize, Clone)]
pub struct DecoderInitializationConfig {
    pub codec: CodecType,
    pub config_buffer: Vec<u8>, // e.g. SPS + PPS NALs
}

#[derive(Serialize, Deserialize)]
pub enum ServerControlPacket {
    StartStream,
    DecoderConfig(DecoderInitializationConfig),
    Restarting,
    KeepAlive,
    ServerPredictionAverage(Duration), // todo: remove
    Reserved(String),
    ReservedBuffer(Vec<u8>),
}

#[derive(Serialize, Deserialize, Clone)]
pub struct ViewsConfig {
    // Note: the head-to-eye transform is always a translation along the x axis
    pub ipd_m: f32,
    pub fov: [Fov; 2],
}

#[derive(Serialize, Deserialize, Clone)]
pub struct BatteryInfo {
    pub device_id: u64,
    pub gauge_value: f32, // range [0, 1]
    pub is_plugged: bool,
}

// WiFi signal metrics for PHY/MAC layer monitoring
// Collected at tracking frequency (3x refresh rate) for DRL-based ABR research
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct WiFiMetrics {
    pub timestamp_ns: u64,                    // Nanosecond precision timestamp (xr_runtime_now)
    pub rssi_dbm: i32,                        // Received Signal Strength Indicator (-100 to 0 dBm)
    pub frequency_mhz: u32,                   // WiFi frequency band (2400, 5000, 6000 MHz)
    pub link_speed_mbps: u32,                 // Current link speed in Mbps
    pub mcs_index: Option<u8>,                // Modulation and Coding Scheme index (0-9)
    pub snr_db: Option<f32>,                  // Signal-to-Noise Ratio in dB
    pub tx_bitrate_mbps: Option<f32>,         // Transmit bitrate in Mbps
    pub rx_bitrate_mbps: Option<f32>,         // Receive bitrate in Mbps
    pub tx_retries: Option<u32>,              // Number of transmit retries
    pub tx_failures: Option<u32>,             // Number of transmit failures
    pub wifi_standard: String,                // "802.11ac", "802.11ax", etc.
    pub channel_width: u32,                   // Channel width in MHz (20, 40, 80, 160)
    pub guard_interval: Option<u32>,          // Guard interval in ns (400, 800)
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug)]
pub enum ButtonValue {
    Binary(bool),
    Scalar(f32),
}

#[derive(Serialize, Deserialize)]
pub struct ButtonEntry {
    pub path_id: u64,
    pub value: ButtonValue,
}

// to be de/serialized with ClientControlPacket::Reserved()
#[derive(Serialize, Deserialize)]
pub enum ReservedClientControlPacket {
    CustomInteractionProfile {
        device_id: u64,
        input_ids: HashSet<u64>,
    },
}

pub fn encode_reserved_client_control_packet(
    packet: &ReservedClientControlPacket,
) -> ClientControlPacket {
    ClientControlPacket::Reserved(json::to_string(packet).unwrap())
}

#[derive(Serialize, Deserialize)]
pub enum ClientControlPacket {
    PlayspaceSync(Option<Vec2>),
    RequestIdr,
    KeepAlive,
    StreamReady, // This flag notifies the server the client streaming socket is ready listening
    ViewsConfig(ViewsConfig),
    Battery(BatteryInfo),
    VideoErrorReport, // legacy
    Buttons(Vec<ButtonEntry>),
    ActiveInteractionProfile { device_id: u64, profile_id: u64 },
    Log { level: LogSeverity, message: String },
    Reserved(String),
    ReservedBuffer(Vec<u8>),
}

// PICO Eye Pose Status - matches XrEyePoseStatusPICO enum (u32 bitfield)
pub type EyePoseStatusPICO = u32;

// PICO Eye Pose Status bit flags (from PICO documentation)
pub mod eye_pose_status_pico {
    // Primary gaze data flags (most important)
    pub const GAZE_POINT_VALID: u32 = 1 << 0; // XR_ET_GAZE_POINT_VALID_PICO
    pub const GAZE_VECTOR_VALID: u32 = 1 << 1; // XR_ET_GAZE_VECTOR_VALID_PICO
    pub const EYE_OPENNESS_VALID: u32 = 1 << 2; // XR_ET_EYE_OPENNESS_VALID_PICO
    pub const EYE_PUPIL_DILATION_VALID: u32 = 1 << 3; // XR_ET_EYE_PUPIL_DILATION_VALID_PICO
    pub const EYE_POSITION_GUIDE_VALID: u32 = 1 << 4; // XR_ET_EYE_POSITION_GUIDE_VALID_PICO
    pub const EYE_PUPIL_POSITION_VALID: u32 = 1 << 5; // XR_ET_EYE_PUPIL_POSITION_VALID_PICO
    pub const EYE_CONVERGENCE_DISTANCE_VALID: u32 = 1 << 6; // XR_ET_EYE_CONVERGENCE_DISTANCE_VALID_PICO

    // Extended flags (possibly duplicates or different meanings)
    pub const EYE_GAZE_POINT_VALID_EXT: u32 = 1 << 7; // XR_ET_EYE_GAZE_POINT_VALID_PICO (duplicate?)
    pub const EYE_GAZE_VECTOR_VALID_EXT: u32 = 1 << 8; // XR_ET_EYE_GAZE_VECTOR_VALID_PICO (duplicate?)
    pub const PUPIL_DISTANCE_VALID: u32 = 1 << 9; // XR_ET_PUPIL_DISTANCE_VALID_PICO
    pub const CONVERGENCE_DISTANCE_VALID_EXT: u32 = 1 << 10; // XR_ET_CONVERGENCE_DISTANCE_VALID_PICO (duplicate?)
    pub const PUPIL_DIAMETER_VALID: u32 = 1 << 11; // XR_ET_PUPIL_DIAMETER_VALID_PICO

    pub fn has_flag(status: u32, flag: u32) -> bool {
        (status & flag) != 0
    }

    pub fn is_gaze_point_valid(status: u32) -> bool {
        // Check both primary and extended flags
        has_flag(status, GAZE_POINT_VALID) || has_flag(status, EYE_GAZE_POINT_VALID_EXT)
    }

    pub fn is_gaze_vector_valid(status: u32) -> bool {
        // Check both primary and extended flags
        has_flag(status, GAZE_VECTOR_VALID) || has_flag(status, EYE_GAZE_VECTOR_VALID_EXT)
    }

    pub fn is_eye_openness_valid(status: u32) -> bool {
        has_flag(status, EYE_OPENNESS_VALID)
    }

    pub fn is_pupil_dilation_valid(status: u32) -> bool {
        // Check both primary (bit 3) and extended (bit 11) pupil flags
        // PICO firmware may use either EYE_PUPIL_DILATION_VALID or PUPIL_DIAMETER_VALID
        // depending on headset model/version (similar to gaze_vector dual-flag pattern)
        has_flag(status, EYE_PUPIL_DILATION_VALID) || has_flag(status, PUPIL_DIAMETER_VALID)
    }

    // Diagnostic function to show all flag states
    pub fn get_flag_debug_info(status: u32) -> String {
        format!(
            "Status: {} | GP0:{} GP7:{} | GV1:{} GV8:{} | EO2:{} | PD3:{} PD11:{} | Raw: {:012b}",
            status,
            has_flag(status, GAZE_POINT_VALID),
            has_flag(status, EYE_GAZE_POINT_VALID_EXT),
            has_flag(status, GAZE_VECTOR_VALID),
            has_flag(status, EYE_GAZE_VECTOR_VALID_EXT),
            has_flag(status, EYE_OPENNESS_VALID),
            has_flag(status, EYE_PUPIL_DILATION_VALID),
            has_flag(status, PUPIL_DIAMETER_VALID),
            status
        )
    }
}

#[repr(C)]
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct EyeTrackingDataPICO {
    pub time: u64, // sys::Time as u64
    pub left_eye_pose_status: EyePoseStatusPICO,
    pub right_eye_pose_status: EyePoseStatusPICO,
    pub combined_eye_pose_status: EyePoseStatusPICO,
    pub left_eye_gaze_point: [f32; 3],
    pub right_eye_gaze_point: [f32; 3],
    pub combined_eye_gaze_point: [f32; 3],
    pub left_eye_gaze_vector: [f32; 3],
    pub right_eye_gaze_vector: [f32; 3],
    pub combined_eye_gaze_vector: [f32; 3],
    pub left_eye_openness: f32,
    pub right_eye_openness: f32,
    pub left_eye_pupil_dilation: f32,
    pub right_eye_pupil_dilation: f32,
    pub left_eye_position_guide: [f32; 3],
    pub right_eye_position_guide: [f32; 3],
    pub foveated_gaze_direction: [f32; 3],
    pub foveated_gaze_tracking_state: i32,
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum EyeTrackingInputStatus {
    #[default]
    Unsupported,
    /// Eye tracking hardware is present but currently not feeding gaze samples (eyes closed or HMD removed)
    Standby,
    /// Valid gaze data is being produced this frame
    Active,
}

#[derive(Serialize, Deserialize, Clone, Default)]
pub struct FaceData {
    pub eye_gazes: [Option<Pose>; 2],
    pub fb_face_expression: Option<Vec<f32>>, // issue: Serialize does not support [f32; 63]
    pub htc_eye_expression: Option<Vec<f32>>,
    pub htc_lip_expression: Option<Vec<f32>>, // issue: Serialize does not support [f32; 37]
    pub pico_eye_tracking_data: Option<EyeTrackingDataPICO>, // PICO eye tracking data
    #[serde(default)]
    pub eye_tracking_state: EyeTrackingInputStatus,
}

#[derive(Serialize, Deserialize, Clone, Copy)]
pub struct DFRShiftData {
    pub shift_x: f32,
    pub shift_y: f32,
    pub left_shift_x: f32,
    pub left_shift_y: f32,
    pub right_shift_x: f32,
    pub right_shift_y: f32,
    pub sequence_id: u64,
    pub is_eye_tracked: bool,
}

#[derive(Serialize, Deserialize)]
pub struct VideoPacketHeader {
    pub timestamp: Duration,
    pub is_idr: bool,
    pub dfr_shift: Option<DFRShiftData>,
}

// Note: face_data does not respect target_timestamp.
#[derive(Serialize, Deserialize, Default)]
pub struct Tracking {
    pub target_timestamp: Duration,
    pub device_motions: Vec<(u64, DeviceMotion)>,
    pub hand_skeletons: [Option<[Pose; 26]>; 2],
    pub face_data: FaceData,
    pub wifi_metrics: Option<WiFiMetrics>, // High-frequency WiFi signal monitoring
    // Raw HEAD motion before linear extrapolation prediction.
    // Used for telemetry to compare actual vs predicted poses.
    pub raw_head_motion: Option<DeviceMotion>,
}

#[derive(Serialize, Deserialize)]
pub struct Haptics {
    pub device_id: u64,
    pub duration: Duration,
    pub frequency: f32,
    pub amplitude: f32,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct AudioDevicesList {
    pub output: Vec<String>,
    pub input: Vec<String>,
}

#[derive(Serialize, Deserialize, Clone)]
pub enum PathSegment {
    Name(String),
    Index(usize),
}

impl Debug for PathSegment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PathSegment::Name(name) => write!(f, "{name}"),
            PathSegment::Index(index) => write!(f, "[{index}]"),
        }
    }
}

impl From<&str> for PathSegment {
    fn from(value: &str) -> Self {
        PathSegment::Name(value.to_owned())
    }
}

impl From<String> for PathSegment {
    fn from(value: String) -> Self {
        PathSegment::Name(value)
    }
}

impl From<usize> for PathSegment {
    fn from(value: usize) -> Self {
        PathSegment::Index(value)
    }
}

// todo: support indices
pub fn parse_path(path: &str) -> Vec<PathSegment> {
    path.split('.').map(|s| s.into()).collect()
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum ClientListAction {
    AddIfMissing {
        trusted: bool,
        manual_ips: Vec<IpAddr>,
    },
    SetDisplayName(String),
    Trust,
    SetManualIps(Vec<IpAddr>),
    RemoveEntry,
    UpdateCurrentIp(Option<IpAddr>),
    SetConnectionState(ConnectionState),
}

#[derive(Serialize, Deserialize, Default, Clone)]
pub struct ClientStatistics {
    pub target_timestamp: Duration, // identifies the frame
    pub frame_interval: Duration,
    pub video_decode: Duration,
    pub video_decoder_queue: Duration,
    pub rendering: Duration,
    pub vsync_queue: Duration,
    pub total_pipeline_latency: Duration,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct PathValuePair {
    pub path: Vec<PathSegment>,
    pub value: json::Value,
}

#[derive(Serialize, Deserialize, Debug)]
pub enum FirewallRulesAction {
    Add,
    Remove,
}

#[derive(Serialize, Deserialize, Debug)]
pub enum ServerRequest {
    Log(LogEntry),
    GetSession,
    UpdateSession(Box<SessionConfig>),
    SetValues(Vec<PathValuePair>),
    UpdateClientList {
        hostname: String,
        action: ClientListAction,
    },
    GetAudioDevices,
    CaptureFrame,
    InsertIdr,
    StartRecording,
    StopRecording,
    FirewallRules(FirewallRulesAction),
    RegisterAlvrDriver,
    UnregisterDriver(PathBuf),
    GetDriverList,
    RestartSteamvr,
    ShutdownSteamvr,
}

// Note: server sends a packet to the client at low frequency, binary encoding, without ensuring
// compatibility between different versions, even if within the same major version.
#[derive(Serialize, Deserialize)]
pub struct RealTimeConfig {
    pub passthrough: Option<PassthroughMode>,
    pub clientside_post_processing: Option<ClientsidePostProcessingConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dynamic_foveated_center: Option<DynamicFoveatedCenter>,
}

// Dynamic foveated center for eye tracking with per-eye support
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct DynamicFoveatedCenter {
    pub center_shift_x: f32, // Combined horizontal center shift [-1, 1] (legacy/fallback)
    pub center_shift_y: f32, // Combined vertical center shift [-1, 1] (legacy/fallback)
    #[serde(default)]
    pub left_shift_x: f32, // Left eye horizontal shift [-1, 1]
    #[serde(default)]
    pub left_shift_y: f32, // Left eye vertical shift [-1, 1]
    #[serde(default)]
    pub right_shift_x: f32, // Right eye horizontal shift [-1, 1]
    #[serde(default)]
    pub right_shift_y: f32, // Right eye vertical shift [-1, 1]
    #[serde(default)]
    pub sequence_id: Option<u64>, // Synchronization sequence ID for pipeline consistency
}

impl RealTimeConfig {
    pub fn encode(&self) -> Result<ServerControlPacket> {
        Ok(ServerControlPacket::ReservedBuffer(bincode::serialize(
            self,
        )?))
    }

    pub fn decode(buffer: &[u8]) -> Result<Self> {
        Ok(bincode::deserialize(buffer)?)
    }

    pub fn from_settings(settings: &Settings) -> Self {
        Self {
            passthrough: settings.video.passthrough.clone().into_option(),
            clientside_post_processing: settings
                .video
                .clientside_post_processing
                .clone()
                .into_option(),
            dynamic_foveated_center: None, // Will be set by eye tracking system
        }
    }
}

// Per eye view parameters
// todo: send together with video frame
#[derive(Serialize, Deserialize, Clone, Copy, Default)]
pub struct ViewParams {
    pub pose: Pose,
    pub fov: Fov,
}
