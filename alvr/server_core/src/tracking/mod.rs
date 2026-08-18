mod body;
mod face;
mod vmc;
pub mod eye_tracked_ffr;

pub use body::*;
pub use face::*;
pub use vmc::*;
pub use eye_tracked_ffr::*;

use crate::{
    connection::STREAMING_RECV_TIMEOUT,
    hand_gestures::{self, HandGestureManager, HAND_GESTURE_BUTTON_SET},
    input_mapping::ButtonMappingManager,
    ConnectionContext, ServerCoreEvent, SESSION_MANAGER, FILESYSTEM_LAYOUT,
};
use alvr_common::{
    glam::{EulerRot, Quat, Vec3},
    parking_lot::Mutex,
    ConnectionError, DeviceMotion, Pose, BODY_CHEST_ID, BODY_HIPS_ID, BODY_LEFT_ELBOW_ID,
    BODY_LEFT_FOOT_ID, BODY_LEFT_KNEE_ID, BODY_RIGHT_ELBOW_ID, BODY_RIGHT_FOOT_ID,
    BODY_RIGHT_KNEE_ID, DEVICE_ID_TO_PATH, HAND_LEFT_ID, HAND_RIGHT_ID, HEAD_ID,
};
use alvr_events::{EventType, TrackingEvent};
use alvr_packets::{EyeTrackingDataPICO, FaceData, Tracking};
use serde::{Deserialize, Serialize};
use alvr_session::{
    settings_schema::Switch, BodyTrackingConfig, HeadsetConfig, PositionRecenteringMode,
    RotationRecenteringMode, Settings, VMCConfig,
};
use alvr_sockets::StreamReceiver;
use chrono;
use serde_json;
use std::{
    cmp::Ordering,
    collections::{HashMap, VecDeque},
    f32::consts::PI,
    fs::{self, OpenOptions},
    io::Write,
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

const DEG_TO_RAD: f32 = PI / 180.0;

// Timer for tracking file rotation
static LAST_FILE_MINUTE: std::sync::Mutex<Option<u64>> = std::sync::Mutex::new(None);

// Global DFR renderer instance
static DFR_RENDERER: std::sync::LazyLock<DynamicFoveatedRenderer> =
    std::sync::LazyLock::new(|| DynamicFoveatedRenderer::new());

/// PICO Eye Pose Status for logging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EyePoseStatusLog {
    pub status_flags: u32,
    pub gaze_point_valid: bool,
    pub gaze_vector_valid: bool,
    pub eye_openness_valid: bool,
    pub pupil_dilation_valid: bool,
}

/// PICO Eye Tracking Data structure for logging (mirrors client-side EyeTrackingDataPICO)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EyeTrackingDataLog {
    pub timestamp_ns: u64,
    pub left_eye_pose_status: EyePoseStatusLog,
    pub right_eye_pose_status: EyePoseStatusLog,
    pub combined_eye_pose_status: EyePoseStatusLog,
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

/// Validates if eye tracking data is genuine (not default/invalid values)
fn is_valid_eye_data(pose: &Pose) -> bool {
    // Check if all values are non-zero or not too close to default values
    let pos = pose.position;
    let ori = pose.orientation;
    
    // If position is essentially zero and orientation is identity, likely invalid
    let pos_near_zero = pos.length_squared() < 1e-6;
    let ori_near_identity = (ori.w.abs() - 1.0).abs() < 1e-6 && 
                           ori.x.abs() < 1e-6 && ori.y.abs() < 1e-6 && ori.z.abs() < 1e-6;
    
    // Valid data should have some meaningful values
    !pos_near_zero || !ori_near_identity
}

/// Gets current time components for file naming
fn get_time_components() -> (String, u64, u64) {
    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
    let timestamp_ms = now.as_millis() as u64;
    let unix_timestamp = now.as_secs();
    let current_minute = unix_timestamp / 60;
    
    // Create local time string in yy-mm-dd-hh-mm format
    let local_time = chrono::DateTime::<chrono::Local>::from(SystemTime::now());
    let time_str = local_time.format("%y-%m-%d-%H-%M").to_string();
    
    (time_str, current_minute, timestamp_ms)
}


fn log_gaze_data_to_file(face_data: &FaceData, timestamp: Duration) {
    if let Some(filesystem_layout) = FILESYSTEM_LAYOUT.get() {
        // Use the actual timestamp from tracking loop (system uptime)
        let timestamp_ms = timestamp.as_millis() as u64;
        // Convert system uptime to unix UTC timestamp
        let unix_timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        
        // For file naming, still use current time to avoid too frequent file creation  
        let (time_str, current_minute, _) = get_time_components();
        
        // Create gazelog directory in build folder
        let gazelog_dir = filesystem_layout.log_dir.join("gazelog");
        if let Err(_) = fs::create_dir_all(&gazelog_dir) {
            return;
        }
        
        // Check if we need to rotate files (new minute)
        let _should_rotate = {
            let mut last_minute = LAST_FILE_MINUTE.lock().unwrap();
            match *last_minute {
                None => {
                    *last_minute = Some(current_minute);
                    true
                }
                Some(last) if last != current_minute => {
                    *last_minute = Some(current_minute);
                    true
                }
                _ => false,
            }
        };
        
        // Only log if we have genuinely valid eye gaze data
        let left_valid = face_data.eye_gazes[0].as_ref().map_or(false, is_valid_eye_data);
        let right_valid = face_data.eye_gazes[1].as_ref().map_or(false, is_valid_eye_data);
        
        if !left_valid && !right_valid {
            return; // No valid eye tracking data
        }
        
        let json_path = gazelog_dir.join(format!("{}.json", time_str));
        let csv_path = gazelog_dir.join(format!("{}.csv", time_str));
        
        // Create CSV header if file doesn't exist
        let csv_exists = csv_path.exists();
        
        // Prepare the data structure
        
        // Log JSON format
        if let Ok(mut json_file) = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&json_path)
        {
            let mut device_motion_data = serde_json::Map::new();
            
            if left_valid {
                if let Some(left_eye) = &face_data.eye_gazes[0] {
                    device_motion_data.insert("left_eye".to_string(), serde_json::json!({
                        "position": [left_eye.position.x, left_eye.position.y, left_eye.position.z],
                        "orientation": [left_eye.orientation.x, left_eye.orientation.y, left_eye.orientation.z, left_eye.orientation.w]
                    }));
                }
            } else {
                device_motion_data.insert("left_eye".to_string(), serde_json::Value::Null);
            }
            
            if right_valid {
                if let Some(right_eye) = &face_data.eye_gazes[1] {
                    device_motion_data.insert("right_eye".to_string(), serde_json::json!({
                        "position": [right_eye.position.x, right_eye.position.y, right_eye.position.z],
                        "orientation": [right_eye.orientation.x, right_eye.orientation.y, right_eye.orientation.z, right_eye.orientation.w]
                    }));
                }
            } else {
                device_motion_data.insert("right_eye".to_string(), serde_json::Value::Null);
            }
            
            let log_entry = serde_json::json!({
                "timestamp_ms": timestamp_ms,
                "unix_timestamp": unix_timestamp,
                "device_motion_key": "eye_tracking",
                "data": device_motion_data
            });
            
            let _ = writeln!(json_file, "{}", log_entry);
            let _ = json_file.flush();
        }
        
        // Log CSV format
        if let Ok(mut csv_file) = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&csv_path)
        {
            // Write header if new file
            if !csv_exists {
                let _ = writeln!(csv_file, "timestamp_ms,unix_timestamp,device_motion_key,left_eye_pos_x,left_eye_pos_y,left_eye_pos_z,left_eye_ori_x,left_eye_ori_y,left_eye_ori_z,left_eye_ori_w,right_eye_pos_x,right_eye_pos_y,right_eye_pos_z,right_eye_ori_x,right_eye_ori_y,right_eye_ori_z,right_eye_ori_w");
            }
            
            // Prepare CSV row data
            let mut row = vec![
                timestamp_ms.to_string(),
                unix_timestamp.to_string(),
                "eye_tracking".to_string(),
            ];
            
            // Left eye data
            if left_valid {
                if let Some(left_eye) = &face_data.eye_gazes[0] {
                    row.extend(vec![
                        left_eye.position.x.to_string(),
                        left_eye.position.y.to_string(),
                        left_eye.position.z.to_string(),
                        left_eye.orientation.x.to_string(),
                        left_eye.orientation.y.to_string(),
                        left_eye.orientation.z.to_string(),
                        left_eye.orientation.w.to_string(),
                    ]);
                } else {
                    row.extend(vec!["null".to_string(); 7]);
                }
            } else {
                row.extend(vec!["null".to_string(); 7]);
            }
            
            // Right eye data
            if right_valid {
                if let Some(right_eye) = &face_data.eye_gazes[1] {
                    row.extend(vec![
                        right_eye.position.x.to_string(),
                        right_eye.position.y.to_string(),
                        right_eye.position.z.to_string(),
                        right_eye.orientation.x.to_string(),
                        right_eye.orientation.y.to_string(),
                        right_eye.orientation.z.to_string(),
                        right_eye.orientation.w.to_string(),
                    ]);
                } else {
                    row.extend(vec!["null".to_string(); 7]);
                }
            } else {
                row.extend(vec!["null".to_string(); 7]);
            }
            
            let _ = writeln!(csv_file, "{}", row.join(","));
            let _ = csv_file.flush();
        }
    }
}

/// Log PICO eye tracking data to file for debugging and analysis
fn log_eye_tracking_data_to_file(eye_data: &EyeTrackingDataLog, timestamp: Duration) {
    if let Some(filesystem_layout) = FILESYSTEM_LAYOUT.get() {
        // Use the actual timestamp from tracking loop (system uptime)
        let timestamp_ms = timestamp.as_millis() as u64;
        // Convert system uptime to unix UTC timestamp
        let unix_timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        
        // For file naming, still use current time to avoid too frequent file creation
        let (time_str, current_minute, _) = get_time_components();
        
        // Create eyelog directory in build folder  
        let eyelog_dir = filesystem_layout.log_dir.join("eyelog");
        if let Err(_) = fs::create_dir_all(&eyelog_dir) {
            return;
        }
        
        // Check if we need to rotate files (new minute)
        let _should_rotate = {
            let mut last_minute = LAST_FILE_MINUTE.lock().unwrap();
            match *last_minute {
                None => {
                    *last_minute = Some(current_minute);
                    true
                }
                Some(last) if last != current_minute => {
                    *last_minute = Some(current_minute);
                    true
                }
                _ => false,
            }
        };
        
        let json_path = eyelog_dir.join(format!("{}.json", time_str));
        let csv_path = eyelog_dir.join(format!("{}.csv", time_str));
        
        // Create CSV header if file doesn't exist
        let csv_exists = csv_path.exists();
        
        // Log JSON format - complete eye tracking data
        if let Ok(mut json_file) = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&json_path)
        {
            let log_entry = serde_json::json!({
                "timestamp_ms": timestamp_ms,
                "unix_timestamp": unix_timestamp,
                "device_motion_key": "pico_eye_tracking_raw",
                "data": {
                    "timestamp_ns": eye_data.timestamp_ns,
                    "left_eye_pose_status": {
                        "status_flags": eye_data.left_eye_pose_status.status_flags,
                        "gaze_point_valid": eye_data.left_eye_pose_status.gaze_point_valid,
                        "gaze_vector_valid": eye_data.left_eye_pose_status.gaze_vector_valid,
                        "eye_openness_valid": eye_data.left_eye_pose_status.eye_openness_valid,
                        "pupil_dilation_valid": eye_data.left_eye_pose_status.pupil_dilation_valid
                    },
                    "right_eye_pose_status": {
                        "status_flags": eye_data.right_eye_pose_status.status_flags,
                        "gaze_point_valid": eye_data.right_eye_pose_status.gaze_point_valid,
                        "gaze_vector_valid": eye_data.right_eye_pose_status.gaze_vector_valid,
                        "eye_openness_valid": eye_data.right_eye_pose_status.eye_openness_valid,
                        "pupil_dilation_valid": eye_data.right_eye_pose_status.pupil_dilation_valid
                    },
                    "combined_eye_pose_status": {
                        "status_flags": eye_data.combined_eye_pose_status.status_flags,
                        "gaze_point_valid": eye_data.combined_eye_pose_status.gaze_point_valid,
                        "gaze_vector_valid": eye_data.combined_eye_pose_status.gaze_vector_valid,
                        "eye_openness_valid": eye_data.combined_eye_pose_status.eye_openness_valid,
                        "pupil_dilation_valid": eye_data.combined_eye_pose_status.pupil_dilation_valid
                    },
                    "left_eye_gaze_point": eye_data.left_eye_gaze_point,
                    "right_eye_gaze_point": eye_data.right_eye_gaze_point,
                    "combined_eye_gaze_point": eye_data.combined_eye_gaze_point,
                    "left_eye_gaze_vector": eye_data.left_eye_gaze_vector,
                    "right_eye_gaze_vector": eye_data.right_eye_gaze_vector,
                    "combined_eye_gaze_vector": eye_data.combined_eye_gaze_vector,
                    "left_eye_openness": eye_data.left_eye_openness,
                    "right_eye_openness": eye_data.right_eye_openness,
                    "left_eye_pupil_dilation": eye_data.left_eye_pupil_dilation,
                    "right_eye_pupil_dilation": eye_data.right_eye_pupil_dilation,
                    "left_eye_position_guide": eye_data.left_eye_position_guide,
                    "right_eye_position_guide": eye_data.right_eye_position_guide,
                    "foveated_gaze_direction": eye_data.foveated_gaze_direction,
                    "foveated_gaze_tracking_state": eye_data.foveated_gaze_tracking_state
                }
            });
            
            let _ = writeln!(json_file, "{}", log_entry);
            let _ = json_file.flush();
        }
        
        // Log CSV format - comprehensive eye tracking data
        if let Ok(mut csv_file) = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&csv_path)
        {
            // Write header if new file
            if !csv_exists {
                let _ = writeln!(csv_file, "timestamp_ms,unix_timestamp,device_motion_key,timestamp_ns,left_status_flags,left_gaze_point_valid,left_gaze_vector_valid,left_eye_openness_valid,left_pupil_dilation_valid,right_status_flags,right_gaze_point_valid,right_gaze_vector_valid,right_eye_openness_valid,right_pupil_dilation_valid,combined_status_flags,combined_gaze_point_valid,combined_gaze_vector_valid,combined_eye_openness_valid,combined_pupil_dilation_valid,left_gaze_point_x,left_gaze_point_y,left_gaze_point_z,right_gaze_point_x,right_gaze_point_y,right_gaze_point_z,combined_gaze_point_x,combined_gaze_point_y,combined_gaze_point_z,left_gaze_vector_x,left_gaze_vector_y,left_gaze_vector_z,right_gaze_vector_x,right_gaze_vector_y,right_gaze_vector_z,combined_gaze_vector_x,combined_gaze_vector_y,combined_gaze_vector_z,left_eye_openness,right_eye_openness,left_pupil_dilation,right_pupil_dilation,left_position_guide_x,left_position_guide_y,left_position_guide_z,right_position_guide_x,right_position_guide_y,right_position_guide_z,foveated_gaze_direction_x,foveated_gaze_direction_y,foveated_gaze_direction_z,foveated_gaze_tracking_state");
            }
            
            let row = format!(
                "{},{},pico_eye_tracking_raw,{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
                timestamp_ms,
                unix_timestamp,
                eye_data.timestamp_ns,
                eye_data.left_eye_pose_status.status_flags,
                eye_data.left_eye_pose_status.gaze_point_valid,
                eye_data.left_eye_pose_status.gaze_vector_valid,
                eye_data.left_eye_pose_status.eye_openness_valid,
                eye_data.left_eye_pose_status.pupil_dilation_valid,
                eye_data.right_eye_pose_status.status_flags,
                eye_data.right_eye_pose_status.gaze_point_valid,
                eye_data.right_eye_pose_status.gaze_vector_valid,
                eye_data.right_eye_pose_status.eye_openness_valid,
                eye_data.right_eye_pose_status.pupil_dilation_valid,
                eye_data.combined_eye_pose_status.status_flags,
                eye_data.combined_eye_pose_status.gaze_point_valid,
                eye_data.combined_eye_pose_status.gaze_vector_valid,
                eye_data.combined_eye_pose_status.eye_openness_valid,
                eye_data.combined_eye_pose_status.pupil_dilation_valid,
                eye_data.left_eye_gaze_point[0],
                eye_data.left_eye_gaze_point[1],
                eye_data.left_eye_gaze_point[2],
                eye_data.right_eye_gaze_point[0],
                eye_data.right_eye_gaze_point[1],
                eye_data.right_eye_gaze_point[2],
                eye_data.combined_eye_gaze_point[0],
                eye_data.combined_eye_gaze_point[1],
                eye_data.combined_eye_gaze_point[2],
                eye_data.left_eye_gaze_vector[0],
                eye_data.left_eye_gaze_vector[1],
                eye_data.left_eye_gaze_vector[2],
                eye_data.right_eye_gaze_vector[0],
                eye_data.right_eye_gaze_vector[1],
                eye_data.right_eye_gaze_vector[2],
                eye_data.combined_eye_gaze_vector[0],
                eye_data.combined_eye_gaze_vector[1],
                eye_data.combined_eye_gaze_vector[2],
                eye_data.left_eye_openness,
                eye_data.right_eye_openness,
                eye_data.left_eye_pupil_dilation,
                eye_data.right_eye_pupil_dilation,
                eye_data.left_eye_position_guide[0],
                eye_data.left_eye_position_guide[1],
                eye_data.left_eye_position_guide[2],
                eye_data.right_eye_position_guide[0],
                eye_data.right_eye_position_guide[1],
                eye_data.right_eye_position_guide[2],
                eye_data.foveated_gaze_direction[0],
                eye_data.foveated_gaze_direction[1],
                eye_data.foveated_gaze_direction[2],
                eye_data.foveated_gaze_tracking_state
            );
            
            let _ = writeln!(csv_file, "{}", row);
            let _ = csv_file.flush();
        }
    }
}

#[derive(Debug)]
pub enum HandType {
    Left = 0,
    Right = 1,
}

// todo: Move this struct to Settings and use it for every tracked device
#[derive(Default)]
struct MotionConfig {
    // Position offset applied after rotation offset
    pose_offset: Pose,
    linear_velocity_cutoff: f32,
    angular_velocity_cutoff: f32,
}

pub struct TrackingManager {
    last_head_pose: Pose,             // client's reference space
    inverse_recentering_origin: Pose, // client's reference space
    device_motions_history: HashMap<u64, VecDeque<(Duration, DeviceMotion)>>,
    hand_skeletons_history: [VecDeque<(Duration, [Pose; 26])>; 2],
    last_face_data: FaceData,
    last_pico_eye_data: Option<EyeTrackingDataLog>, // PICO eye tracking state cache
    last_pico_timestamp: Option<u64>, // Track last PICO timestamp to avoid duplicates
    max_history_size: usize,
}

impl TrackingManager {
    pub fn new(max_history_size: usize) -> TrackingManager {
        TrackingManager {
            last_head_pose: Pose::default(),
            inverse_recentering_origin: Pose::default(),
            device_motions_history: HashMap::new(),
            hand_skeletons_history: [VecDeque::new(), VecDeque::new()],
            last_face_data: FaceData::default(),
            last_pico_eye_data: None,
            last_pico_timestamp: None,
            max_history_size,
        }
    }

    pub fn recenter(
        &mut self,
        position_recentering_mode: PositionRecenteringMode,
        rotation_recentering_mode: RotationRecenteringMode,
    ) {
        let position = match position_recentering_mode {
            PositionRecenteringMode::Disabled => Vec3::ZERO,
            PositionRecenteringMode::LocalFloor => {
                let mut pos = self.last_head_pose.position;
                pos.y = 0.0;

                pos
            }
            PositionRecenteringMode::Local { view_height } => {
                self.last_head_pose.position
                    - self.last_head_pose.orientation * Vec3::new(0.0, view_height, 0.0)
            }
        };

        let orientation = match rotation_recentering_mode {
            RotationRecenteringMode::Disabled => Quat::IDENTITY,
            RotationRecenteringMode::Yaw => {
                let mut rot = self.last_head_pose.orientation;
                // extract yaw rotation
                rot.x = 0.0;
                rot.z = 0.0;
                rot = rot.normalize();

                rot
            }
            RotationRecenteringMode::Tilted => self.last_head_pose.orientation,
        };

        self.inverse_recentering_origin = Pose {
            position,
            orientation,
        }
        .inverse();
    }

    pub fn recenter_pose(&self, pose: Pose) -> Pose {
        self.inverse_recentering_origin * pose
    }

    pub fn recenter_motion(&self, motion: DeviceMotion) -> DeviceMotion {
        self.inverse_recentering_origin * motion
    }

    // Performs all kinds of tracking transformations, driven by settings.
    pub fn report_device_motions(
        &mut self,
        headset_config: &HeadsetConfig,
        timestamp: Duration,
        device_motions: &[(u64, DeviceMotion)],
    ) {
        let mut device_motion_configs = HashMap::new();
        device_motion_configs.insert(*HEAD_ID, MotionConfig::default());
        device_motion_configs.extend([
            (*BODY_CHEST_ID, MotionConfig::default()),
            (*BODY_HIPS_ID, MotionConfig::default()),
            (*BODY_LEFT_ELBOW_ID, MotionConfig::default()),
            (*BODY_RIGHT_ELBOW_ID, MotionConfig::default()),
            (*BODY_LEFT_KNEE_ID, MotionConfig::default()),
            (*BODY_LEFT_FOOT_ID, MotionConfig::default()),
            (*BODY_RIGHT_KNEE_ID, MotionConfig::default()),
            (*BODY_RIGHT_FOOT_ID, MotionConfig::default()),
        ]);

        if let Switch::Enabled(controllers) = &headset_config.controllers {
            let t = controllers.left_controller_position_offset;
            let r = controllers.left_controller_rotation_offset;

            device_motion_configs.insert(
                *HAND_LEFT_ID,
                MotionConfig {
                    pose_offset: Pose {
                        orientation: Quat::from_euler(
                            EulerRot::XYZ,
                            r[0] * DEG_TO_RAD,
                            r[1] * DEG_TO_RAD,
                            r[2] * DEG_TO_RAD,
                        ),
                        position: Vec3::new(t[0], t[1], t[2]),
                    },
                    linear_velocity_cutoff: controllers.linear_velocity_cutoff,
                    angular_velocity_cutoff: controllers.angular_velocity_cutoff * DEG_TO_RAD,
                },
            );

            device_motion_configs.insert(
                *HAND_RIGHT_ID,
                MotionConfig {
                    pose_offset: Pose {
                        orientation: Quat::from_euler(
                            EulerRot::XYZ,
                            r[0] * DEG_TO_RAD,
                            -r[1] * DEG_TO_RAD,
                            -r[2] * DEG_TO_RAD,
                        ),
                        position: Vec3::new(-t[0], t[1], t[2]),
                    },
                    linear_velocity_cutoff: controllers.linear_velocity_cutoff,
                    angular_velocity_cutoff: controllers.angular_velocity_cutoff * DEG_TO_RAD,
                },
            );
        }

        for &(device_id, mut motion) in device_motions {
            if device_id == *HEAD_ID {
                self.last_head_pose = motion.pose;
            }

            if let Some(config) = device_motion_configs.get(&device_id) {
                // Recenter
                motion = self.recenter_motion(motion);

                // Apply custom transform
                motion.pose.orientation *= config.pose_offset.orientation;
                motion.pose.position += motion.pose.orientation * config.pose_offset.position;

                motion.linear_velocity += motion
                    .angular_velocity
                    .cross(motion.pose.orientation * config.pose_offset.position);
                motion.angular_velocity =
                    motion.pose.orientation.conjugate() * motion.angular_velocity;

                fn cutoff(v: Vec3, threshold: f32) -> Vec3 {
                    if v.length_squared() > threshold * threshold {
                        v
                    } else {
                        Vec3::ZERO
                    }
                }

                motion.linear_velocity =
                    cutoff(motion.linear_velocity, config.linear_velocity_cutoff);
                motion.angular_velocity =
                    cutoff(motion.angular_velocity, config.angular_velocity_cutoff);
            }

            if let Some(motions) = self.device_motions_history.get_mut(&device_id) {
                motions.push_front((timestamp, motion));

                if motions.len() > self.max_history_size {
                    motions.pop_back();
                }
            } else {
                self.device_motions_history
                    .insert(device_id, VecDeque::from(vec![(timestamp, motion)]));
            }
        }
    }

    // If the exact sample_timestamp is not found, use the closest one if it's not older. This makes
    // sure that we return None if there is no newer sample and always return Some otherwise.
    pub fn get_device_motion(
        &self,
        device_id: u64,
        sample_timestamp: Duration,
    ) -> Option<DeviceMotion> {
        self.device_motions_history
            .get(&device_id)
            .and_then(|motions| {
                // Get first element to initialize a valid motion reference
                if let Some((_, motion)) = motions.front() {
                    let mut best_timestamp_diff = Duration::MAX;
                    let mut best_motion_ref = motion;

                    // Note: we are iterating from most recent to oldest
                    for (ts, m) in motions {
                        match ts.cmp(&sample_timestamp) {
                            Ordering::Equal => return Some(*m),
                            Ordering::Greater => {
                                let diff = ts.saturating_sub(sample_timestamp);
                                if diff < best_timestamp_diff {
                                    best_timestamp_diff = diff;
                                    best_motion_ref = m;
                                }
                            }
                            Ordering::Less => continue,
                        }
                    }

                    (best_timestamp_diff != Duration::MAX).then_some(*best_motion_ref)
                } else {
                    None
                }
            })
    }

    pub fn report_hand_skeleton(
        &mut self,
        hand_type: HandType,
        timestamp: Duration,
        mut skeleton: [Pose; 26],
    ) {
        for pose in &mut skeleton {
            *pose = self.recenter_pose(*pose);
        }

        let skeleton_history = &mut self.hand_skeletons_history[hand_type as usize];

        skeleton_history.push_back((timestamp, skeleton));

        if skeleton_history.len() > self.max_history_size {
            skeleton_history.pop_front();
        }
    }

    pub fn get_hand_skeleton(
        &self,
        hand_type: HandType,
        sample_timestamp: Duration,
    ) -> Option<&[Pose; 26]> {
        self.hand_skeletons_history[hand_type as usize]
            .iter()
            .find(|(timestamp, _)| *timestamp == sample_timestamp)
            .map(|(_, skeleton)| skeleton)
    }

    pub fn report_face_data(&mut self, face_data: FaceData) {
        // PICO eye tracking data is already in head local space, no coordinate transformation needed
        self.last_face_data = face_data;
    }

    pub fn get_face_data(&self) -> &FaceData {
        &self.last_face_data
    }

    // Convert EyeTrackingDataPICO to EyeTrackingDataLog format (following face_data pattern)
    fn convert_pico_eye_data(&self, eye_tracking_data: &EyeTrackingDataPICO) -> EyeTrackingDataLog {
        let left_pose_status = EyePoseStatusLog {
            status_flags: eye_tracking_data.left_eye_pose_status,
            gaze_point_valid: alvr_packets::eye_pose_status_pico::is_gaze_point_valid(eye_tracking_data.left_eye_pose_status),
            gaze_vector_valid: alvr_packets::eye_pose_status_pico::is_gaze_vector_valid(eye_tracking_data.left_eye_pose_status),
            eye_openness_valid: alvr_packets::eye_pose_status_pico::is_eye_openness_valid(eye_tracking_data.left_eye_pose_status),
            pupil_dilation_valid: alvr_packets::eye_pose_status_pico::is_pupil_dilation_valid(eye_tracking_data.left_eye_pose_status),
        };
        let right_pose_status = EyePoseStatusLog {
            status_flags: eye_tracking_data.right_eye_pose_status,
            gaze_point_valid: alvr_packets::eye_pose_status_pico::is_gaze_point_valid(eye_tracking_data.right_eye_pose_status),
            gaze_vector_valid: alvr_packets::eye_pose_status_pico::is_gaze_vector_valid(eye_tracking_data.right_eye_pose_status),
            eye_openness_valid: alvr_packets::eye_pose_status_pico::is_eye_openness_valid(eye_tracking_data.right_eye_pose_status),
            pupil_dilation_valid: alvr_packets::eye_pose_status_pico::is_pupil_dilation_valid(eye_tracking_data.right_eye_pose_status),
        };
        let combined_pose_status = EyePoseStatusLog {
            status_flags: eye_tracking_data.combined_eye_pose_status,
            gaze_point_valid: alvr_packets::eye_pose_status_pico::is_gaze_point_valid(eye_tracking_data.combined_eye_pose_status),
            gaze_vector_valid: alvr_packets::eye_pose_status_pico::is_gaze_vector_valid(eye_tracking_data.combined_eye_pose_status),
            eye_openness_valid: alvr_packets::eye_pose_status_pico::is_eye_openness_valid(eye_tracking_data.combined_eye_pose_status),
            pupil_dilation_valid: alvr_packets::eye_pose_status_pico::is_pupil_dilation_valid(eye_tracking_data.combined_eye_pose_status),
        };
        
        EyeTrackingDataLog {
            timestamp_ns: eye_tracking_data.time,
            left_eye_pose_status: left_pose_status,
            right_eye_pose_status: right_pose_status,
            combined_eye_pose_status: combined_pose_status,
            left_eye_gaze_point: eye_tracking_data.left_eye_gaze_point,
            right_eye_gaze_point: eye_tracking_data.right_eye_gaze_point,
            combined_eye_gaze_point: eye_tracking_data.combined_eye_gaze_point,
            left_eye_gaze_vector: eye_tracking_data.left_eye_gaze_vector,
            right_eye_gaze_vector: eye_tracking_data.right_eye_gaze_vector,
            combined_eye_gaze_vector: eye_tracking_data.combined_eye_gaze_vector,
            left_eye_openness: eye_tracking_data.left_eye_openness,
            right_eye_openness: eye_tracking_data.right_eye_openness,
            left_eye_pupil_dilation: eye_tracking_data.left_eye_pupil_dilation,
            right_eye_pupil_dilation: eye_tracking_data.right_eye_pupil_dilation,
            left_eye_position_guide: eye_tracking_data.left_eye_position_guide,
            right_eye_position_guide: eye_tracking_data.right_eye_position_guide,
            foveated_gaze_direction: eye_tracking_data.foveated_gaze_direction,
            foveated_gaze_tracking_state: eye_tracking_data.foveated_gaze_tracking_state,
        }
    }

    // Report PICO eye tracking data (following face_data pattern)
    pub fn report_pico_eye_data(&mut self, eye_tracking_data: EyeTrackingDataPICO) {
        // Check for duplicate data based on hardware timestamp
        if let Some(last_timestamp) = self.last_pico_timestamp {
            if eye_tracking_data.time == last_timestamp {
                // Skip duplicate data - hardware timestamp hasn't changed
                return;
            }
        }
        
        // Add diagnostic info for debugging data issues
        if eye_tracking_data.time == 0 {
            println!("Warning: PICO eye tracking timestamp is 0");
        }
        
        // Debug status flags
        println!("PICO Eye Tracking Debug:");
        println!("  Left: {}", alvr_packets::eye_pose_status_pico::get_flag_debug_info(eye_tracking_data.left_eye_pose_status));
        println!("  Right: {}", alvr_packets::eye_pose_status_pico::get_flag_debug_info(eye_tracking_data.right_eye_pose_status));
        println!("  Combined: {}", alvr_packets::eye_pose_status_pico::get_flag_debug_info(eye_tracking_data.combined_eye_pose_status));
        
        // Convert and store the data (only if it's new)
        let converted_data = self.convert_pico_eye_data(&eye_tracking_data);
        self.last_pico_eye_data = Some(converted_data);
        self.last_pico_timestamp = Some(eye_tracking_data.time);
    }

    // Get PICO eye tracking data (following face_data pattern)
    pub fn get_pico_eye_data(&self) -> Option<&EyeTrackingDataLog> {
        self.last_pico_eye_data.as_ref()
    }
}

pub fn tracking_loop(
    ctx: &ConnectionContext,
    initial_settings: Settings,
    multimodal_protocol: bool,
    hand_gesture_manager: Arc<Mutex<HandGestureManager>>,
    mut tracking_receiver: StreamReceiver<Tracking>,
    is_streaming: impl Fn() -> bool,
) {
    let mut gestures_button_mapping_manager =
        initial_settings
            .headset
            .controllers
            .as_option()
            .map(|config| {
                ButtonMappingManager::new_automatic(
                    &HAND_GESTURE_BUTTON_SET,
                    &config.emulation_mode,
                    &config.button_mapping_config,
                )
            });

    let mut face_tracking_sink = initial_settings
        .headset
        .face_tracking
        .into_option()
        .and_then(|config| {
            FaceTrackingSink::new(config.sink, initial_settings.connection.osc_local_port).ok()
        });

    let mut body_tracking_sink = initial_settings
        .headset
        .body_tracking
        .into_option()
        .and_then(|config| {
            BodyTrackingSink::new(config.sink, initial_settings.connection.osc_local_port).ok()
        });

    let mut vmc_sink = initial_settings
        .headset
        .vmc
        .into_option()
        .and_then(|config| VMCSink::new(config).ok());

    while is_streaming() {
        let data = match tracking_receiver.recv(STREAMING_RECV_TIMEOUT) {
            Ok(tracking) => tracking,
            Err(ConnectionError::TryAgain(_)) => continue,
            Err(ConnectionError::Other(_)) => return,
        };
        let Ok(mut tracking) = data.get_header() else {
            return;
        };

        let timestamp = tracking.target_timestamp;

        if let Some(stats) = &mut *ctx.statistics_manager.write() {
            stats.report_tracking_received(timestamp);
        }

        if !multimodal_protocol {
            if tracking.hand_skeletons[0].is_some() {
                tracking
                    .device_motions
                    .retain(|(id, _)| *id != *HAND_LEFT_ID);
            }

            if tracking.hand_skeletons[1].is_some() {
                tracking
                    .device_motions
                    .retain(|(id, _)| *id != *HAND_RIGHT_ID);
            }
        }

        let controllers_config = {
            let data_lock = SESSION_MANAGER.read();
            data_lock
                .settings()
                .headset
                .controllers
                .clone()
                .into_option()
        };

        let device_motion_keys = {
            let mut tracking_manager_lock = ctx.tracking_manager.write();
            let session_manager_lock = SESSION_MANAGER.read();
            let headset_config = &session_manager_lock.settings().headset;

            let device_motion_keys = tracking
                .device_motions
                .iter()
                .map(|(id, _)| *id)
                .collect::<Vec<_>>();

            tracking_manager_lock.report_device_motions(
                headset_config,
                timestamp,
                &tracking.device_motions,
            );

            if let Some(skeleton) = tracking.hand_skeletons[0] {
                tracking_manager_lock.report_hand_skeleton(HandType::Left, timestamp, skeleton);
            }
            if let Some(skeleton) = tracking.hand_skeletons[1] {
                tracking_manager_lock.report_hand_skeleton(HandType::Right, timestamp, skeleton);
            }

            // DFR: Update eye-tracked foveated rendering parameters before moving data
            DFR_RENDERER.update_from_face_data(&tracking.face_data);
            
            //.report_face_data()的作用是将tracking获取的FaceData整体映射后赋给last_face_data<FaceData>，我认为映射方式有误。
            tracking_manager_lock.report_face_data(tracking.face_data);
            
            // Process PICO eye tracking data (following face_data pattern)  
            let eye_data_to_report = {
                let face_data = tracking_manager_lock.get_face_data();
                face_data.pico_eye_tracking_data.clone()
            };
            if let Some(eye_data) = eye_data_to_report {
                tracking_manager_lock.report_pico_eye_data(eye_data);
            }
            
            // Log primary gaze data (from eye_tracker, preferred method)
            let face_data = tracking_manager_lock.get_face_data();
            log_gaze_data_to_file(face_data, timestamp);
            
            
            // Log PICO eye tracking data using unified interface
            if let Some(eye_data) = tracking_manager_lock.get_pico_eye_data() {
                log_eye_tracking_data_to_file(eye_data, timestamp);
            }
            
            if let Some(sink) = &mut face_tracking_sink {
                sink.send_tracking(tracking_manager_lock.get_face_data().clone());
            }

            if session_manager_lock.settings().extra.logging.log_tracking {
                let face_data = tracking_manager_lock.get_face_data().clone();

                let device_motions = device_motion_keys
                    .iter()
                    .filter_map(move |id| {
                        Some((
                            (*DEVICE_ID_TO_PATH.get(id)?).into(),
                            tracking_manager_lock
                                .get_device_motion(*id, timestamp)
                                .unwrap(),
                        ))
                    })
                    .collect::<Vec<(String, DeviceMotion)>>();

                alvr_events::send_event(EventType::Tracking(Box::new(TrackingEvent {
                    device_motions,
                    hand_skeletons: tracking.hand_skeletons,
                    eye_gazes: face_data.eye_gazes,
                    fb_face_expression: face_data.fb_face_expression,
                    htc_eye_expression: face_data.htc_eye_expression,
                    htc_lip_expression: face_data.htc_lip_expression,
                    pico_eye_tracking_data: face_data.pico_eye_tracking_data.clone(), // PICO eye tracking data
                })))
            }

            device_motion_keys
        };

        // Handle hand gestures
        if let (Some(gestures_config), Some(gestures_button_mapping_manager)) = (
            controllers_config
                .as_ref()
                .and_then(|c| c.hand_tracking_interaction.as_option()),
            &mut gestures_button_mapping_manager,
        ) {
            let mut hand_gesture_manager_lock = hand_gesture_manager.lock();

            if !device_motion_keys.contains(&*HAND_LEFT_ID) {
                if let Some(hand_skeleton) = tracking.hand_skeletons[0] {
                    ctx.events_sender
                        .send(ServerCoreEvent::Buttons(
                            hand_gestures::trigger_hand_gesture_actions(
                                gestures_button_mapping_manager,
                                *HAND_LEFT_ID,
                                &hand_gesture_manager_lock.get_active_gestures(
                                    &hand_skeleton,
                                    gestures_config,
                                    *HAND_LEFT_ID,
                                ),
                                gestures_config.only_touch,
                            ),
                        ))
                        .ok();
                }
            }
            if !device_motion_keys.contains(&*HAND_RIGHT_ID) {
                if let Some(hand_skeleton) = tracking.hand_skeletons[1] {
                    ctx.events_sender
                        .send(ServerCoreEvent::Buttons(
                            hand_gestures::trigger_hand_gesture_actions(
                                gestures_button_mapping_manager,
                                *HAND_RIGHT_ID,
                                &hand_gesture_manager_lock.get_active_gestures(
                                    &hand_skeleton,
                                    gestures_config,
                                    *HAND_RIGHT_ID,
                                ),
                                gestures_config.only_touch,
                            ),
                        ))
                        .ok();
                }
            }
        }

        ctx.events_sender
            .send(ServerCoreEvent::Tracking {
                sample_timestamp: tracking.target_timestamp,
            })
            .ok();

        let publish_vmc = matches!(
            SESSION_MANAGER.read().settings().headset.vmc,
            Switch::Enabled(VMCConfig { publish: true, .. })
        );
        if publish_vmc {
            let orientation_correction = matches!(
                SESSION_MANAGER.read().settings().headset.vmc,
                Switch::Enabled(VMCConfig {
                    orientation_correction: true,
                    ..
                })
            );

            if let Some(sink) = &mut vmc_sink {
                let tracking_manager_lock = ctx.tracking_manager.read();
                let device_motions = device_motion_keys
                    .iter()
                    .map(move |id| {
                        (
                            *id,
                            tracking_manager_lock
                                .get_device_motion(*id, timestamp)
                                .unwrap(),
                        )
                    })
                    .collect::<Vec<(u64, DeviceMotion)>>();

                if let Some(skeleton) = tracking.hand_skeletons[0] {
                    sink.send_hand_tracking(HandType::Left, &skeleton, orientation_correction);
                }
                if let Some(skeleton) = tracking.hand_skeletons[1] {
                    sink.send_hand_tracking(HandType::Right, &skeleton, orientation_correction);
                }
                sink.send_tracking(&device_motions, orientation_correction);
            }
        }

        let track_body = matches!(
            SESSION_MANAGER.read().settings().headset.body_tracking,
            Switch::Enabled(BodyTrackingConfig { tracked: true, .. })
        );
        if track_body {
            if let Some(sink) = &mut body_tracking_sink {
                let tracking_manager_lock = ctx.tracking_manager.read();
                let device_motions = device_motion_keys
                    .iter()
                    .map(move |id| {
                        (
                            *id,
                            tracking_manager_lock
                                .get_device_motion(*id, timestamp)
                                .unwrap(),
                        )
                    })
                    .collect::<Vec<_>>();
                sink.send_tracking(&device_motions);
            }
        }
    }
}
