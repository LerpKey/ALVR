// Dynamic Foveated Rendering (DFR) implementation
// Converts eye tracking data from FaceData to foveation center shift parameters
// Supports per-eye tracking and falls back to a combined gaze model.

use alvr_common::glam::{Vec2, Vec3};
use alvr_packets::{EyeTrackingInputStatus, FaceData};
use std::collections::BTreeMap;
use std::sync::{
    atomic::{AtomicU32, Ordering},
    RwLock,
};
use std::time::{Duration, Instant, SystemTime};

// DFR shift parameters for C++ interop
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DFRShiftParams {
    pub shift_x: f32,      // Combined horizontal shift [-1, 1]
    pub shift_y: f32,      // Combined vertical shift [-1, 1]
    pub left_shift_x: f32, // Left eye shift
    pub left_shift_y: f32,
    pub right_shift_x: f32, // Right eye shift
    pub right_shift_y: f32,
    pub is_eye_tracked: bool, // Whether valid eye tracking data is available
}

impl Default for DFRShiftParams {
    fn default() -> Self {
        let center = static_ffr_center();
        Self {
            shift_x: center.x,
            shift_y: center.y,
            left_shift_x: center.x,
            left_shift_y: center.y,
            right_shift_x: center.x,
            right_shift_y: center.y,
            is_eye_tracked: false,
        }
    }
}

// Global state for DFR
static GLOBAL_DFR_SHIFT: RwLock<Option<DFRShiftParams>> = RwLock::new(None);
static GLOBAL_LAST_UPDATE: RwLock<Option<SystemTime>> = RwLock::new(None);
static STATIC_FFR_CENTER: std::sync::LazyLock<RwLock<Vec2>> =
    std::sync::LazyLock::new(|| RwLock::new(Vec2::ZERO));
static GLOBAL_EYE_TRACKING_STATE: RwLock<EyeTrackingInputStatus> =
    RwLock::new(EyeTrackingInputStatus::Unsupported);
static ACTIVE_LOG_COUNTER: AtomicU32 = AtomicU32::new(0);
static STANDBY_LOG_COUNTER: AtomicU32 = AtomicU32::new(0);
static FFR_LOG_COUNTER: AtomicU32 = AtomicU32::new(0);

// Synchronized DFR state for pipeline consistency
// Ensures server encoding and client inverse-FFR use identical shift values
static SYNCHRONIZED_DFR_SHIFT: RwLock<Option<DFRShiftParams>> = RwLock::new(None);
static SYNCHRONIZED_LAST_UPDATE: RwLock<Option<SystemTime>> = RwLock::new(None);
static SYNCHRONIZED_SEQUENCE_ID: RwLock<u64> = RwLock::new(0);

// Caches the actual eye shift data used by the FR shader for a given frame.
// This ensures the VideoPacketHeader corresponds exactly to the rendered frame.
#[derive(Debug, Clone, Copy)]
struct RenderedShiftData {
    shift: DFRShiftParams,
    sequence_id: u64,
    timestamp: Instant,
}

static LAST_FR_RENDERED_SHIFT: std::sync::LazyLock<RwLock<Option<RenderedShiftData>>> =
    std::sync::LazyLock::new(|| RwLock::new(None));

// Unified timestamp generator for Frame-Perfect binding.
// Ensures that DFR encoding and shift data use the identical SteamVR targetTimestampNs.
static TIMESTAMP_SHIFT_CACHE: std::sync::LazyLock<RwLock<BTreeMap<u64, DFRShiftParams>>> =
    std::sync::LazyLock::new(|| RwLock::new(BTreeMap::new()));

// DFR configuration constants
const EYE_TRACKING_TIMEOUT: Duration = Duration::from_millis(200);
const SHIFT_GAIN: f32 = 0.8; // Scale factor for eye shift
const MAX_SHIFT: f32 = 0.4; // Maximum allowed shift

// PICO hardware coordinate ranges - critical for accurate mapping
const PICO_MAX_X: f32 = 0.6; // x gaze_vector range: [-0.6, 0.6]
const PICO_MAX_Y: f32 = 0.4; // y gaze_vector range: [-0.4, 0.4]

pub struct DynamicFoveatedRenderer {
    // No additional fields needed - PICO gaze vectors are directly usable
}

impl Default for DynamicFoveatedRenderer {
    fn default() -> Self {
        Self {}
    }
}

impl DynamicFoveatedRenderer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Update DFR parameters from face tracking data with synchronized pipeline
    pub fn update_from_face_data(&self, face_data: &FaceData) {
        let now = SystemTime::now();
        let previous_state = get_eye_tracking_state();
        let mut eye_state = face_data.eye_tracking_state;
        let mut pending_shift = None;

        match eye_state {
            EyeTrackingInputStatus::Active => {
                pending_shift = self.calculate_dfr_shift(face_data);
            }
            EyeTrackingInputStatus::Standby => {}
            EyeTrackingInputStatus::Unsupported => {
                let inferred_shift = self.calculate_dfr_shift(face_data);
                if let Some(shift) = inferred_shift {
                    pending_shift = Some(shift);
                    eye_state = EyeTrackingInputStatus::Active;
                } else if self.has_any_gaze_payload(face_data) {
                    // Legacy clients send gaze samples without the new state flag.
                    eye_state = EyeTrackingInputStatus::Standby;
                }
            }
        }

        if previous_state != eye_state {
            set_eye_tracking_state(eye_state);
        }

        match eye_state {
            EyeTrackingInputStatus::Active => {
                if let Some(shift_params) = pending_shift {
                    self.record_active_shift(now, shift_params);
                } else {
                    alvr_common::warn!(
                        "DFR: Eye tracking marked active but produced no valid gaze vector"
                    );
                    self.log_standby_state();
                }
            }
            EyeTrackingInputStatus::Standby => {
                self.log_standby_state();
            }
            EyeTrackingInputStatus::Unsupported => {
                if previous_state != EyeTrackingInputStatus::Unsupported {
                    self.reset_dfr_state();
                }
            }
        }
    }

    fn has_any_gaze_payload(&self, face_data: &FaceData) -> bool {
        if face_data.eye_gazes.iter().any(|pose| pose.is_some()) {
            return true;
        }

        if let Some(pico_data) = &face_data.pico_eye_tracking_data {
            return alvr_packets::eye_pose_status_pico::is_gaze_vector_valid(
                pico_data.left_eye_pose_status,
            ) || alvr_packets::eye_pose_status_pico::is_gaze_vector_valid(
                pico_data.right_eye_pose_status,
            ) || alvr_packets::eye_pose_status_pico::is_gaze_vector_valid(
                pico_data.combined_eye_pose_status,
            );
        }

        false
    }

    fn record_active_shift(&self, now: SystemTime, shift_params: DFRShiftParams) {
        if let Ok(mut time_guard) = GLOBAL_LAST_UPDATE.write() {
            *time_guard = Some(now);
        }
        if let Ok(mut shift_guard) = GLOBAL_DFR_SHIFT.write() {
            *shift_guard = Some(shift_params);
        }

        // Update synchronized shift for pipeline consistency
        if self.should_update_synchronized_shift(&shift_params) {
            if let Ok(mut sync_shift_guard) = SYNCHRONIZED_DFR_SHIFT.write() {
                if let Ok(mut sync_time_guard) = SYNCHRONIZED_LAST_UPDATE.write() {
                    if let Ok(mut seq_id_guard) = SYNCHRONIZED_SEQUENCE_ID.write() {
                        *seq_id_guard += 1;
                        let sequence_id = *seq_id_guard;

                        *sync_shift_guard = Some(shift_params);
                        *sync_time_guard = Some(now);
                        alvr_common::debug!(
                            "Pipeline Sync: Updated synchronized shift [seq={}] - DFR:({:.3},{:.3})",
                            sequence_id,
                            shift_params.shift_x,
                            shift_params.shift_y
                        );
                    }
                }
            }
        }

        let frames = ACTIVE_LOG_COUNTER.fetch_add(1, Ordering::Relaxed) + 1;
        if frames % 60 == 0 {
            alvr_common::info!("DFR Active: Frame {} - Using eye tracking", frames);
        }
    }

    fn log_standby_state(&self) {
        let frames = STANDBY_LOG_COUNTER.fetch_add(1, Ordering::Relaxed) + 1;
        if frames % 120 == 0 {
            alvr_common::debug!(
                "DFR Standby: {} frames without gaze samples - keeping last known center",
                frames
            );
        }
    }

    fn reset_dfr_state(&self) {
        if let Ok(mut shift_guard) = GLOBAL_DFR_SHIFT.write() {
            *shift_guard = None;
        }
        if let Ok(mut time_guard) = GLOBAL_LAST_UPDATE.write() {
            *time_guard = None;
        }
        if let Ok(mut sync_shift_guard) = SYNCHRONIZED_DFR_SHIFT.write() {
            *sync_shift_guard = None;
        }
        if let Ok(mut sync_time_guard) = SYNCHRONIZED_LAST_UPDATE.write() {
            *sync_time_guard = None;
        }
        if let Ok(mut seq_id_guard) = SYNCHRONIZED_SEQUENCE_ID.write() {
            *seq_id_guard += 1;
        }

        let frames = FFR_LOG_COUNTER.fetch_add(1, Ordering::Relaxed) + 1;
        if frames % 300 == 0 {
            alvr_common::info!(
                "FFR Active: Frame {} - Eye tracking disabled or unsupported",
                frames
            );
        }
    }

    /// Determine if synchronized shift should be updated
    fn should_update_synchronized_shift(&self, new_shift: &DFRShiftParams) -> bool {
        // Check if we have existing synchronized data
        if let Ok(sync_guard) = SYNCHRONIZED_DFR_SHIFT.read() {
            if let Some(current_sync) = *sync_guard {
                // Compare eye tracking availability first
                if current_sync.is_eye_tracked != new_shift.is_eye_tracked {
                    return true; // Mode change: DFR ↔ FFR
                }

                if new_shift.is_eye_tracked {
                    // Compare shift values with threshold
                    const SHIFT_THRESHOLD: f32 = 0.02; // 2% threshold to avoid micro-movements

                    let x_diff = (current_sync.shift_x - new_shift.shift_x).abs();
                    let y_diff = (current_sync.shift_y - new_shift.shift_y).abs();

                    return x_diff > SHIFT_THRESHOLD || y_diff > SHIFT_THRESHOLD;
                }
            }
        }

        // No existing data or first update
        true
    }

    /// Calculate DFR shift using per-eye gaze when available; fallback to combined
    fn calculate_dfr_shift(&self, face_data: &FaceData) -> Option<DFRShiftParams> {
        if let Some(pico_data) = &face_data.pico_eye_tracking_data {
            let left_valid = alvr_packets::eye_pose_status_pico::is_gaze_vector_valid(
                pico_data.left_eye_pose_status,
            );
            let right_valid = alvr_packets::eye_pose_status_pico::is_gaze_vector_valid(
                pico_data.right_eye_pose_status,
            );
            let combined_valid = alvr_packets::eye_pose_status_pico::is_gaze_vector_valid(
                pico_data.combined_eye_pose_status,
            );

            let mut shifts = Vec::new();
            let mut left_shift = None;
            let mut right_shift = None;

            if left_valid {
                let dir = Vec3::new(
                    pico_data.left_eye_gaze_vector[0],
                    pico_data.left_eye_gaze_vector[1],
                    pico_data.left_eye_gaze_vector[2],
                );
                if self.validate_gaze_direction(dir) {
                    let s = self.gaze_to_screen_shift(dir);
                    left_shift = Some(s);
                    shifts.push(s);
                }
            }

            if right_valid {
                let dir = Vec3::new(
                    pico_data.right_eye_gaze_vector[0],
                    pico_data.right_eye_gaze_vector[1],
                    pico_data.right_eye_gaze_vector[2],
                );
                if self.validate_gaze_direction(dir) {
                    let s = self.gaze_to_screen_shift(dir);
                    right_shift = Some(s);
                    shifts.push(s);
                }
            }

            if !shifts.is_empty() {
                let sum = shifts.iter().fold(Vec2::ZERO, |acc, v| acc + *v);
                let avg = sum / shifts.len() as f32;
                return Some(DFRShiftParams {
                    shift_x: avg.x,
                    shift_y: avg.y,
                    left_shift_x: left_shift.map_or(avg.x, |v| v.x),
                    left_shift_y: left_shift.map_or(avg.y, |v| v.y),
                    right_shift_x: right_shift.map_or(avg.x, |v| v.x),
                    right_shift_y: right_shift.map_or(avg.y, |v| v.y),
                    is_eye_tracked: true,
                });
            }

            if combined_valid {
                let dir = Vec3::new(
                    pico_data.combined_eye_gaze_vector[0],
                    pico_data.combined_eye_gaze_vector[1],
                    pico_data.combined_eye_gaze_vector[2],
                );
                if self.validate_gaze_direction(dir) {
                    let screen_shift = self.gaze_to_screen_shift(dir);
                    return Some(DFRShiftParams {
                        shift_x: screen_shift.x,
                        shift_y: screen_shift.y,
                        left_shift_x: screen_shift.x,
                        left_shift_y: screen_shift.y,
                        right_shift_x: screen_shift.x,
                        right_shift_y: screen_shift.y,
                        is_eye_tracked: true,
                    });
                }
            }
        }

        None
    }

    /// Validate that gaze direction is reasonable
    fn validate_gaze_direction(&self, gaze_direction: Vec3) -> bool {
        let length = gaze_direction.length();

        // Check for valid normalized vector (should be close to 1.0)
        if length < 0.5 || length > 2.0 {
            return false;
        }

        // Check for reasonable gaze ranges (PICO specific)
        if gaze_direction.x.abs() > PICO_MAX_X * 1.2 || gaze_direction.y.abs() > PICO_MAX_Y * 1.2 {
            return false;
        }

        true
    }

    /// Convert gaze direction to screen shift coordinates
    fn gaze_to_screen_shift(&self, gaze_direction: Vec3) -> Vec2 {
        // Direct mapping from PICO hardware ranges to screen coordinates
        // This ensures full eye movement range maps to full screen
        let screen_x = (gaze_direction.x / PICO_MAX_X).clamp(-1.0, 1.0);
        let screen_y = (gaze_direction.y / PICO_MAX_Y).clamp(-1.0, 1.0);

        // Apply gain and max shift limiting
        let shifted_x = (screen_x * SHIFT_GAIN).clamp(-MAX_SHIFT, MAX_SHIFT);
        let shifted_y = (screen_y * SHIFT_GAIN).clamp(-MAX_SHIFT, MAX_SHIFT);

        Vec2::new(shifted_x, shifted_y)
    }
}

/// Public interface for C++ integration.
/// Key modification: Saves the actual data used by the FR shader to ensure Frame-Perfect synchronization.
#[no_mangle]
pub extern "C" fn get_eye_tracked_ffr_shift() -> DFRShiftParams {
    let shift_data = current_or_fallback_shift();

    // Key: Save the actual data used by the FR shader.
    // This ensures the shift transmitted in the VideoPacketHeader matches the one used by the FR shader.
    if let Ok(mut rendered_shift_guard) = LAST_FR_RENDERED_SHIFT.write() {
        let current_seq_id = get_synchronized_sequence_id();
        *rendered_shift_guard = Some(RenderedShiftData {
            shift: shift_data,
            sequence_id: current_seq_id,
            timestamp: Instant::now(),
        });

        // Debug log: Verify data capture
        if shift_data.is_eye_tracked {
            alvr_common::debug!(
                "FR_CAPTURE: Captured eyeshift [seq={}] DFR:({:.3},{:.3}) for frame rendering",
                current_seq_id,
                shift_data.shift_x,
                shift_data.shift_y
            );
        } else {
            alvr_common::debug!(
                "FR_CAPTURE: Captured FFR default [seq={}] for frame rendering",
                current_seq_id
            );
        }
    }

    shift_data
}

/// Get synchronized DFR shift for pipeline consistency
pub fn get_synchronized_dfr_shift() -> Option<DFRShiftParams> {
    if let Ok(guard) = SYNCHRONIZED_DFR_SHIFT.read() {
        *guard
    } else {
        None
    }
}

/// Get synchronized sequence ID for tracking pipeline sync
pub fn get_synchronized_sequence_id() -> u64 {
    if let Ok(guard) = SYNCHRONIZED_SEQUENCE_ID.read() {
        *guard
    } else {
        0
    }
}

/// Check if eye tracking data is recent enough to be valid
pub fn is_eye_tracking_active() -> bool {
    match get_eye_tracking_state() {
        EyeTrackingInputStatus::Unsupported => false,
        EyeTrackingInputStatus::Standby => GLOBAL_DFR_SHIFT
            .read()
            .map(|shift| shift.is_some())
            .unwrap_or(false),
        EyeTrackingInputStatus::Active => {
            if let Ok(guard) = GLOBAL_LAST_UPDATE.read() {
                if let Some(last_update) = *guard {
                    return SystemTime::now()
                        .duration_since(last_update)
                        .map(|duration| duration < EYE_TRACKING_TIMEOUT)
                        .unwrap_or(false);
                }
            }
            false
        }
    }
}

/// Gets the actual eye shift data used by the FR shader.
/// Used for Frame-Perfect binding in the VideoPacketHeader.
pub fn get_last_fr_rendered_shift() -> Option<DFRShiftParams> {
    if let Ok(guard) = LAST_FR_RENDERED_SHIFT.read() {
        guard.as_ref().map(|data| data.shift)
    } else {
        None
    }
}

/// Gets the sequence number used by the FR shader.
/// Ensures temporal consistency.
pub fn get_last_fr_rendered_sequence_id() -> Option<u64> {
    if let Ok(guard) = LAST_FR_RENDERED_SHIFT.read() {
        guard.as_ref().map(|data| data.sequence_id)
    } else {
        None
    }
}

/// Verifies if the FR rendered data is recent (for edge case handling).
pub fn is_last_fr_rendered_shift_recent() -> bool {
    if let Ok(guard) = LAST_FR_RENDERED_SHIFT.read() {
        if let Some(data) = guard.as_ref() {
            // Check if the data is within a reasonable time window (e.g., 1 second)
            return data.timestamp.elapsed() < Duration::from_millis(1000);
        }
    }
    false
}

/// NEW: Frame-Perfect timestamp binding function.
/// Binds shift data to a precise SteamVR targetTimestampNs to eliminate jitter.
#[no_mangle]
pub extern "C" fn get_eye_tracked_ffr_shift_with_timestamp(
    target_timestamp_ns: u64,
) -> DFRShiftParams {
    // Edge case: If the timestamp is 0 or invalid, fall back to the current shift.
    if target_timestamp_ns == 0 {
        alvr_common::warn!("TIMESTAMP_BIND: Invalid timestamp 0, using fallback");
        return get_eye_tracked_ffr_shift(); // Use old interface as a fallback.
    }

    // Use precise timestamp binding.
    if let Ok(mut cache_guard) = TIMESTAMP_SHIFT_CACHE.write() {
        // Get the latest shift data.
        let current_shift = current_or_fallback_shift();

        // Bind the shift data to the precise SteamVR timestamp.
        cache_guard.insert(target_timestamp_ns, current_shift);

        // Clean up expired data (keep the last 100 frames to prevent infinite growth).
        if cache_guard.len() > 100 {
            let keys_to_remove: Vec<u64> = cache_guard
                .keys()
                .take(cache_guard.len() - 100)
                .cloned()
                .collect();
            for key in keys_to_remove {
                cache_guard.remove(&key);
            }
        }

        // Debug log: Verify data binding.
        if current_shift.is_eye_tracked {
            alvr_common::debug!(
                "TIMESTAMP_BIND: Cached shift for timestamp {} - DFR:({:.3},{:.3})",
                target_timestamp_ns,
                current_shift.shift_x,
                current_shift.shift_y
            );
        } else {
            alvr_common::debug!(
                "TIMESTAMP_BIND: Cached FFR default for timestamp {}",
                target_timestamp_ns
            );
        }

        return current_shift;
    }

    // Write lock failed: return default.
    alvr_common::error!("TIMESTAMP_BIND: Failed to acquire cache write lock");
    DFRShiftParams::default()
}

/// Frame-Perfect retrieval for VideoPacketHeader.
/// Retrieves the corresponding shift data based on the precise targetTimestampNs.
pub fn get_dfr_shift_for_timestamp(target_timestamp_ns: u64) -> Option<DFRShiftParams> {
    if let Ok(cache_guard) = TIMESTAMP_SHIFT_CACHE.read() {
        if let Some(cached_shift) = cache_guard.get(&target_timestamp_ns) {
            alvr_common::debug!(
                "TIMESTAMP_RETRIEVE: Found cached shift for timestamp {} - DFR:({:.3},{:.3})",
                target_timestamp_ns,
                cached_shift.shift_x,
                cached_shift.shift_y
            );
            return Some(*cached_shift);
        } else {
            alvr_common::warn!(
                "TIMESTAMP_RETRIEVE: Timestamp {} not found in cache (cache size: {})",
                target_timestamp_ns,
                cache_guard.len()
            );
        }
    } else {
        alvr_common::error!("TIMESTAMP_RETRIEVE: Failed to acquire cache read lock");
    }

    None
}

/// Check timestamp cache status (for debugging).
pub fn get_timestamp_cache_info() -> (usize, Option<u64>, Option<u64>) {
    if let Ok(cache_guard) = TIMESTAMP_SHIFT_CACHE.read() {
        let size = cache_guard.len();
        let oldest = cache_guard.keys().next().cloned();
        let newest = cache_guard.keys().next_back().cloned();
        (size, oldest, newest)
    } else {
        (0, None, None)
    }
}

/// Initialize DFR renderer
pub fn initialize_dfr() -> DynamicFoveatedRenderer {
    DynamicFoveatedRenderer::new()
}

pub fn set_static_ffr_center(center: Vec2) {
    if let Ok(mut guard) = STATIC_FFR_CENTER.write() {
        *guard = center;
    }
}

fn static_ffr_center() -> Vec2 {
    STATIC_FFR_CENTER
        .read()
        .map(|value| *value)
        .unwrap_or(Vec2::ZERO)
}

fn set_eye_tracking_state(state: EyeTrackingInputStatus) {
    if let Ok(mut guard) = GLOBAL_EYE_TRACKING_STATE.write() {
        if *guard != state {
            alvr_common::info!(
                "Eye tracking input state changed: {:?} -> {:?}",
                *guard,
                state
            );
        }
        *guard = state;
    }
}

fn get_eye_tracking_state() -> EyeTrackingInputStatus {
    GLOBAL_EYE_TRACKING_STATE
        .read()
        .map(|state| *state)
        .unwrap_or(EyeTrackingInputStatus::Unsupported)
}

fn current_or_fallback_shift() -> DFRShiftParams {
    let now = SystemTime::now();

    if let (Ok(shift_guard), Ok(time_guard)) = (GLOBAL_DFR_SHIFT.read(), GLOBAL_LAST_UPDATE.read())
    {
        if let (Some(shift), Some(last_update)) = (*shift_guard, *time_guard) {
            if now
                .duration_since(last_update)
                .map(|elapsed| elapsed < EYE_TRACKING_TIMEOUT)
                .unwrap_or(false)
            {
                return shift;
            }
        }
    }

    if matches!(get_eye_tracking_state(), EyeTrackingInputStatus::Standby) {
        if let Ok(standby_guard) = GLOBAL_DFR_SHIFT.read() {
            if let Some(shift) = *standby_guard {
                return shift;
            }
        }
    }

    DFRShiftParams::default()
}
