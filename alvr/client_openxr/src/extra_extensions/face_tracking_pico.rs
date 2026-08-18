use crate::extra_extensions::get_instance_proc;
use alvr_packets::EyeTrackingDataPICO;
use openxr::{self as xr, sys};

const TRACKING_MODE_FACE_BIT: u64 = 0x00000008;
const TRACKING_MODE_EYE_BIT: u64 = 0x00000004;
const PICO_FACE_EXPRESSION_COUNT: usize = 52;

#[repr(C)]
struct FaceTrackingDataPICO {
    time: u64, // sys::Time as u64
    blend_shape_weight: [f32; 72],
    is_video_input_valid: [f32; 10],
    laughing_probability: f32,
    emotion_probability: [f32; 10],
    reserved: [f32; 128],
}

type StartEyeTrackingPICO = unsafe extern "system" fn(sys::Session) -> sys::Result;

type StopEyeTrackingPICO = unsafe extern "system" fn(sys::Session, u64) -> sys::Result;

type SetTrackingModePICO = unsafe extern "system" fn(sys::Session, u64) -> sys::Result;

type GetFaceTrackingDataPICO = unsafe extern "system" fn(
    sys::Session,
    sys::Time,
    i32,
    *mut FaceTrackingDataPICO,
) -> sys::Result;

type GetEyeTrackingDataPICO =
    unsafe extern "system" fn(sys::Session, sys::Time, *mut EyeTrackingDataPICO) -> sys::Result;

pub struct FaceTrackerPico {
    session: xr::Session<xr::AnyGraphics>,
    start_eye_tracking: StartEyeTrackingPICO,
    stop_eye_tracking: StopEyeTrackingPICO,
    set_tracking_mode: SetTrackingModePICO,
    get_face_tracking_data: GetFaceTrackingDataPICO,
    get_eye_tracking_data: GetEyeTrackingDataPICO,
}

impl FaceTrackerPico {
    pub fn new<G>(session: xr::Session<G>) -> xr::Result<Self> {
        // PICO has its own eye tracking API that doesn't require ext_eye_gaze_interaction
        // We'll try to get the function pointers directly

        // Check if eye gaze interaction is available (optional for PICO)
        if let Some(_ext) = session.instance().exts().ext_eye_gaze_interaction {
            println!("OpenXR Eye Gaze Interaction extension is available");
        } else {
            println!(
                "OpenXR Eye Gaze Interaction extension not available, trying PICO-specific API"
            );
        }

        // Try to get PICO-specific function pointers
        let start_eye_tracking = match get_instance_proc(&session, "xrStartEyeTrackingPICO") {
            Ok(ptr) => {
                println!("Successfully loaded xrStartEyeTrackingPICO");
                ptr
            }
            Err(e) => {
                println!("Failed to load xrStartEyeTrackingPICO: {:?}", e);
                return Err(e);
            }
        };

        let stop_eye_tracking = get_instance_proc(&session, "xrStopEyeTrackingPICO")?;
        let set_tracking_mode = get_instance_proc(&session, "xrSetTrackingModePICO")?;
        let get_face_tracking_data = get_instance_proc(&session, "xrGetFaceTrackingDataPICO")?;
        let get_eye_tracking_data = match get_instance_proc(&session, "xrGetEyeTrackingDataPICO") {
            Ok(ptr) => {
                println!("Successfully loaded xrGetEyeTrackingDataPICO");
                ptr
            }
            Err(e) => {
                println!("Failed to load xrGetEyeTrackingDataPICO: {:?}", e);
                return Err(e);
            }
        };

        Ok(Self {
            session: session.into_any_graphics(),
            start_eye_tracking,
            stop_eye_tracking,
            set_tracking_mode,
            get_face_tracking_data,
            get_eye_tracking_data,
        })
    }

    pub fn get_face_tracking_data(&self, time: xr::Time) -> xr::Result<Option<Vec<f32>>> {
        let mut face_tracking_data = FaceTrackingDataPICO {
            time: 0, // Will be set to actual time from OpenXR
            blend_shape_weight: [0.0; 72],
            is_video_input_valid: [0.0; 10],
            laughing_probability: 0.0,
            emotion_probability: [0.0; 10],
            reserved: [0.0; 128],
        };

        unsafe {
            super::xr_res((self.get_face_tracking_data)(
                self.session.as_raw(),
                time,
                0,
                &mut face_tracking_data,
            ))?;

            if face_tracking_data.time != 0 {
                let blend_shape_slice =
                    face_tracking_data.blend_shape_weight[..PICO_FACE_EXPRESSION_COUNT].to_vec();

                Ok(Some(blend_shape_slice))
            } else {
                Ok(None)
            }
        }
    }

    pub fn get_eye_tracking_data(&self, time: xr::Time) -> xr::Result<Option<EyeTrackingDataPICO>> {
        let mut eye_tracking_data = EyeTrackingDataPICO {
            time: 0, // Will be set to actual time from OpenXR
            left_eye_pose_status: 0,
            right_eye_pose_status: 0,
            combined_eye_pose_status: 0,
            left_eye_gaze_point: [0.0; 3],
            right_eye_gaze_point: [0.0; 3],
            combined_eye_gaze_point: [0.0; 3],
            left_eye_gaze_vector: [0.0; 3],
            right_eye_gaze_vector: [0.0; 3],
            combined_eye_gaze_vector: [0.0; 3],
            left_eye_openness: 0.0,
            right_eye_openness: 0.0,
            left_eye_pupil_dilation: 0.0,
            right_eye_pupil_dilation: 0.0,
            left_eye_position_guide: [0.0; 3],
            right_eye_position_guide: [0.0; 3],
            foveated_gaze_direction: [0.0; 3],
            foveated_gaze_tracking_state: 0,
        };

        unsafe {
            super::xr_res((self.get_eye_tracking_data)(
                self.session.as_raw(),
                time,
                &mut eye_tracking_data,
            ))?;

            if eye_tracking_data.time != 0 {
                // Add diagnostic logging for data validation
                static mut LAST_TIME: u64 = 0;
                static mut CALL_COUNT: u32 = 0;

                CALL_COUNT += 1;

                if eye_tracking_data.time == LAST_TIME {
                    if CALL_COUNT % 60 == 0 {
                        // Log every 60 calls to avoid spam
                        println!(
                            "Warning: PICO eye tracking timestamp not changing: {} (call #{})",
                            eye_tracking_data.time, CALL_COUNT
                        );
                        println!(
                            "  Left status: {:012b} ({})",
                            eye_tracking_data.left_eye_pose_status,
                            eye_tracking_data.left_eye_pose_status
                        );
                        println!(
                            "  Right status: {:012b} ({})",
                            eye_tracking_data.right_eye_pose_status,
                            eye_tracking_data.right_eye_pose_status
                        );
                    }
                } else {
                    println!(
                        "PICO eye tracking timestamp changed: {} -> {} (call #{})",
                        LAST_TIME, eye_tracking_data.time, CALL_COUNT
                    );
                }
                LAST_TIME = eye_tracking_data.time;

                Ok(Some(eye_tracking_data))
            } else {
                println!("PICO eye tracking data has zero timestamp, skipping");
                Ok(None)
            }
        }
    }

    pub fn start_face_tracking(&self) -> xr::Result<()> {
        unsafe {
            super::xr_res((self.start_eye_tracking)(self.session.as_raw()))?;
            // Enable both face and eye tracking modes
            super::xr_res((self.set_tracking_mode)(
                self.session.as_raw(),
                TRACKING_MODE_FACE_BIT | TRACKING_MODE_EYE_BIT,
            ))
        }
    }

    pub fn start_eye_tracking(&self) -> xr::Result<()> {
        unsafe {
            println!("Starting PICO eye tracking...");

            // First start eye tracking
            match super::xr_res((self.start_eye_tracking)(self.session.as_raw())) {
                Ok(_) => println!("✅ Successfully called xrStartEyeTrackingPICO"),
                Err(e) => {
                    println!("❌ Failed to call xrStartEyeTrackingPICO: {:?}", e);
                    return Err(e);
                }
            }

            // Then set the tracking mode
            match super::xr_res((self.set_tracking_mode)(
                self.session.as_raw(),
                TRACKING_MODE_EYE_BIT,
            )) {
                Ok(_) => {
                    println!(
                        "✅ Successfully set eye tracking mode ({})",
                        TRACKING_MODE_EYE_BIT
                    );
                    Ok(())
                }
                Err(e) => {
                    println!("❌ Failed to set eye tracking mode: {:?}", e);
                    Err(e)
                }
            }
        }
    }

    pub fn stop_face_tracking(&self) -> xr::Result<()> {
        unsafe {
            super::xr_res((self.stop_eye_tracking)(
                self.session.as_raw(),
                TRACKING_MODE_FACE_BIT,
            ))
        }
    }
}
