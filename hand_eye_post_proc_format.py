from collections import namedtuple


HAND_EYE_POST_PROC_FORMAT = namedtuple('hand_eye_post_proc_format',
                                       'frame_id ' 
                                       
                                       #given by get_eye_gaze_point functin in project_hand_eye_to_pv.py file:
                                       'eye_gaze_point_x eye_gaze_point_y eye_gaze_point_z '
                                       
                                       #given by project_hand_eye_to_pv functin in project_hand_eye_to_pv.py file:
                                       'eye_gaze_rgb_projection_x eye_gaze_rgb_projection_y '
                                       #need to add origin
                                       
                                       #for all 52 joints (26 per hand:)
                                       'joint_point_x joint_point_y joint_point_z '
                                       'joint_rgb_projection_x joint_rgb_projection_y'
                                       )
