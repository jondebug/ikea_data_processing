from collections import namedtuple


HAND_EYE_POST_PROC_FORMAT = namedtuple('hand_eye_post_proc_format',

                                       'frame_id ' 
                                       
                                       'eye_gaze_origin_x eye_gaze_origin_y eye_gaze_origin_z '
                                       #given by get_eye_gaze_point functin in project_hand_eye_to_pv.py file:
                                       'eye_gaze_point_x eye_gaze_point_y eye_gaze_point_z '
                                       
                                       #given by project_hand_eye_to_pv functin in project_hand_eye_to_pv.py file:
                                       'eye_gaze_rgb_projection_x eye_gaze_rgb_projection_y '
                                       
                                       #for all 52 joints (26 per hand:)
                                       'joint_point_x joint_point_y joint_point_z '
                                       'joint_rgb_projection_x joint_rgb_projection_y'
                                       )

EYE_POST_PROC_FORMAT = namedtuple('hand_eye_post_proc_format',

                                  'frame_id '

                                  'eye_gaze_origin_x eye_gaze_origin_y eye_gaze_origin_z '
                                  
                                  # given by get_eye_gaze_point functin in project_hand_eye_to_pv.py file:
                                  'eye_gaze_point_x eye_gaze_point_y eye_gaze_point_z '

                                  # given by project_hand_eye_to_pv functin in project_hand_eye_to_pv.py file:
                                  'eye_gaze_rgb_projection_x eye_gaze_rgb_projection_y ')