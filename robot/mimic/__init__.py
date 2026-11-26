"""Copy a person's arm/leg movements seen by a camera (tools/mimic.py).

pose.py      camera frame -> 3D body landmarks (MediaPipe PoseLandmarker), DemoPerson for tests
retarget.py  landmarks -> robot joint angles (limb directions matched with the URDF, joint limits)
"""
