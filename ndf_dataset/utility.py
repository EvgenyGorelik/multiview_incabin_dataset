import numpy as np

def vertices_to_3d_bbox(verts):
    min_pt = np.min(verts, axis=0)
    max_pt = np.max(verts, axis=0)
    bbox_corners = np.array([
        [min_pt[0], min_pt[1], min_pt[2]],
        [max_pt[0], min_pt[1], min_pt[2]],
        [max_pt[0], max_pt[1], min_pt[2]],
        [min_pt[0], max_pt[1], min_pt[2]],
        [min_pt[0], min_pt[1], max_pt[2]],
        [max_pt[0], min_pt[1], max_pt[2]],
        [max_pt[0], max_pt[1], max_pt[2]],
        [min_pt[0], max_pt[1], max_pt[2]],
    ])
    return bbox_corners


MHR_NAMES = {
    0: "nose",
    1: "left-eye",
    2: "right-eye",
    3: "left-ear",
    4: "right-ear",
    5: "left-shoulder",
    6: "right-shoulder",
    7: "left-elbow",
    8: "right-elbow",
    9: "left-hip",
    10: "right-hip",
    11: "left-knee",
    12: "right-knee",
    13: "left-ankle",
    14: "right-ankle",
    15: "left-big-toe-tip",
    16: "left-small-toe-tip",
    17: "left-heel",
    18: "right-big-toe-tip",
    19: "right-small-toe-tip",
    20: "right-heel",
    21: "right-thumb-tip",
    22: "right-thumb-first-joint",
    23: "right-thumb-second-joint",
    24: "right-thumb-third-joint",
    25: "right-index-tip",
    26: "right-index-first-joint",
    27: "right-index-second-joint",
    28: "right-index-third-joint",
    29: "right-middle-tip",
    30: "right-middle-first-joint",
    31: "right-middle-second-joint",
    32: "right-middle-third-joint",
    33: "right-ring-tip",
    34: "right-ring-first-joint",
    35: "right-ring-second-joint",
    36: "right-ring-third-joint",
    37: "right-pinky-tip",
    38: "right-pinky-first-joint",
    39: "right-pinky-second-joint",
    40: "right-pinky-third-joint",
    41: "right-wrist",
    42: "left-thumb-tip",
    43: "left-thumb-first-joint",
    44: "left-thumb-second-joint",
    45: "left-thumb-third-joint",
    46: "left-index-tip",
    47: "left-index-first-joint",
    48: "left-index-second-joint",
    49: "left-index-third-joint",
    50: "left-middle-tip",
    51: "left-middle-first-joint",
    52: "left-middle-second-joint",
    53: "left-middle-third-joint",
    54: "left-ring-tip",
    55: "left-ring-first-joint",
    56: "left-ring-second-joint",
    57: "left-ring-third-joint",
    58: "left-pinky-tip",
    59: "left-pinky-first-joint",
    60: "left-pinky-second-joint",
    61: "left-pinky-third-joint",
    62: "left-wrist",
    63: "left-olecranon",
    64: "right-olecranon",
    65: "left-cubital-fossa",
    66: "right-cubital-fossa",
    67: "left-acromion",
    68: "right-acromion",
    69: "neck",
}


MHR_CONNECTIONS = [
    (0, 1), 
    (1, 3), 
    (0, 2), 
    (2, 4), 
    (0, 69),
    (69, 67),
    (69, 68),
    (67, 5),
    (68, 6),
    (69, 5),
    (69, 6),
    (5, 9),
    (6, 10),
    (5, 7),
    (7, 65),
    (7, 63),
    (7, 62),
    (6, 8),
    (8, 66),
    (8, 64),
    (8, 41),
    (41, 24),
    (24, 23),
    (23, 22),
    (22, 21),
    (41, 28),
    (28, 27),
    (27, 26),
    (26, 25),
    (41, 32),
    (32, 31),
    (31, 30),
    (30, 29),
    (41, 36),
    (36, 35),
    (35, 34),
    (34, 33),
    (41, 40),
    (40, 39),
    (39, 38),
    (38, 37),
    (62, 45),
    (45, 44),
    (44, 43),
    (43, 42),
    (62, 49),
    (49, 48),
    (48, 47),
    (47, 46),
    (62, 53),
    (53, 52),
    (52, 51),
    (51, 50),
    (62, 57),
    (57, 56),
    (56, 55),
    (55, 54),
    (62, 61),
    (61, 60),
    (60, 59),
    (59, 58),
    (9, 10),
    (9, 11),
    (11, 13),
    (13, 17),
    (13, 15),
    (13, 16),
    (10, 12),
    (12, 14),
    (14, 20),
    (14, 18),
    (14, 19)
]

def keypoints_to_skeleton_edges(keypoints):
    edges = []
    for joint_a, joint_b in MHR_CONNECTIONS:
        if joint_a in keypoints and joint_b in keypoints:
            edges.append((keypoints[joint_a], keypoints[joint_b]))
    return edges

def get_intrinsics(cam_info):
    K = np.eye(3)
    K[0, 0] = cam_info.get('fx', 1.0)
    K[1, 1] = cam_info.get('fy', 1.0)
    K[0, 2] = cam_info.get('cx', 0.0)
    K[1, 2] = cam_info.get('cy', 0.0)
    return K

def compute_child_to_base(transforms_child_to_parent, child):
    """Walk child->parent links until base_link or until can't continue.
        Compose transforms so final T maps original child coords -> base_link coords.
    """
    cur = child
    mats = []
    visited = set()
    while True:
        if cur in visited:
            return None
        visited.add(cur)
        entry = transforms_child_to_parent.get(cur)
        if entry is None:
            return None
        parent, T_child_to_parent = entry
        mats.append(T_child_to_parent)  # child -> parent
        if parent == "base_link":
            # compose: total = T_parent_n * ... * T_child_to_parent
            total = np.eye(4)
            for M in mats:
                total = M @ total
            return total
        # next child is parent frame (walk upward)
        cur = parent
        if cur is None:
            return None
