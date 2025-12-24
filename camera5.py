import cv2
import mediapipe as mp
import numpy as np
import json
import os

mp_hands = mp.solutions.hands

HAND_LABELS_FILE = "hand_labels.json"

PINCH_SENSITIVITY_RADIUS = 0.05

HAND_SCALE_REFERENCE = 0.2

VOLUME_SENSITIVITY = 2.0

LANDMARK_NAMES = {
    0: "Wrist",
    1: "Thumb_CMC", 2: "Thumb_MCP", 3: "Thumb_IP", 4: "Thumb_Tip",
    5: "Index_MCP", 6: "Index_PIP", 7: "Index_DIP", 8: "Index_Tip",
    9: "Middle_MCP", 10: "Middle_PIP", 11: "Middle_DIP", 12: "Middle_Tip",
    13: "Ring_MCP", 14: "Ring_PIP", 15: "Ring_DIP", 16: "Ring_Tip",
    17: "Pinky_MCP", 18: "Pinky_PIP", 19: "Pinky_DIP", 20: "Pinky_Tip"
}

FINGER_GROUPS = {
    "Thumb": [1, 2, 3, 4],
    "Index": [5, 6, 7, 8],
    "Middle": [9, 10, 11, 12],
    "Ring": [13, 14, 15, 16],
    "Pinky": [17, 18, 19, 20]
}

def calculate_angle(point1, point2, point3):
    a = np.array([point1.x, point1.y])
    b = np.array([point2.x, point2.y])
    c = np.array([point3.x, point3.y])
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
    
    return angle

def draw_bone_lines(image, landmarks, connections, color=(0, 0, 255), thickness=2):
    h, w, _ = image.shape
    
    for connection in connections:
        start_idx, end_idx = connection
        if start_idx < len(landmarks) and end_idx < len(landmarks):
            start_point = landmarks[start_idx]
            end_point = landmarks[end_idx]
            x1 = int(start_point.x * w)
            y1 = int(start_point.y * h)
            x2 = int(end_point.x * w)
            y2 = int(end_point.y * h)
            cv2.line(image, (x1, y1), (x2, y2), color, thickness)

def label_landmarks(image, landmarks, hand_num=0):
    h, w, _ = image.shape
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.3
    thickness = 1
    
    for idx, landmark in enumerate(landmarks):
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        cv2.circle(image, (x, y), 3, (0, 255, 0), -1)
        label = LANDMARK_NAMES.get(idx, f"P{idx}")
        cv2.putText(image, label, (x + 5, y - 5), font, font_scale, (255, 255, 255), thickness)

def calculate_finger_angles(landmarks):
    angles = {}
    wrist = landmarks[0]
    finger_order = ["Index", "Middle", "Ring", "Pinky"]
    
    thumb_indices = FINGER_GROUPS["Thumb"]
    if len(thumb_indices) >= 3:
        base = landmarks[thumb_indices[0]]
        knuckle = landmarks[thumb_indices[1]]
        joint = landmarks[thumb_indices[2]]
        angle = calculate_angle(base, knuckle, joint)
        angles["Thumb_MCP"] = angle
    
    prev_mcp_idx = None
    for finger_name in finger_order:
        indices = FINGER_GROUPS[finger_name]
        knuckle_idx = indices[0]
        joint_idx = indices[1]
        
        if knuckle_idx < len(landmarks) and joint_idx < len(landmarks):
            if finger_name == "Index":
                base = wrist
            elif prev_mcp_idx is not None:
                base = landmarks[prev_mcp_idx]
            else:
                base = wrist
            
            knuckle = landmarks[knuckle_idx]
            joint = landmarks[joint_idx]
            angle = calculate_angle(base, knuckle, joint)
            angles[f"{finger_name}_MCP"] = angle
            prev_mcp_idx = knuckle_idx
    
    return angles

def calculate_distance(point1, point2):
    return np.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

def calculate_hand_scale(landmarks):
    if len(landmarks) < 21:
        return 1.0
    
    thumb_cmc = landmarks[1]
    pinky_mcp = landmarks[17]
    palm_width = calculate_distance(thumb_cmc, pinky_mcp)
    
    middle_mcp = landmarks[9]
    middle_tip = landmarks[12]
    finger_length = calculate_distance(middle_mcp, middle_tip)
    
    hand_size = (palm_width + finger_length) / 2
    scale_factor = hand_size / HAND_SCALE_REFERENCE
    scale_factor = max(0.3, min(2.0, scale_factor))
    
    return scale_factor

def calculate_sensitivity_multiplier(hand_scale):
    if hand_scale <= 1.0:
        sensitivity = 1.0
    else:
        t = (hand_scale - 1.0) / (2.0 - 1.0)
        t = max(0.0, min(1.0, t))
        sensitivity = 1.0 + 0.75 * (t ** 2.5)
    
    return sensitivity

def detect_pinch(landmarks, threshold=None):
    if len(landmarks) < 9:
        return False
    
    if threshold is None:
        threshold = PINCH_SENSITIVITY_RADIUS
    
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    distance = calculate_distance(thumb_tip, index_tip)
    return distance < threshold

def is_index_finger_extended(landmarks):
    if len(landmarks) < 9:
        return False
    tip = landmarks[8]
    pip = landmarks[6]
    mcp = landmarks[5]
    return tip.y < pip.y < mcp.y

def is_index_finger_highest(landmarks):
    if len(landmarks) < 21:
        return False
    
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]
    thumb_tip = landmarks[4]
    
    index_y = index_tip.y
    return (index_y < middle_tip.y and 
            index_y < ring_tip.y and 
            index_y < pinky_tip.y and 
            index_y < thumb_tip.y)

def is_palm_open(landmarks):
    if len(landmarks) < 21:
        return False
    
    index_extended = landmarks[8].y < landmarks[6].y < landmarks[5].y
    middle_extended = landmarks[12].y < landmarks[10].y < landmarks[9].y
    ring_extended = landmarks[16].y < landmarks[14].y < landmarks[13].y
    pinky_extended = landmarks[20].y < landmarks[18].y < landmarks[17].y
    
    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]
    thumb_mcp = landmarks[2]
    thumb_extended = (thumb_tip.x > thumb_ip.x > thumb_mcp.x) or (thumb_tip.x < thumb_ip.x < thumb_mcp.x)
    
    return index_extended and middle_extended and ring_extended and pinky_extended and thumb_extended

def draw_thumb_pinch_semicircle(frame, landmarks, hand_label="Unknown"):
    if len(landmarks) < 5:
        return
    
    h, w = frame.shape[:2]
    thumb_mcp = landmarks[2]
    thumb_tip = landmarks[4]
    tip_x = int(thumb_tip.x * w)
    tip_y = int(thumb_tip.y * h)
    mcp_x = int(thumb_mcp.x * w)
    mcp_y = int(thumb_mcp.y * h)
    
    bone_dx = tip_x - mcp_x
    bone_dy = tip_y - mcp_y
    bone_angle = np.degrees(np.arctan2(bone_dy, bone_dx))
    perp_angle = bone_angle + 90
    
    if hand_label == "Left":
        perp_angle += 180
    
    hand_scale = calculate_hand_scale(landmarks)
    base_radius = PINCH_SENSITIVITY_RADIUS * (w + h) / 2
    radius_pixels = int(base_radius * hand_scale)
    
    flat_angle_rad = np.radians(perp_angle)
    flat_end1_x = int(tip_x + radius_pixels * np.cos(flat_angle_rad))
    flat_end1_y = int(tip_y + radius_pixels * np.sin(flat_angle_rad))
    flat_end2_x = int(tip_x - radius_pixels * np.cos(flat_angle_rad))
    flat_end2_y = int(tip_y - radius_pixels * np.sin(flat_angle_rad))
    
    cv2.line(frame, (flat_end1_x, flat_end1_y), (flat_end2_x, flat_end2_y), (0, 255, 0), 2)
    cv2.ellipse(frame, (tip_x, tip_y), (radius_pixels, radius_pixels), 
                perp_angle, -90, 90, (0, 255, 0), 2)

def save_hand_labels(hand_labels):
    try:
        with open(HAND_LABELS_FILE, 'w') as f:
            json.dump(hand_labels, f, indent=2)
    except Exception as e:
        print(f"Error saving hand labels: {e}")

def load_hand_labels():
    if os.path.exists(HAND_LABELS_FILE):
        try:
            with open(HAND_LABELS_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading hand labels: {e}")
    return {}

def draw_cube(frame, size, position_x, position_y, rotation_x=0.0, rotation_y=0.0, rotation_z=0.0):
    cube_size = int(size)
    center_x = position_x
    center_y = position_y
    vertices_3d = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]
    ]) * (cube_size / 2)

    rot_x_matrix = np.array([
        [1, 0, 0],
        [0, np.cos(rotation_x), -np.sin(rotation_x)],
        [0, np.sin(rotation_x), np.cos(rotation_x)]
    ])

    rot_y_matrix = np.array([
        [np.cos(rotation_y), 0, np.sin(rotation_y)],
        [0, 1, 0],
        [-np.sin(rotation_y), 0, np.cos(rotation_y)]
    ])

    rot_z_matrix = np.array([
        [np.cos(rotation_z), -np.sin(rotation_z), 0],
        [np.sin(rotation_z), np.cos(rotation_z), 0],
        [0, 0, 1]
    ])
    
    combined_rotation = rot_x_matrix @ rot_y_matrix @ rot_z_matrix

    vertices_2d = []
    vertices_z = []
    for vertex in vertices_3d:
        rotated = combined_rotation @ vertex
        vertices_z.append(rotated[2])
        x = int(center_x + rotated[0])
        y = int(center_y + rotated[1] - rotated[2] * 0.45)
        vertices_2d.append((x, y))

    base_blue = (200, 100, 0)
    
    faces = [
        ([0, 1, 2, 3], np.array([0, 0, -1])),
        ([4, 7, 6, 5], np.array([0, 0, 1])),
        ([3, 2, 6, 7], np.array([0, 1, 0])),
        ([0, 4, 5, 1], np.array([0, -1, 0])),
        ([1, 5, 6, 2], np.array([1, 0, 0])),
        ([0, 3, 7, 4], np.array([-1, 0, 0]))
    ]

    light_dir = np.array([0.5, -0.5, -1.0])
    light_dir = light_dir / np.linalg.norm(light_dir)

    faces_with_depth = []
    for face_indices, normal in faces:
        avg_z = np.mean([vertices_z[i] for i in face_indices])
        
        rotated_normal = combined_rotation @ normal
        rotated_normal = rotated_normal / np.linalg.norm(rotated_normal)
        
        dot_product = np.dot(rotated_normal, light_dir)
        brightness = max(0.2, min(1.0, (dot_product + 1.0) / 2.0))
        
        shaded_color = tuple(int(c * brightness) for c in base_blue)
        
        faces_with_depth.append((face_indices, shaded_color, avg_z))
    faces_with_depth.sort(key=lambda x: x[2], reverse=True)

    for face_indices, color, _ in faces_with_depth:
        pts = np.array([vertices_2d[i] for i in face_indices], np.int32)
        cv2.fillPoly(frame, [pts], color)

def draw_volume_bar(frame, volume_level, bar_width=40, margin=20):
    h, w = frame.shape[:2]
    bar_x = w - bar_width - margin
    full_height = h - 2 * margin
    bar_height = int(full_height * 0.85)
    center_y = h // 2
    bar_top = center_y - bar_height // 2
    bar_bottom = center_y + bar_height // 2
    
    filled_height = int(volume_level * bar_height)
    filled_bottom = bar_bottom
    filled_top = bar_bottom - filled_height
    
    if filled_height < bar_height:
        empty_roi = frame[bar_top:filled_top, bar_x:bar_x + bar_width].copy()
        black_overlay = np.zeros_like(empty_roi)
        blended = cv2.addWeighted(empty_roi, 0.75, black_overlay, 0.25, 0)
        frame[bar_top:filled_top, bar_x:bar_x + bar_width] = blended
    
    if filled_height > 0:
        cv2.rectangle(frame, (bar_x, filled_top), (bar_x + bar_width, filled_bottom), (0, 255, 0), -1)
    
    cv2.rectangle(frame, (bar_x, bar_top), (bar_x + bar_width, bar_bottom), (255, 255, 255), 1)
    return bar_x, bar_top, bar_bottom

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
        
        print("Starting hand detection. Press 'q' to quit.")
        
        volume_level = 0.5
        is_pinching = False
        last_pinch_y = None
        hand_labels_dict = {}
        saved_labels = load_hand_labels()
        cube_size = 70.0
        cube_rotation_x = 0.0
        cube_rotation_y = 0.0
        cube_rotation_z = 0.0
        rotation_velocity_x = 0.0
        rotation_velocity_y = 0.0
        last_two_hand_distance = None
        prev_index_tip_pos = {}
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            bar_x, bar_top, bar_bottom = draw_volume_bar(frame, volume_level)
            any_hand_pinching = False
            current_sensitivity = None
            pinching_hands = []
            swipe_hands = []
            
            cube_x = w // 2
            cube_y = h // 2
            
            open_palm_count = 0
            one_hand_raised = False
            
            if results.multi_hand_landmarks:
                hand_count = len(results.multi_hand_landmarks)
                
                if results.multi_handedness:
                    for idx, classification in enumerate(results.multi_handedness):
                        hand_label = classification.classification[0].label
                        hand_labels_dict[idx] = hand_label
                        save_hand_labels(hand_labels_dict)
                
                for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    hand_label = hand_labels_dict.get(hand_idx, "Unknown")
                    hand_connections = mp_hands.HAND_CONNECTIONS
                    draw_bone_lines(frame, hand_landmarks.landmark, hand_connections, color=(0, 0, 255), thickness=1)
                    label_landmarks(frame, hand_landmarks.landmark, hand_idx)
                    draw_thumb_pinch_semicircle(frame, hand_landmarks.landmark, hand_label)
                    pinching = detect_pinch(hand_landmarks.landmark)
                    
                    wrist = hand_landmarks.landmark[0]
                    wrist_y_pixel = int(wrist.y * h)
                    wrist_x_pixel = int(wrist.x * w)
                    label_text = f"{hand_label}"
                    if pinching:
                        label_text += " [PINCHING]"
                        any_hand_pinching = True
                        pinching_hands.append((wrist_x_pixel, wrist_y_pixel, wrist, hand_idx))
                    cv2.putText(frame, label_text, (wrist_x_pixel - 30, wrist_y_pixel - 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
                    if is_palm_open(hand_landmarks.landmark):
                        open_palm_count += 1
                    
                    if is_index_finger_highest(hand_landmarks.landmark):
                        tip = hand_landmarks.landmark[8]
                        tip_x = tip.x
                        tip_y = tip.y
                        prev_pos = prev_index_tip_pos.get(hand_idx)
                        movement_delta_x = None
                        movement_delta_y = None
                        if prev_pos is not None:
                            movement_delta_x = tip_x - prev_pos[0]
                            movement_delta_y = tip_y - prev_pos[1]
                        prev_index_tip_pos[hand_idx] = (tip_x, tip_y)
                        swipe_hands.append((hand_idx, movement_delta_x, movement_delta_y))
                    else:
                        prev_index_tip_pos.pop(hand_idx, None)
                    
                    hand_scale = calculate_hand_scale(hand_landmarks.landmark)
                    sensitivity_mult = calculate_sensitivity_multiplier(hand_scale)
                    total_sensitivity = sensitivity_mult * VOLUME_SENSITIVITY
                    current_sensitivity = total_sensitivity
                    
                
                two_hands_pinching = len(pinching_hands) == 2
                
                if two_hands_pinching:
                    wrist1 = pinching_hands[0][2]
                    wrist2 = pinching_hands[1][2]
                    current_distance = calculate_distance(wrist1, wrist2)
                    
                    if last_two_hand_distance is not None:
                        distance_delta = current_distance - last_two_hand_distance
                        scale_factor = 1.0 + (distance_delta * 10.0)
                        cube_size *= scale_factor
                        cube_size = max(20.0, min(200.0, cube_size))
                    
                    last_two_hand_distance = current_distance
                    is_pinching = False
                    last_pinch_y = None
                else:
                    last_two_hand_distance = None
                    cube_rotation_controlled = False
                    if len(pinching_hands) == 1:
                        pinch_hand_idx = pinching_hands[0][3]
                        
                        for hand_idx, delta_x, delta_y in swipe_hands:
                            if hand_idx != pinch_hand_idx and (delta_x is not None or delta_y is not None):
                                flipped_delta_x = -delta_x if delta_x is not None else 0.0
                                delta_y_val = delta_y if delta_y is not None else 0.0
                                
                                rotation_sensitivity = 1.5
                                
                                movement_threshold = 0.001
                                
                                abs_delta_x = abs(flipped_delta_x)
                                abs_delta_y = abs(delta_y_val)
                                
                                has_movement = abs_delta_x > movement_threshold or abs_delta_y > movement_threshold
                                
                                if has_movement:
                                    if abs_delta_x > movement_threshold:
                                        rotation_velocity_y += flipped_delta_x * rotation_sensitivity
                                    
                                    if abs_delta_y > movement_threshold:
                                        rotation_velocity_x += delta_y_val * rotation_sensitivity
                                    
                                    rotation_velocity_x = max(-0.1, min(0.1, rotation_velocity_x))
                                    rotation_velocity_y = max(-0.1, min(0.1, rotation_velocity_y))
                                    
                                    cube_rotation_controlled = True
                                break
                    
                    if not cube_rotation_controlled and len(pinching_hands) == 1 and current_sensitivity is not None:
                        wrist_y_pixel = pinching_hands[0][1]
                        if not is_pinching:
                            is_pinching = True
                            last_pinch_y = wrist_y_pixel
                        elif is_pinching:
                            if last_pinch_y is not None:
                                delta_y = last_pinch_y - wrist_y_pixel
                                bar_height = bar_bottom - bar_top
                                volume_delta = (delta_y / bar_height) * current_sensitivity
                                volume_level += volume_delta
                                volume_level = max(0.0, min(1.0, volume_level))
                            last_pinch_y = wrist_y_pixel
                
                if not any_hand_pinching:
                    is_pinching = False
                    last_pinch_y = None
                
                cv2.putText(frame, f"Hands: {hand_count}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Volume: {int(volume_level * 100)}%", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                if current_sensitivity is not None:
                    cv2.putText(frame, f"Sensitivity: {current_sensitivity:.2f}x", (10, 110), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            else:
                is_pinching = False
                last_pinch_y = None
                cv2.putText(frame, "No hands detected", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            if open_palm_count == 1:
                rotation_velocity_x = 0.0
                rotation_velocity_y = 0.0
            
            if open_palm_count == 2:
                rotation_velocity_x = 0.0
                rotation_velocity_y = 0.0
            
            cube_rotation_x += rotation_velocity_x
            cube_rotation_y += rotation_velocity_y
            
            if cube_rotation_x > 2 * np.pi:
                cube_rotation_x -= 2 * np.pi
            elif cube_rotation_x < -2 * np.pi:
                cube_rotation_x += 2 * np.pi
            if cube_rotation_y > 2 * np.pi:
                cube_rotation_y -= 2 * np.pi
            elif cube_rotation_y < -2 * np.pi:
                cube_rotation_y += 2 * np.pi
            
            rotation_velocity_x *= 0.98 
            rotation_velocity_y *= 0.98 
            
            draw_cube(frame, cube_size, cube_x, cube_y, cube_rotation_x, cube_rotation_y, cube_rotation_z)
            
            scale_factor = 0.65
            new_width = int(w * scale_factor)
            new_height = int(h * scale_factor)
            display_frame = cv2.resize(frame, (new_width, new_height))
            cv2.imshow('follow @mustithegoat', display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    save_hand_labels(hand_labels_dict)
    cap.release()
    cv2.destroyAllWindows()
    print("\nHand detection stopped.")

if __name__ == "__main__":
    main()
