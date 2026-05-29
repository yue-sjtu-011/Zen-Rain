import cv2
import mediapipe as mp
import random
import numpy as np
import math

# --- 1. 颜色配置 (BGR) ---
COLOR_BG = (245, 251, 253)      
COLOR_DARK_BLUE = (173, 134, 74) 
COLOR_LIGHT_BLUE = (227, 212, 176) 
GLOBAL_THICKNESS = 2

# --- 透明融合辅助函数 ---
def draw_alpha_ellipse(img, center, axes, angle, start_angle, end_angle, color, thickness, alpha):
    """
    在原图上绘制具有真实透明度的椭圆/线条
    """
    if alpha <= 0: return img
    # 创建一个临时的覆盖层
    overlay = img.copy()
    cv2.ellipse(overlay, center, axes, angle, start_angle, end_angle, color, thickness, cv2.LINE_AA)
    # 按照 alpha 比例融合
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    return img

# --- 2. 舒展的长雨丝类 ---
class ArtisticDrop:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        angle_rad = math.radians(random.uniform(55, 88))
        speed = random.uniform(16, 24)
        self.vx = -speed * math.cos(angle_rad)
        self.vy = speed * math.sin(angle_rad)
        self.line_length_factor = random.uniform(5.0, 10.0) # 进一步拉长
        self.target_y = y + random.randint(150, 450)

    def move(self):
        self.x += self.vx
        self.y += self.vy
        return self.y >= self.target_y

    def draw(self, canvas):
        end_x = int(self.x - self.vx * self.line_length_factor * 0.5)
        end_y = int(self.y - self.vy * self.line_length_factor * 0.5)
        cv2.line(canvas, (int(self.x), int(self.y)), (end_x, end_y), 
                 COLOR_DARK_BLUE, GLOBAL_THICKNESS, cv2.LINE_AA)

# --- 3. 真正透明、具有层次感的涟漪类 ---
class ArtisticRipple:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.radius = 4
        self.num_rings = random.randint(1, 4)
        self.max_base_radius = random.randint(80, 160)
        
        # 每一圈的独立配置
        self.ring_configs = []
        for i in range(self.num_rings):
            self.ring_configs.append({
                "indent": i * random.randint(25, 55),
                "dx": random.randint(-18, 18) if i > 0 else 0,
                "dy": random.randint(-12, 12) if i > 0 else 0,
                "aspect": random.uniform(0.32, 0.4)
            })
            
        self.alpha = 1.0
        # 控制淡出节奏
        self.fade_speed = random.uniform(0.015, 0.025) 
        self.expand_speed = 3.2

    def update(self):
        self.radius += self.expand_speed
        progress = min(self.radius / (self.max_base_radius + 60), 1.0)
        self.alpha = 1.0 - progress

    def draw(self, canvas):
        if self.alpha <= 0: return
        
        # 1. 绘制底层填充色块 (最外圈)
        # 填充色块需要更淡一些，避免遮挡
        draw_alpha_ellipse(canvas, (int(self.x), int(self.y)), 
                          (int(self.radius), int(self.radius * 0.35)), 
                          0, 0, 360, COLOR_LIGHT_BLUE, -1, self.alpha * 0.3)
        
        # 2. 绘制多重同心线条
        for config in self.ring_configs:
            r = self.radius - config["indent"]
            if r > 5:
                cx, cy = int(self.x + config["dx"]), int(self.y + config["dy"])
                # 线条使用透明融合绘制
                draw_alpha_ellipse(canvas, (cx, cy), 
                                  (int(r), int(r * config["aspect"])), 
                                  0, 0, 360, COLOR_DARK_BLUE, GLOBAL_THICKNESS, self.alpha)

# --- 4. 主程序 ---
mp_hands = mp.solutions.hands
hand_detector = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8)
cap = cv2.VideoCapture(0)

last_state = "open"
drops = []
ripples = []

while True:
    success, frame = cap.read()
    if not success: break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    # 纯净画布
    canvas = np.full((h, w, 3), COLOR_BG, dtype=np.uint8)
    
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hand_detector.process(img_rgb)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            tips_ids = [4, 8, 12, 16, 20]
            mcps_ids = [2, 6, 10, 14, 18]
            opened_tips = [t for t, m in zip(tips_ids, mcps_ids) 
                           if handLms.landmark[t].y < handLms.landmark[m].y]
            current_state = "fist" if len(opened_tips) <= 1 else "open"

            if last_state == "fist" and current_state == "open":
                # 触发极简雨滴
                for tip_id in opened_tips:
                    if random.random() > 0.3:
                        tx, ty = int(handLms.landmark[tip_id].x * w), int(handLms.landmark[tip_id].y * h)
                        for _ in range(random.randint(1, 2)):
                            rx, ry = tx + random.randint(-50, 50), ty + random.randint(-50, 50)
                            drops.append(ArtisticDrop(rx, ry))
            last_state = current_state

    # 先绘制涟漪 (涟漪在底层，支持互相叠加穿透)
    for r in ripples[:]:
        r.update()
        if r.alpha <= 0: ripples.remove(r)
        else: r.draw(canvas)

    # 后绘制雨滴 (雨滴在顶层)
    for d in drops[:]:
        if d.move():
            ripples.append(ArtisticRipple(d.x, d.y))
            drops.remove(d)
        else:
            d.draw(canvas)

    # 文案排版
    cv2.putText(canvas, "Synapse", (60, 80), cv2.FONT_HERSHEY_DUPLEX, 1.5, COLOR_DARK_BLUE, GLOBAL_THICKNESS)
    cv2.putText(canvas, "Corner", (60, 130), cv2.FONT_HERSHEY_DUPLEX, 1.5, COLOR_DARK_BLUE, GLOBAL_THICKNESS)

    cv2.imshow("Zen Rain - True Alpha", canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()