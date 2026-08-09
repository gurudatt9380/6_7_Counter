import cv2
import mediapipe as mp
import numpy as np
import time

# ================================================================
# CONFIG
# ================================================================

CAM_INDEX = 1
FLIP_CAMERA = True
ROUND_SECONDS = 60

RAISE_THRESHOLD = 0.045
DEBOUNCE_TIME = 0.08
VISIBILITY_MIN = 0.35

CAM_WIDTH = 1280
CAM_HEIGHT = 720

# ================================================================
# COLORS - BGR
# ================================================================

BG = (15, 16, 20)
WHITE = (245, 247, 250)
LIGHT_GRAY = (165, 170, 180)
BLUE = (255, 149, 10)
GREEN = (90, 210, 90)
RED = (70, 70, 235)
DARK_PANEL = (25, 27, 33)
PANEL = (32, 34, 42)

FONT = cv2.FONT_HERSHEY_SIMPLEX

# ================================================================
# MEDIAPIPE
# ================================================================

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

pose_detector = mp_pose.Pose(
    model_complexity=0,
    smooth_landmarks=False,
    min_detection_confidence=0.4,
    min_tracking_confidence=0.3,
)

L_SHOULDER = mp_pose.PoseLandmark.LEFT_SHOULDER
R_SHOULDER = mp_pose.PoseLandmark.RIGHT_SHOULDER

L_ELBOW = mp_pose.PoseLandmark.LEFT_ELBOW
R_ELBOW = mp_pose.PoseLandmark.RIGHT_ELBOW

L_WRIST = mp_pose.PoseLandmark.LEFT_WRIST
R_WRIST = mp_pose.PoseLandmark.RIGHT_WRIST


# ================================================================
# UI HELPERS
# ================================================================

def alpha_rect(img, x1, y1, x2, y2, color, alpha=0.85):
    overlay = img.copy()
    cv2.rectangle(
        overlay,
        (x1, y1),
        (x2, y2),
        color,
        -1
    )
    cv2.addWeighted(
        overlay,
        alpha,
        img,
        1 - alpha,
        0,
        img
    )


def rounded_rect(img, pt1, pt2, color, radius=18, alpha=0.9):
    x1, y1 = pt1
    x2, y2 = pt2

    overlay = img.copy()

    cv2.rectangle(
        overlay,
        (x1 + radius, y1),
        (x2 - radius, y2),
        color,
        -1
    )

    cv2.rectangle(
        overlay,
        (x1, y1 + radius),
        (x2, y2 - radius),
        color,
        -1
    )

    cv2.circle(
        overlay,
        (x1 + radius, y1 + radius),
        radius,
        color,
        -1
    )

    cv2.circle(
        overlay,
        (x2 - radius, y1 + radius),
        radius,
        color,
        -1
    )

    cv2.circle(
        overlay,
        (x1 + radius, y2 - radius),
        radius,
        color,
        -1
    )

    cv2.circle(
        overlay,
        (x2 - radius, y2 - radius),
        radius,
        color,
        -1
    )

    cv2.addWeighted(
        overlay,
        alpha,
        img,
        1 - alpha,
        0,
        img
    )


def put_text(img, text, x, y, scale, color,
             thickness=1, align="left"):

    (tw, th), _ = cv2.getTextSize(
        text,
        FONT,
        scale,
        thickness
    )

    if align == "center":
        x -= tw // 2

    elif align == "right":
        x -= tw

    cv2.putText(
        img,
        text,
        (int(x), int(y)),
        FONT,
        scale,
        color,
        thickness,
        cv2.LINE_AA
    )


def get_screen_size():

    try:
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()

        sw = root.winfo_screenwidth()
        sh = root.winfo_screenheight()

        root.destroy()

        return sw, sh

    except Exception:

        return 1920, 1080


# ================================================================
# CAMERA -> 16:9
# ================================================================

def crop_to_16_9(frame):

    h, w = frame.shape[:2]

    target_ratio = 16 / 9
    current_ratio = w / h

    if abs(current_ratio - target_ratio) < 0.02:
        return frame

    if current_ratio > target_ratio:

        # Too wide
        new_w = int(h * target_ratio)

        x1 = (w - new_w) // 2
        x2 = x1 + new_w

        frame = frame[:, x1:x2]

    else:

        # Too tall
        new_h = int(w / target_ratio)

        y1 = (h - new_h) // 2
        y2 = y1 + new_h

        frame = frame[y1:y2, :]

    return frame


# ================================================================
# MAIN
# ================================================================

def main():

    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)

    if not cap.isOpened():
        cap = cv2.VideoCapture(CAM_INDEX)

    if not cap.isOpened():
        print("ERROR: Could not open camera.")
        return

    # Request proper widescreen camera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # ------------------------------------------------------------
    # GAME VARIABLES
    # ------------------------------------------------------------

    count = 0
    state = None

    last_rep_time = 0

    running = False
    start_time = None

    elapsed_frozen = 0.0
    time_up = False

    # ------------------------------------------------------------
    # SCREEN
    # ------------------------------------------------------------

    screen_w, screen_h = get_screen_size()

    cv2.namedWindow(
        "6-7",
        cv2.WINDOW_NORMAL
    )

    cv2.setWindowProperty(
        "6-7",
        cv2.WND_PROP_FULLSCREEN,
        cv2.WINDOW_FULLSCREEN
    )

    # ============================================================
    # LOOP
    # ============================================================

    while True:

        ok, frame = cap.read()

        if not ok:
            break

        if FLIP_CAMERA:
            frame = cv2.flip(frame, 1)

        # --------------------------------------------------------
        # Make camera 16:9
        # --------------------------------------------------------

        frame = crop_to_16_9(frame)

        frame = cv2.resize(
            frame,
            (screen_w, screen_h),
            interpolation=cv2.INTER_LINEAR
        )

        h, w = frame.shape[:2]

        # --------------------------------------------------------
        # POSE
        # --------------------------------------------------------

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        rgb.flags.writeable = False

        results = pose_detector.process(rgb)

        detected = False

        right_diff = 0.0
        left_diff = 0.0

        if results.pose_landmarks:

            lm = results.pose_landmarks.landmark

            rw = lm[R_WRIST]
            re = lm[R_ELBOW]

            lw = lm[L_WRIST]
            le = lm[L_ELBOW]

            if (
                rw.visibility > VISIBILITY_MIN and
                re.visibility > VISIBILITY_MIN and
                lw.visibility > VISIBILITY_MIN and
                le.visibility > VISIBILITY_MIN
            ):

                detected = True

                right_diff = re.y - rw.y
                left_diff = le.y - lw.y

            # Subtle skeleton
            mp_draw.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,

                mp_draw.DrawingSpec(
                    color=(210, 210, 210),
                    thickness=1,
                    circle_radius=2
                ),

                mp_draw.DrawingSpec(
                    color=BLUE,
                    thickness=2
                )
            )

        # ========================================================
        # COUNTING LOGIC
        # ========================================================

        if running and detected and not time_up:

            combined = right_diff - left_diff

            now = time.time()

            if state is None:

                if abs(combined) >= RAISE_THRESHOLD:

                    state = (
                        "right_up"
                        if combined > 0
                        else "left_up"
                    )

            else:

                flipped = (
                    state == "right_up"
                    and combined <= -RAISE_THRESHOLD
                ) or (
                    state == "left_up"
                    and combined >= RAISE_THRESHOLD
                )

                if flipped:

                    if now - last_rep_time >= DEBOUNCE_TIME:

                        count += 1
                        last_rep_time = now

                    state = (
                        "left_up"
                        if state == "right_up"
                        else "right_up"
                    )

        # ========================================================
        # TIMER
        # ========================================================

        if running and not time_up:

            elapsed = (
                elapsed_frozen +
                (time.time() - start_time)
            )

            if elapsed >= ROUND_SECONDS:

                elapsed = ROUND_SECONDS

                running = False
                time_up = True

        else:

            elapsed = elapsed_frozen

        remaining = max(
            0.0,
            ROUND_SECONDS - elapsed
        )

        # ========================================================
        # UI
        # ========================================================

        # --------------------------------------------------------
        # TOP HUD
        # --------------------------------------------------------

        # Dark translucent header
        alpha_rect(
            frame,
            0,
            0,
            w,
            112,
            BG,
            0.82
        )

        # Bottom border
        cv2.line(
            frame,
            (0, 111),
            (w, 111),
            (55, 58, 68),
            1
        )

        # --------------------------------------------------------
        # LOGO
        # --------------------------------------------------------

        put_text(
            frame,
            "6—7",
            42,
            48,
            1.25,
            BLUE,
            3
        )

        put_text(
            frame,
            "POSE CHALLENGE",
            44,
            78,
            0.42,
            LIGHT_GRAY,
            1
        )

        # --------------------------------------------------------
        # SCORE
        # --------------------------------------------------------

        put_text(
            frame,
            str(count),
            w // 2,
            67,
            1.65,
            WHITE,
            4,
            align="center"
        )

        put_text(
            frame,
            "REPETITIONS",
            w // 2,
            92,
            0.38,
            LIGHT_GRAY,
            1,
            align="center"
        )

        # --------------------------------------------------------
        # TIMER CARD
        # --------------------------------------------------------

        timer_x1 = w - 190
        timer_x2 = w - 35

        rounded_rect(
            frame,
            (timer_x1, 23),
            (timer_x2, 88),
            PANEL,
            radius=14,
            alpha=0.95
        )

        if remaining <= 10 and running:
            timer_color = RED
        else:
            timer_color = WHITE

        put_text(
            frame,
            f"{remaining:04.1f}",
            (timer_x1 + timer_x2) // 2,
            59,
            0.75,
            timer_color,
            2,
            align="center"
        )

        put_text(
            frame,
            "SECONDS",
            (timer_x1 + timer_x2) // 2,
            78,
            0.28,
            LIGHT_GRAY,
            1,
            align="center"
        )

        # ========================================================
        # STATUS
        # ========================================================

        if time_up:

            status_text = "TIME'S UP"
            status_color = RED

        elif running and not detected:

            status_text = "SHOW BOTH ARMS"
            status_color = WHITE

        elif running:

            status_text = "●  RUNNING"
            status_color = GREEN

        else:

            status_text = "PAUSED"
            status_color = LIGHT_GRAY

        # --------------------------------------------------------
        # STATUS PILL
        # --------------------------------------------------------

        (tw, th), _ = cv2.getTextSize(
            status_text,
            FONT,
            0.48,
            1
        )

        pill_w = tw + 46
        pill_h = 46

        pill_x1 = (w - pill_w) // 2
        pill_x2 = pill_x1 + pill_w

        pill_y1 = h - 105
        pill_y2 = pill_y1 + pill_h

        rounded_rect(
            frame,
            (pill_x1, pill_y1),
            (pill_x2, pill_y2),
            BG,
            radius=22,
            alpha=0.88
        )

        put_text(
            frame,
            status_text,
            w // 2,
            pill_y1 + 30,
            0.48,
            status_color,
            1,
            align="center"
        )

        # ========================================================
        # BOTTOM CONTROL BAR
        # ========================================================

        alpha_rect(
            frame,
            0,
            h - 45,
            w,
            h,
            BG,
            0.85
        )

        controls = (
            "SPACE  START / PAUSE       "
            "R  RESET       "
            "Q  QUIT"
        )

        put_text(
            frame,
            controls,
            w // 2,
            h - 16,
            0.34,
            LIGHT_GRAY,
            1,
            align="center"
        )

        # ========================================================
        # DISPLAY
        # ========================================================

        cv2.imshow(
            "6-7",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        # --------------------------------------------------------
        # CONTROLS
        # --------------------------------------------------------

        if key in (ord("q"), 27):

            break

        elif key == ord(" "):

            if not time_up:

                if running:

                    elapsed_frozen = elapsed
                    running = False

                else:

                    start_time = time.time()
                    running = True

        elif key in (ord("r"), ord("R")):

            running = False
            start_time = None
            elapsed_frozen = 0.0

            count = 0
            state = None

            time_up = False

    # ============================================================
    # CLEANUP
    # ============================================================

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
