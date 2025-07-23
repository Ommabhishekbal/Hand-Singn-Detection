import cv2
import os
import time
import mediapipe as mp

def capture_gesture(label, target_count=2000):
    save_dir = f"data/custom_captures/{label.upper()}"
    os.makedirs(save_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("❌ Camera not accessible")
        return

    hands = mp.solutions.hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    drawing = mp.solutions.drawing_utils

    print(f"\n🟡 Ready to capture for '{label.upper()}'.")
    print("Press SPACE to start capturing, Q to quit...")

    auto_capture = False
    count = len(os.listdir(save_dir))

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp.solutions.hands.HAND_CONNECTIONS
                )

            if auto_capture and count < target_count:
                timestamp = int(time.time() * 1000)
                cv2.imwrite(os.path.join(save_dir, f"{label}_{timestamp}.jpg"), frame)
                count += 1
                cv2.putText(frame, f"Captured: {count}/{target_count}", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                time.sleep(0.05)  # delay to prevent duplicates

            if count >= target_count:
                print(f"✅ Done capturing 1000 images for {label}")
                break
        else:
            cv2.putText(frame, "✋ No hand detected", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Capture Gesture", frame)

        key = cv2.waitKey(1)
        if key == ord(' '):  # Start auto-capture
            auto_capture = True
        elif key == ord('q'):  # Quit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    label = input("Enter the gesture label (e.g., A-Z): ").strip().upper()
    if len(label) == 1 and label.isalpha():
        capture_gesture(label)
    else:
        print("❌ Invalid label. Please enter a single alphabet.")
