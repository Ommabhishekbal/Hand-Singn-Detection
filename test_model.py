import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

class ASLTester:
    def __init__(self, model_path='model/best_model.h5'):
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

        self.model = tf.keras.models.load_model(model_path)
        self.class_names = np.load('model/label_classes.npy', allow_pickle=True)
        print(f"✅ Loaded model with classes: {self.class_names}")

        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Error: Could not open webcam")
            return

        print("▶️ ASL Tester running. Press 'q' to quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)

            if results.multi_hand_landmarks:
                features = self._extract_features(results)
                if features is not None:
                    prediction = self.model.predict(np.array([features]), verbose=0)
                    pred_idx = np.argmax(prediction)
                    pred_class = self.class_names[pred_idx]
                    confidence = prediction[0][pred_idx]

                    label = f"{pred_class} ({confidence*100:.1f}%)"
                    if confidence < 0.6:
                        label = "Unknown"

                    print(f"▶️ Prediction: {pred_class}, Confidence: {confidence:.3f}")
                    cv2.putText(frame, label, (20, 60), self.font, 1.2, (0, 255, 0), 2)

                # Draw landmarks
                for hand_landmarks in results.multi_hand_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp.solutions.hands.HAND_CONNECTIONS,
                        mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                        mp.solutions.drawing_styles.get_default_hand_connections_style()
                    )
            else:
                cv2.putText(frame, "No hand detected", (20, 60), self.font, 1.2, (0, 0, 255), 2)

            cv2.imshow("ASL Real-Time Tester", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def _extract_features(self, results):
        try:
            landmarks = results.multi_hand_landmarks[0].landmark
            coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])

            # Normalize relative to wrist (landmark 0)
            coords -= coords[0]

            max_val = np.max(np.abs(coords))
            if max_val > 0:
                coords /= max_val

            return coords.flatten()
        except Exception as e:
            print(f"❌ Feature extraction error: {e}")
            return None

if __name__ == "__main__":
    tester = ASLTester()
    tester.run()
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import math

class ASLTester:
    def __init__(self, model_path='model/best_model.h5'):
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.model = tf.keras.models.load_model(model_path)
        self.class_names = np.load('model/label_classes.npy', allow_pickle=True)
        print(f"✅ Loaded model with classes: {self.class_names}")
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Error: Could not open webcam")
            return

        print("▶️ ASL Tester running. Press 'q' to quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)

            if results.multi_hand_landmarks:
                features = self._extract_features(results)
                if features is not None:
                    prediction = self.model.predict(np.array([features]), verbose=0)
                    pred_idx = np.argmax(prediction)
                    pred_class = self.class_names[pred_idx]
                    confidence = prediction[0][pred_idx]

                    label = f"{pred_class} ({confidence*100:.1f}%)"
                    if confidence < 0.6:
                        label = "Unknown"

                    print("▶️", label)
                    cv2.putText(frame, label, (20, 60), self.font, 1.2, (0, 255, 0), 2)

                # Draw landmarks
                for hand_landmarks in results.multi_hand_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp.solutions.hands.HAND_CONNECTIONS,
                        mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                        mp.solutions.drawing_styles.get_default_hand_connections_style()
                    )
            else:
                cv2.putText(frame, "No hand detected", (20, 60), self.font, 1.2, (0, 0, 255), 2)

            cv2.imshow("ASL Real-Time Tester", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def _extract_features(self, results):
        try:
            landmarks = results.multi_hand_landmarks[0].landmark
            x = np.array([lm.x for lm in landmarks])
            y = np.array([lm.y for lm in landmarks])
            z = np.array([lm.z for lm in landmarks])

            # Normalize by wrist (landmark 0)
            x -= x[0]
            y -= y[0]
            z -= z[0]

            coords = np.concatenate([x, y, z])

            # Compute angles between adjacent joints for each finger
            angle_features = []
            connections = [
                (0, 5, 8),   # Index
                (0, 9, 12),  # Middle
                (0, 13, 16), # Ring
                (0, 17, 20), # Pinky
                (0, 1, 4),   # Thumb
            ]

            for a, b, c in connections:
                angle = self._calculate_angle(landmarks[a], landmarks[b], landmarks[c])
                angle_features.append(angle / 180.0)  # Normalize to 0–1

            final_features = np.concatenate([coords, angle_features])
            return final_features

        except Exception as e:
            print(f"❌ Feature extraction error: {e}")
            return None

    def _calculate_angle(self, a, b, c):
        """Calculate angle between three landmarks (in degrees)"""
        a = np.array([a.x, a.y])
        b = np.array([b.x, b.y])
        c = np.array([c.x, c.y])

        ba = a - b
        bc = c - b

        cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        return np.degrees(angle)

if __name__ == "__main__":
    tester = ASLTester()
    tester.run()
