import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

def create_mask(image):
    #Эллипсы для удаления
    ellipses_params = [
        ((12, 18), -285, 187),
        ((12, 18), -268, 187),
        ((12, 18), -247, 187),
        ((12, 18), -228, 187),
        ((12, 18), -208, 187),
        ((12, 18), -189, 187),
        ((12, 18), -65, 187),
        ((12, 18), -48, 187),
        ((12, 18), -27, 187),
        ((12, 18), -8, 187),
        ((12, 18), 14, 187),
        ((12, 18), 34, 187),
        ((12, 18), 178, 187),
        ((12, 18), 198, 187),
        ((12, 18), 215, 187),
        ((12, 18), 237, 187),
        ((12, 18), 254, 187)
    ]
    # треугольники для удаления
    corner_width=500
    corner_height=150
    """Создает маску для ORB, исключая верхние углы изображения (размер corner_width x corner_height) и области с эллипсами."""
    height, width = image.shape[:2]

    # Создаем полностью белую маску (255 - разрешение для поиска)
    mask = np.ones((height, width), dtype=np.uint8) * 255

    # Определяем координаты для треугольников в углах
    left_triangle = np.array([[0, 0], [corner_width, 0], [0, corner_height]])
    right_triangle = np.array([[width, 0], [width - corner_width, 0], [width, corner_height]])

    # Закрываем (делаем черными - 0) треугольники в маске, чтобы в этих областях не производился поиск
    cv2.fillPoly(mask, [left_triangle], 0)  # Левый верхний угол
    cv2.fillPoly(mask, [right_triangle], 0)  # Правый верхний угол

    # Исключаем эллипсы из области поиска
    for params in ellipses_params:
        axes, shift_x, shift_y = params
        center_x = width // 2 + shift_x  # Координаты центра эллипса
        center_y = height // 2 + shift_y
        # Рисуем эллипсы на маске, делая их черными (0)
        cv2.ellipse(mask, (center_x, center_y), axes, 0, 0, 360, 0, -1)

    return mask
class VisualOdometry:
    def __init__(self, camera_matrix, dist_coeffs, theta=0.0):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.theta = theta  # Наклон камеры в радианах
        self.R_cam_tilt = R.from_euler('x', theta, degrees=False)
        self.orb = cv2.ORB_create(5000)
        # Используем KNN-сопоставление с тестом Лоу
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.prev_kp = None
        self.prev_des = None
        self.R = R.identity()  # Инициализация начальной ориентации в виде объекта Rotation
        self.t = np.zeros(3)  # Инициализация начальной позиции (вектор формы (3,))
        self.trajectory = []  # Список для сохранения траектории

    def process_frame(self, img, roll, pitch, yaw):
        """
        Обрабатывает один кадр и обновляет позицию и ориентацию камеры
        на основе изменений углов и анализа ключевых точек.

        :param img: Входное изображение
        :param roll: Угол Roll (крен) из таблицы (в радианах)
        :param pitch: Угол Pitch (тангаж) из таблицы (в радианах)
        :param yaw: Угол Yaw (рыскание) из таблицы (в радианах)
        """

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = create_mask(img_gray)

        kp, des = self.orb.detectAndCompute(img_gray, mask=mask)

        if self.prev_des is not None and des is not None:

            matches = self.bf.knnMatch(self.prev_des, des, k=2)


            good_matches = []
            for m, n in matches:
                if m.distance < 0.55 * n.distance:
                    good_matches.append(m)

            if len(good_matches) >= 100:  # Проверяем, что достаточно соответствий для анализа
                prev_pts = np.float32([self.prev_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
                curr_pts = np.float32([kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

                E, mask_E = cv2.findEssentialMat(curr_pts, prev_pts, self.camera_matrix, method=cv2.RANSAC, prob=0.999, threshold=1.0)

                if E is not None and E.shape == (3, 3):
                    _, R_rel_cam, t_rel, mask_pose = cv2.recoverPose(E, curr_pts, prev_pts, self.camera_matrix)

                    R_rel_cam = R.from_matrix(R_rel_cam)


                    R_rel_drone = self.R_cam_tilt.inv() * R_rel_cam.inv() * self.R_cam_tilt

                    self.R = R_rel_drone * self.R


                    t_rel = t_rel.flatten()
                    self.t += self.R.apply(t_rel)


                    new_roll, new_pitch, new_yaw = self.R.as_euler('xyz', degrees=False)

                    if len(self.trajectory) > 0:
                        prev_angles = np.array(self.prev_angles)
                        current_angles = np.array([new_roll, new_pitch, new_yaw])

                        unwrapped_angles = np.unwrap([prev_angles, current_angles], axis=0)[1]


                        self.R = R.from_euler('xyz', unwrapped_angles, degrees=False)

                    self.prev_angles = [new_roll, new_pitch, new_yaw] # Сохраняем текущие углы для следующего шага
                else:
                    print("Не удалось восстановить ориентацию.")
            else:
                print(f"Недостаточно хороших совпадений для текущего кадра.")
        else:
            print("Недостаточно дескрипторов для сопоставления или предыдущие дескрипторы отсутствуют.")

            self.prev_angles = [roll, pitch, yaw]
            self.R = R.from_euler('xyz', self.prev_angles, degrees=False)


        self.prev_kp = kp
        self.prev_des = des


        self.trajectory.append(self.t.copy())

    def get_trajectory(self):
        """
        Возвращает траекторию камеры.
        """
        return np.array(self.trajectory)