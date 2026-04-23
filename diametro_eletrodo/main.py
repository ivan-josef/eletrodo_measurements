import cv2
import numpy as np
import os


class SubPixelEdge:
    def __init__(self, path):
        self.path = path
        self.files = os.listdir(self.path)
        self.index = 0
        self.window = 'eletrodo'

        # criar janela de controle
        cv2.namedWindow("controls")

        # trackbars
        cv2.createTrackbar("blur", "controls", 1, 10, lambda x: None)
        cv2.createTrackbar("canny1", "controls", 50, 255, lambda x: None)
        cv2.createTrackbar("canny2", "controls", 150, 255, lambda x: None)

        cv2.createTrackbar("hough_param2", "controls", 30, 100, lambda x: None)
        cv2.createTrackbar("minR", "controls", 50, 500, lambda x: None)
        cv2.createTrackbar("maxR", "controls", 300, 800, lambda x: None)

        cv2.createTrackbar("roi_min_%", "controls", 60, 100, lambda x: None)
        cv2.createTrackbar("roi_max_%", "controls", 95, 100, lambda x: None)

        cv2.createTrackbar("rays", "controls", 360, 1000, lambda x: None)

    def get_params(self):
        blur = cv2.getTrackbarPos("blur", "controls")
        blur = max(1, blur)
        if blur % 2 == 0:
            blur += 1

        params = {
            "blur": blur,
            "canny1": cv2.getTrackbarPos("canny1", "controls"),
            "canny2": cv2.getTrackbarPos("canny2", "controls"),
            "hough_param2": cv2.getTrackbarPos("hough_param2", "controls"),
            "minR": cv2.getTrackbarPos("minR", "controls"),
            "maxR": cv2.getTrackbarPos("maxR", "controls"),
            "roi_min": cv2.getTrackbarPos("roi_min_%", "controls") / 100.0,
            "roi_max": cv2.getTrackbarPos("roi_max_%", "controls") / 100.0,
            "rays": max(10, cv2.getTrackbarPos("rays", "controls")),
        }

        return params

    def preprocess(self, img, blur):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur_img = cv2.GaussianBlur(gray, (blur, blur), 0)
        return blur_img

    def detect_outer_circle(self, img, params):
        edges = cv2.Canny(img, params["canny1"], params["canny2"])

        circles = cv2.HoughCircles(
            img,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=100,
            param1=100,
            param2=params["hough_param2"],
            minRadius=params["minR"],
            maxRadius=params["maxR"]
        )

        if circles is not None:
            circles = np.uint16(np.around(circles))
            c = circles[0][0]
            return (c[0], c[1], c[2]), edges

        return None, edges

    def create_annular_mask(self, shape, cx, cy, r_ext, params):
        h, w = shape

        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((X - cx)**2 + (Y - cy)**2)

        r_min = params["roi_min"] * r_ext
        r_max = params["roi_max"] * r_ext

        mask = (dist >= r_min) & (dist <= r_max)
        return mask.astype(np.uint8) * 255

    def radial_sampling(self, img, cx, cy, r_ext, params):
        h, w = img.shape
        points = []

        for theta in np.linspace(0, 2*np.pi, params["rays"]):
            for r in np.linspace(params["roi_min"]*r_ext,
                                 params["roi_max"]*r_ext, 80):

                x = int(cx + r * np.cos(theta))
                y = int(cy + r * np.sin(theta))

                if 0 <= x < w and 0 <= y < h:
                    points.append((x, y))

        return points

    def draw(self, img, circle, edges, mask, points):
        vis = img.copy()

        # edges
        edges_col = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        vis = cv2.addWeighted(vis, 0.7, edges_col, 0.3, 0)

        # círculo
        if circle:
            cx, cy, r = circle
            cv2.circle(vis, (cx, cy), r, (0, 255, 0), 2)
            cv2.circle(vis, (cx, cy), 3, (0, 0, 255), -1)

        # máscara
        colored_mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
        vis = cv2.addWeighted(vis, 0.8, colored_mask, 0.2, 0)

        # pontos
        for (x, y) in points[::100]:
            cv2.circle(vis, (x, y), 1, (255, 0, 0), -1)

        return vis

    def process(self, img):
        params = self.get_params()

        pre = self.preprocess(img, params["blur"])

        circle, edges = self.detect_outer_circle(pre, params)

        if circle is None:
            return img

        cx, cy, r = circle

        mask = self.create_annular_mask(pre.shape, cx, cy, r, params)

        points = self.radial_sampling(pre, cx, cy, r, params)

        vis = self.draw(img, circle, edges, mask, points)

        return vis

    def run(self):
        while True:
            path = os.path.join(self.path, self.files[self.index])
            img = cv2.imread(path)

            if img is None:
                continue

            img = cv2.resize(img, (1280, 720))

            result = self.process(img)

            cv2.imshow(self.window, result)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('d'):
                self.index += 1
            elif key == ord('a'):
                self.index -= 1
            elif key == ord('q'):
                break

            self.index = max(0, min(self.index, len(self.files) - 1))


path = 'diametro_eletrodo/images'
obj = SubPixelEdge(path)
obj.run()
cv2.destroyAllWindows()