import argparse
import torch.backends.cudnn as cudnn
# Compatibility helper for YOLOv5 bounding-box rescaling.
# Old YOLOv5 used scale_coords() for xyxy boxes; newer versions use scale_boxes().
# This changes only the API name, not the intended box-coordinate calculation.
try:
    from utils.general import scale_boxes as scale_boxes_compat
except ImportError:
    try:
        from utils.general import scale_coords as scale_boxes_compat
    except ImportError:
        from ultralytics.utils.ops import scale_boxes as scale_boxes_compat
from models.experimental import attempt_load
from utils.dataloaders import LoadStreams, LoadImages
from utils.general import *
from utils.plots import *
from utils.torch_utils import *
import plotly.graph_objects as go
# ============================================================
# Compatibility function for old YOLOv5 plot_one_box()
# Only used for visualization.
# It does NOT change detection or distance estimation results.
# ============================================================
def plot_one_box(x, im, color=(128, 128, 128), label=None, line_thickness=3):
    """
    Draw one bounding box on an image.

    Args:
        x: bounding box coordinates [x1, y1, x2, y2]
        im: OpenCV image
        color: bounding box color
        label: label text
        line_thickness: bounding box thickness
    """

    tl = (
        line_thickness
        or round(0.002 * (im.shape[0] + im.shape[1]) / 2) + 1
    )

    # Convert tensor coordinates to integers
    if isinstance(x[0], torch.Tensor):
        x = [int(v.item()) for v in x]
    else:
        x = [int(v) for v in x]

    c1 = (x[0], x[1])
    c2 = (x[2], x[3])

    # Draw bounding box
    cv2.rectangle(
        im,
        c1,
        c2,
        color,
        thickness=tl,
        lineType=cv2.LINE_AA
    )

    # Draw label
    if label:
        tf = max(tl - 1, 1)

        t_size = cv2.getTextSize(
            label,
            0,
            fontScale=tl / 3,
            thickness=tf
        )[0]

        outside = c1[1] - t_size[1] - 3 >= 0

        c2_label = (
            c1[0] + t_size[0],
            c1[1] - t_size[1] - 3
            if outside
            else c1[1] + t_size[1] + 3
        )

        cv2.rectangle(
            im,
            c1,
            c2_label,
            color,
            -1,
            cv2.LINE_AA
        )

        cv2.putText(
            im,
            label,
            (
                c1[0],
                c1[1] - 2
                if outside
                else c1[1] + t_size[1] + 2
            ),
            0,
            tl / 3,
            (255, 255, 255),
            thickness=tf,
            lineType=cv2.LINE_AA
        )

start_time = time.time()
print('Pandas Version:', pd.__version__)
print('Nunpy Version:', np.__version__)


@torch.no_grad()
class DistanceEstimation:
    def __init__(self):
        self.W = 3000
        self.H = 4000
        self.excel_path = r'camera_parameters.xlsx'

    def camera_parameters(self, excel_path):
        df_intrinsic = pd.read_excel(excel_path, sheet_name='내부 파라미터', header=None)
        df_p = pd.read_excel(excel_path, sheet_name='외부 파라미터', header=None)

        print('Extrinsic matrix shape:', df_p.values.shape)
        print('Intrinsic matrix shape:', df_intrinsic.values.shape)

        return df_p.values, df_intrinsic.values

    def object_point_world_position(self, u, v, w, h, p, k):
        u1 = u
        v1 = v + h / 2
        print('Key point coordinates:', u1, v1)


        fx = k[0, 0]
        fy = k[1, 1]
        H = 0.8
        angle_a = 0.005 #0.005 #0.01 ######E2:0.01 E4:0.005
        angle_b = math.atan((v1 - self.H / 2) / fy)
        angle_c = angle_b + angle_a
        algorithm_2 = H/math.tan(angle_c)
        print('Algorithm_2_distance:', algorithm_2)
        print('angle_b', angle_b)

        depth = (H / np.sin(angle_c)) * math.cos(angle_b)
        print('depth', depth)

        k_inv = np.linalg.inv(k)
        p_inv = np.linalg.inv(p)
        # print(p_inv)
        point_c = np.array([u1, v1, 1])
        point_c = np.transpose(point_c)
        print('point_c', point_c)
        print('k_inv', k_inv)
        c_position = np.matmul(k_inv, depth * point_c)
        print('c_position', c_position)
        c_position = np.append(c_position, 1)
        c_position = np.transpose(c_position)
        c_position = np.matmul(p_inv, c_position)
        d1 = np.array((c_position[0], c_position[1]), dtype=float)
        return d1

    def distance(self, kuang, xw=5, yw=0.1):
        print('=' * 50)
        print('Starting distance estimation')
        fig = go.Figure()
        p, k = self.camera_parameters(self.excel_path)
        if len(kuang):
            obj_position = []
            u, v, w, h = kuang[1] * self.W, kuang[2] * self.H, kuang[3] * self.W, kuang[4] * self.H
            print('Bounding box:', u, v, w, h)
            d1 = self.object_point_world_position(u, v, w, h, p, k)
        distance = 0
        print('Distance coordinates:', d1)
        if d1[0] <= 0:
            d1[:] = 0
        else:
            distance = math.sqrt(math.pow(d1[0], 2) + math.pow(d1[1], 2))
        return distance, d1
