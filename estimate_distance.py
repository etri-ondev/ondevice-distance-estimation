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

    def Detect(self, weights='runs/train/yolov5s_tf_newcar/weights/best.pt',
           source='data/images',  # file/dir/URL/glob, 0 for webcam
           imgsz=640,  # inference size (pixels)
           conf_thres=0.25,  # confidence threshold
           iou_thres=0.45,  # NMS IOU threshold
           max_det=1,  # maximum detections per image
           device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
           view_img=False,  # show results
           save_txt=False,  # save results to *.txt
           save_conf=False,  # save confidences in --save-txt labels
           save_crop=False,  # save cropped prediction boxes
           nosave=False,  # do not save images/videos
           classes=None,  # filter by class: --class 0, or --class 0 2 3
           agnostic_nms=False,  # class-agnostic NMS
           augment=False,  # augmented inference
           update=False,  # update all models
           project='inference/output',  # save results to project/name
           name='exp',  # save results to project/name
           exist_ok=False,  # existing project/name ok, do not increment
           line_thickness=3,  # bounding box thickness (pixels)
           hide_labels=False,  # hide labels
           hide_conf=False,  # hide confidences
           half=False,  # use FP16 half-precision inference
           ):
        save_img = not nosave and not source.endswith('.txt')  # save inference images
        webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
            ('rtsp://', 'rtmp://', 'http://', 'https://'))

        #save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        #(save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
        save_dir = Path(project)

        # Initialize
        set_logging()
        device = select_device(device)
        half &= device.type != 'cpu'  # half precision is only supported on CUDA

        # Load model
        model = attempt_load(weights, device=device)  # load FP32 model
        stride = max(int(model.stride.max()), 32)  # model stride
        names = model.module.names if hasattr(model, "module") else model.names  # get class names

        if half:
            model.half()  # to FP16

        # Set Dataloader
        vid_path, vid_writer = None, None
        if webcam:
            view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz, stride=stride)
        else:
            dataset = LoadImages(source, img_size=imgsz, stride=stride)

        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        t0 = time.time()
        for path, img, im0s, vid_cap, _ in dataset:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_sync()
            pred = model(img, augment=augment)[0]

            # Apply NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            t2 = time_sync()

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if webcam:  # batch_size >= 1
                    p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count  # path[i] is the source, i.e., 0
                else:
                    p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

                p = Path(p)  # convert to Path; p may be inference/images/demo_distance.mp4
                save_path = str(save_dir / p.name)  # img.jpg  inference/output/demo_distance.mp4
                txt_path = str(save_dir / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt   inference/output/demo_distance_frame
                #print('txt', txt_path)
                s += '%gx%g ' % img.shape[2:]  # print image shape, e.g., 640x480
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if save_crop else im0  # for save_crop
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_boxes_compat(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        if not names[int(c)] in ['person', 'car', 'truck', 'bicycle', 'motorcycle', 'bus', 'traffic light', 'stop sign']:
                            continue
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if not names[int(cls)] in ['person','chair', 'car', 'truck', 'bicycle', 'motorcycle', 'bus', 'traffic light', 'stop sign']:
                            continue
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                        kuang = [int(cls), xywh[0], xywh[1], xywh[2], xywh[3]]
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * 5 + '\n') % (int(cls), *xywh))

                        distance, d = self.distance(kuang)

                        if save_img or save_crop or view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            if label != None and distance!=0:
                                label = label + ' ' + str('%.1f' % d[0]) + 'm'+ str('%.1f' % d[1]) + 'm'
                            plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=line_thickness)
                            if save_crop:
                                save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

                # Print time (inference + NMS)
                print(f'{s}Done. ({t2 - t1:.3f}s)')

                # Stream results
                if view_img:
                    cv2.imshow(str(p), im0)
                    cv2.waitKey(1)  # 1 millisecond

                # Save results (image with detections)
                if save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                    else:  # 'video' or 'stream'
                        if vid_path != save_path:  # new video
                            vid_path = save_path
                            if isinstance(vid_writer, cv2.VideoWriter):
                                vid_writer.release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                                save_path += '.mp4'
                            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer.write(im0)

        if save_txt or save_img:
            s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
            print(f"Results saved to {save_dir}{s}")

        if update:
            strip_optimizer(weights)  # update model (to fix SourceChangeWarning)

        print(f'Done. ({time.time() - t0:.3f}s)')

#######################################################################################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str,
                        default='runs/train/yolov5s_tf_newcar/weights/best.pt',
                        help='model.pt path(s)')
    # parser.add_argument('--source', type=str, default='inference/images/demo_distance.mp4', help='inference/dir/URL/glob, 0 for webcam')



    # parser.add_argument('--source', type=str, default='inference/inputs/11_20', help='inference/dir/URL/glob, 0 for webcam')
    parser.add_argument('--source', type=str, default='inference/inputs/test4', help='inference/dir/URL/glob, 0 for webcam')


    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save_txt',default=False, action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='inference/output', help='save results to project/name')  # save directory
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    opt = parser.parse_args()
    print(opt)
    check_requirements(exclude=('tensorboard', 'thop'))

    print('Starting object detection and monocular distance estimation!')
    DE = DistanceEstimation()
    DE.Detect(**vars(opt))
    if time.time()>(start_time + 10):
        cv2.waitKey (0)
        cv2.destroyAllWindows()
