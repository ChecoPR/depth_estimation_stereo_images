import os

import cv2
import time
import argparse
import glob
import numpy as np
import torch
from tqdm import tqdm
import open3d as o3d
import argparse, sys

from utils import get_calibration_parameters, calc_depth_map, find_distances, add_depth, Open3dVisualizer, write_ply
from object_detector import ObjectDetectorAPI
from disparity_estimator.raftstereo_disparity_estimator import RAFTStereoEstimator
from disparity_estimator.fastacv_disparity_estimator import FastACVEstimator
from disparity_estimator.bgnet_disparity_estimator import BGNetEstimator
from disparity_estimator.gwcnet_disparity_estimator import GwcNetEstimator
from disparity_estimator.pasmnet_disparity_estimator import PASMNetEstimator
from disparity_estimator.crestereo_disparity_estimator import CREStereoEstimator
from disparity_estimator.psmnet_disparity_estimator import PSMNetEstimator
from disparity_estimator.hitnet_disparity_estimator import HitNetEstimator
import config
import wandb
import csv

ap = argparse.ArgumentParser(
    description="Demo de estimación de profundidad con distintos modelos de disparidad"
)

ap.add_argument(
    "--arch", 
    type=str, 
    default=config.ARCHITECTURE, 
    choices=["raft-stereo", "psmnet", "hitnet", "gwcnet", "bgnet", "pasmnet", "crestereo", "fastacv-plus"],
    help="Selecciona la arquitectura de disparidad a usar. Opciones: raft-stereo | psmnet | hitnet | gwcnet | bgnet | pasmnet | crestereo | fastacv-plus"
)

# 👇 clave: no romper parsers de libs (RAFT, etc.)
args, unknown = ap.parse_known_args()
config.ARCHITECTURE = args.arch

# deja en sys.argv SOLO los args que no procesaste, para que los consuma la librería
sys.argv = [sys.argv[0]] + unknown


def demo():
    if config.PROFILE_FLAG:
        disp_estimator = None
        if config.ARCHITECTURE == 'raft-stereo':
            disp_estimator = RAFTStereoEstimator()
        elif config.ARCHITECTURE == 'fastacv-plus':
            disp_estimator = FastACVEstimator()
        elif config.ARCHITECTURE == 'bgnet':
            disp_estimator = BGNetEstimator()
        elif config.ARCHITECTURE == 'gwcnet':
            disp_estimator = GwcNetEstimator()
        elif config.ARCHITECTURE == 'pasmnet':
            disp_estimator = PASMNetEstimator()
        elif config.ARCHITECTURE == 'crestereo':
            disp_estimator = CREStereoEstimator()
        elif config.ARCHITECTURE == 'psmnet':
            disp_estimator = PSMNetEstimator()
        elif config.ARCHITECTURE == 'hitnet':
            disp_estimator = HitNetEstimator()
        disp_estimator.profile()
        exit()

    left_images = sorted(glob.glob(config.KITTI_LEFT_IMAGES_PATH, recursive=True))
    right_images = sorted(glob.glob(config.KITTI_RIGHT_IMAGES_PATH, recursive=True))
    calib_files = sorted(glob.glob(config.KITTI_CALIB_FILES_PATH, recursive=True))
    index = 0
    init_open3d = False
    disp_estimator = None
    print("Disparity Architecture Used: {} ".format(config.ARCHITECTURE))
    if config.ARCHITECTURE == 'raft-stereo':
        disp_estimator = RAFTStereoEstimator()
    elif config.ARCHITECTURE == 'fastacv-plus':
        disp_estimator = FastACVEstimator()
    elif config.ARCHITECTURE == 'bgnet':
         disp_estimator = BGNetEstimator()
    elif config.ARCHITECTURE == 'gwcnet':
        disp_estimator = GwcNetEstimator()
    elif config.ARCHITECTURE == 'pasmnet':
        disp_estimator = PASMNetEstimator()
    elif config.ARCHITECTURE == 'crestereo':
        disp_estimator = CREStereoEstimator()
    elif config.ARCHITECTURE == 'psmnet':
        disp_estimator = PSMNetEstimator()
    elif config.ARCHITECTURE == 'hitnet':
        disp_estimator = HitNetEstimator()

    if config.SHOW_DISPARITY_OUTPUT:
        window_name = "Estimated depth with {}".format(config.ARCHITECTURE)
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # directorios de salida
    base_out = "output"
    vis_dir = os.path.join(base_out, "vis")
    kitti_dir = os.path.join(base_out, "kitti_preds", config.ARCHITECTURE)
    stats_dir = os.path.join(base_out, "stats")
    csv_path = os.path.join(stats_dir, f"{config.ARCHITECTURE}.csv")
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(kitti_dir, exist_ok=True)
    os.makedirs(stats_dir, exist_ok=True)

    obj_det = ObjectDetectorAPI()  # crea UNA vez, no dentro del bucle
    detector_name = getattr(obj_det, "weights_name", "yolov8n-seg.pt")  # ver paso 2
    print(f"Disparity Architecture Used: {config.ARCHITECTURE}")
    print(f"Detector: {detector_name}")

    wandb.init(
        project="stereo-kitti",
        name=f"{config.ARCHITECTURE}__{detector_name}",
        config={
            "arch": config.ARCHITECTURE,
            "detector": detector_name,
            "device": config.DEVICE,
            "imgsz": 640,
            "conf": 0.25,
            "iou": 0.45,
        },
        tags=[config.ARCHITECTURE, detector_name, "kitti"]
    )

    for (imfile1, imfile2, calib_file) in tqdm(list(zip(left_images, right_images, calib_files))):
        img = cv2.imread(imfile1)
        parameters = get_calibration_parameters(calib_file)
        start = time.time()
        
        result, pred_bboxes, pred_masks = obj_det.predict(img)
        end = time.time()
        elapsed_time = (end - start) * 1000
        print("Evaluation Time for Object Detection with YOLO is : {} ms ".format(elapsed_time))

        print(pred_bboxes)

        start_d = time.time()
        disparity_map = disp_estimator.estimate(imfile1, imfile2)
        # Si es lista (p. ej., pirámide), quédate con el último nivel (o el primero, según prefieras)
        if isinstance(disparity_map, list) or isinstance(disparity_map, tuple):
            # elige el mapa de mayor resolución; prueba -1 o 0 según tu implementación
            disparity_map = disparity_map[-1]

        # Si es tensor de PyTorch, pásalo a numpy 2D
        if isinstance(disparity_map, torch.Tensor):
            # quita batch/canal si existen
            disparity_map = disparity_map.detach().cpu().squeeze().numpy()

        # Asegura tipo float32 y 2D
        disparity_map = np.asarray(disparity_map).astype(np.float32)
        if disparity_map.ndim == 3:
            # si viene HxWx1
            if disparity_map.shape[-1] == 1:
                disparity_map = disparity_map[..., 0]
            else:
                # si por alguna razón trae canales, elige uno o promedio
                disparity_map = disparity_map.mean(axis=-1)

        print("disparity_map shape:", disparity_map.shape)

        end_d = time.time()
        elapsed_time_d = (end_d - start_d) * 1000
        print("Evaluation Time for Disparity Estimation with {} is : {} ms ".format(config.ARCHITECTURE, elapsed_time_d))

        print("disparity_map: {}".format(disparity_map.shape))
        disparity_left = disparity_map

        k_left = parameters[0]
        t_left = parameters[1]
        p_left = parameters[2]

        k_right = parameters[3]
        t_right = parameters[4]
        p_right = parameters[5]
        print("k_left:{}, t_left:{}".format(k_left, t_left))
        print("k_right:{}, t_right:{}".format(k_right, t_right))
        print("p_left:{}, p_right:{}".format(p_left, p_right))

        disp_float   = disparity_map.astype(np.float32)  # mantener float
        depth_map    = calc_depth_map(disp_float, k_left, t_left, t_right)
        disp_kitti16 = (disp_float * 256.).astype(np.uint16)  # para guardar/estilo KITTI

        disparity_map = (disp_float * 256.).astype(np.uint16)
        color_depth = cv2.applyColorMap(cv2.convertScaleAbs(disp_kitti16, alpha=0.01), cv2.COLORMAP_JET)

        pred_bboxes_norm = pred_bboxes.copy()
        if pred_bboxes_norm.size:
            h_img, w_img = img.shape[:2]
            x1 = pred_bboxes_norm[:, 0]
            y1 = pred_bboxes_norm[:, 1]
            x2 = pred_bboxes_norm[:, 2]
            y2 = pred_bboxes_norm[:, 3]

            cx = ((x1 + x2) / 2.0) / w_img
            cy = ((y1 + y2) / 2.0) / h_img
            w  = (x2 - x1) / w_img
            h  = (y2 - y1) / h_img

            pred_bboxes_norm[:, 0] = cx
            pred_bboxes_norm[:, 1] = cy
            pred_bboxes_norm[:, 2] = w
            pred_bboxes_norm[:, 3] = h

        # 2) Asegura que la "imagen de referencia" tenga el MISMO tamaño que depth_map,
        #    porque find_distances usa h,w de la imagen para construir los índices, pero
        #    indexa sobre depth_map.
        img_for_depth = cv2.resize(img, (depth_map.shape[1], depth_map.shape[0]))

        depth_list = find_distances(depth_map, pred_bboxes_norm, img_for_depth, method="center")

        # Asegura que las máscaras coincidan con el tamaño de depth_map
        # --- masks para depth y para imagen ---
        H_d, W_d = depth_map.shape[:2]
        h_img, w_img = img.shape[:2]

        masks_depth = []
        masks_img = []
        for m in pred_masks:
            # a tamaño del depth_map
            if m.shape != (H_d, W_d):
                m_d = cv2.resize(m.astype(np.uint8), (W_d, H_d), interpolation=cv2.INTER_NEAREST).astype(bool)
            else:
                m_d = m.astype(bool)
            masks_depth.append(m_d)

            # a tamaño de la imagen
            if m.shape != (h_img, w_img):
                m_i = cv2.resize(m.astype(np.uint8), (w_img, h_img), interpolation=cv2.INTER_NEAREST).astype(bool)
            else:
                m_i = m.astype(bool)
            masks_img.append(m_i)

        # profundidad por máscara (usa masks_depth)
        depth_per_mask = []
        for m in masks_depth:
            vals = depth_map[m]
            vals = vals[np.isfinite(vals)]
            depth_per_mask.append(float(np.median(vals)) if vals.size > 0 else float('nan'))
        print("Depth per mask:", depth_per_mask)

        for i, d in enumerate(depth_per_mask):
            if i < pred_bboxes.shape[0] and np.isfinite(d):
                x1, y1, x2, y2, cid, sc = pred_bboxes[i]
                pt = (int(x1), max(0, int(y1)-8))
                cv2.putText(result, f"{d:.2f} m", pt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)


        res = add_depth(depth_list, result, pred_bboxes_norm)
        print("img.shape {}".format(img.shape))
        print("color_depth.shape {}".format(color_depth.shape))
        print("res.shape {}".format(res.shape))
        h = img.shape[0]
        w = img.shape[1]
        color_depth = cv2.resize(color_depth, (w, h))
        print("color_depth.shape after resize {}".format(color_depth.shape))
        combined_image = np.vstack((color_depth, res))
        if config.SHOW_DISPARITY_OUTPUT:
            cv2.imshow(window_name, combined_image)
        if config.SHOW_3D_PROJECTION:
            if init_open3d == False:
                w = img.shape[1]
                h = img.shape[0]
                print("w:{}, h: {}".format(w, h))
                print("kleft[0][0]: {}".format(k_left[0][0]))
                print("kleft[1][2]: {}".format(k_left[1][1]))
                print("kleft[1][2]: {}".format(k_left[0][2]))
                print("kleft[1][2]: {}".format(k_left[1][2]))
                print("kLeft: {}".format(k_left))

                K = o3d.camera.PinholeCameraIntrinsic(width=w,
                                                      height=h,
                                                      fx=k_left[0, 0],
                                                      fy=k_left[1, 1],
                                                      cx=k_left[0][2],
                                                      cy=k_left[1][2])
                open3dVisualizer = Open3dVisualizer(K)
                init_open3d = True
            open3dVisualizer(img, depth_map * 1000)

            o3d_screenshot_mat = open3dVisualizer.vis.capture_screen_float_buffer()
            o3d_screenshot_mat = (255.0 * np.asarray(o3d_screenshot_mat)).astype(np.uint8)
            o3d_screenshot_mat = cv2.cvtColor(o3d_screenshot_mat, cv2.COLOR_RGB2BGR)
        if config.SAVE_POINT_CLOUD:
            # Calculate depth-to-disparity
            cam1 = k_left  # left image - P2
            cam2 = k_right  # right image - P3

            print("p_left: {}".format(p_left))
            print("cam1:{}".format(cam1))

            Tmat = np.array([0.54, 0., 0.])
            Q = np.zeros((4, 4))
            cv2.stereoRectify(cameraMatrix1=cam1, cameraMatrix2=cam2,
                              distCoeffs1=0, distCoeffs2=0,
                              imageSize=img.shape[:2],
                              R=np.identity(3), T=Tmat,
                              R1=None, R2=None,
                              P1=None, P2=None, Q=Q)

            print("Disparity To Depth")
            print(Q)
            print("disparity_left.shape: {}".format(disparity_left.shape))
            print("disparity_left: {}".format(disparity_left))

            points = cv2.reprojectImageTo3D(disparity_left.copy(), Q)
            # reflect on x axis

            reflect_matrix = np.identity(3)
            reflect_matrix[0] *= -1
            points = np.matmul(points, reflect_matrix)

            img_left = cv2.imread(imfile1)
            colors = cv2.cvtColor(img_left.copy(), cv2.COLOR_BGR2RGB)
            print("colors.shape: {}".format(colors.shape))
            disparity_left = cv2.resize(disparity_left, (colors.shape[1], colors.shape[0]))
            points = cv2.resize(points, (colors.shape[1], colors.shape[0]))
            print("points.shape: {}".format(points.shape))
            print("After mod. disparity_left.shape: {}".format(disparity_left.shape))
            # filter by min disparity
            mask = disparity_left > disparity_left.min()
            out_points = points[mask]
            out_colors = colors[mask]

            out_colors = out_colors.reshape(-1, 3)
            path_ply = os.path.join("output/point_clouds/", config.ARCHITECTURE)
            isExist = os.path.exists(path_ply)
            if not isExist:
                os.makedirs(path_ply)
            print("path_ply: {}".format(path_ply))

            file_name = path_ply + "/" +str(index) + ".ply"
            print("file_name: {}".format(file_name))
            write_ply(file_name, out_points, out_colors)
            index = index + 1
        if config.SHOW_DISPARITY_OUTPUT:
            if cv2.waitKey(1) == ord('q'):
                break

        base_name = os.path.splitext(os.path.basename(imfile1))[0]
        # 2.1 Visualizaciones
        cv2.imwrite(os.path.join(vis_dir, f"{base_name}_disp_color.png"), color_depth)
        cv2.imwrite(os.path.join(vis_dir, f"{base_name}_detections.png"), res)

        # overlay de máscaras
        if pred_masks:
            mask_overlay = img.copy()
            alpha = 0.7
            color = np.array([0, 0, 255], dtype=np.uint8)  # BGR rojo

            for m in masks_img:
                if m.any():
                    mask_overlay[m] = ((1 - alpha) * mask_overlay[m] + alpha * color).astype(np.uint8)

            cv2.imwrite(os.path.join(vis_dir, f"{base_name}_masks.png"), mask_overlay)

        # 2.2 Disparidad en formato KITTI (16-bit)
        cv2.imwrite(os.path.join(kitti_dir, f"{base_name}.png"), disp_kitti16)

        csv_path = os.path.join(stats_dir, f"{config.ARCHITECTURE}.csv")

        row = {
            "image": base_name,
            "arch": config.ARCHITECTURE,
            "estimator_class": type(disp_estimator).__name__,  # p. ej. PSMNetEstimator
            "detector": detector_name,                         # p. ej. yolov8n-seg.pt
            "device": config.DEVICE,
            "yolo_ms": round(elapsed_time, 2),
            "disp_ms": round(elapsed_time_d, 2),
            "num_dets": int(pred_bboxes.shape[0]),
            "disp_h": int(disp_float.shape[0]),
            "disp_w": int(disp_float.shape[1]),
            "disp_min": float(np.nanmin(disp_float)),
            "disp_max": float(np.nanmax(disp_float)),
            "disp_mean": float(np.nanmean(disp_float)),
            "depth_center": depth_list,       # lista → quedará como string; suficiente para auditoría
            "depth_mask": depth_per_mask,     # idem
        }
        write_header = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(row)

        wandb.log({
            "yolo_ms": elapsed_time,
            "disp_ms": elapsed_time_d,
            "num_dets": int(pred_bboxes.shape[0]),
            "disp_min": float(np.nanmin(disp_float)),
            "disp_max": float(np.nanmax(disp_float)),
            "disp_mean": float(np.nanmean(disp_float)),
            "disp_shape_h": int(disp_float.shape[0]),
            "disp_shape_w": int(disp_float.shape[1]),
            "disp_color": wandb.Image(color_depth, caption="Disparity (JET)"),
            "detections": wandb.Image(res, caption="Detections"),
        })
        

    if config.SHOW_DISPARITY_OUTPUT:
        cv2.destroyAllWindows()

    # artefactos tras el for
    art = wandb.Artifact(f"kitti_preds_{config.ARCHITECTURE}", type="predictions")
    art.add_dir(kitti_dir)
    wandb.log_artifact(art)

    art2 = wandb.Artifact(f"vis_{config.ARCHITECTURE}", type="images")
    art2.add_dir(vis_dir)
    wandb.log_artifact(art2)

    art3 = wandb.Artifact(f"stats_{config.ARCHITECTURE}", type="stats")
    art3.add_file(csv_path)
    wandb.log_artifact(art3)
    wandb.finish()


if __name__ == '__main__':
    demo()


