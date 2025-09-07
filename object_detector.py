from ultralytics import YOLO
import numpy as np
import cv2
import time
import os

class ObjectDetectorAPI:
    """
    Detector basado en Ultralytics YOLOv8-SEG (PyTorch).
    - Por defecto usa 'yolov8n-seg.pt'.
    - predict(image_bgr) -> (image_with_overlays_bgr, pred_bboxes[N,6], pred_masks:list[np.ndarray(H,W,bool)])
      pred_bboxes: [x1, y1, x2, y2, class_id, score]
      pred_masks: lista de máscaras binarias alineadas a la imagen de entrada
    """

    def __init__(self, weights_path: str = "yolov8n-seg.pt", conf: float = 0.25, iou: float = 0.45, imgsz: int = 640):
        self.model = YOLO(weights_path)
        self.weights_name = os.path.basename(weights_path)  # <- añade esto
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.class_names = self.model.names  # dict: id -> name

    def predict(self, image_bgr: np.ndarray):
        start = time.time()
        results = self.model.predict(
            source=image_bgr,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            verbose=False
        )
        out_img = image_bgr.copy()
        pred_list = []
        pred_masks = []

        for r in results:
            h, w = r.orig_shape  # forma de la imagen original
            # --- cajas ---
            if r.boxes is not None:
                boxes_xyxy = r.boxes.xyxy.cpu().numpy()
                scores = r.boxes.conf.cpu().numpy()
                clsids = r.boxes.cls.cpu().numpy().astype(int)

                for (x1, y1, x2, y2), sc, cid in zip(boxes_xyxy, scores, clsids):
                    pred_list.append([x1, y1, x2, y2, cid, sc])

                    # dibujar caja y etiqueta
                    pt1, pt2 = (int(x1), int(y1)), (int(x2), int(y2))
                    cv2.rectangle(out_img, pt1, pt2, (0, 255, 0), 2)
                    label = f"{self.class_names.get(int(cid), str(int(cid)))} {sc:.2f}"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(out_img, (pt1[0], pt1[1]-th-6), (pt1[0]+tw+2, pt1[1]), (0,255,0), -1)
                    cv2.putText(out_img, label, (pt1[0]+1, pt1[1]-4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)

            # --- máscaras (segmentación) ---
            if r.masks is not None:
                # r.masks.xy es lista de polígonos en coords de la imagen original
                for poly in r.masks.xy:
                    mask = np.zeros((h, w), dtype=np.uint8)
                    pts = [np.round(poly).astype(np.int32)]
                    cv2.fillPoly(mask, pts, 1)
                    pred_masks.append(mask.astype(bool))
                    # overlay rápido (opcional)
                    overlay = out_img.copy()
                    overlay[mask.astype(bool)] = (0.2*out_img[mask.astype(bool)] + 0.8*np.array([0,0,255])).astype(np.uint8)
                    out_img = overlay

        pred_bboxes = np.array(pred_list, dtype=np.float32) if pred_list else np.zeros((0,6), dtype=np.float32)
        # print(f"[YOLOv8-SEG] dets={len(pred_bboxes)}, masks={len(pred_masks)}, {1000*(time.time()-start):.1f} ms")
        return out_img, pred_bboxes, pred_masks
