# tools/sanitize_cfg.py
from pathlib import Path

SRC = Path("Yolov4/yolov4-tiny.cfg")          # tu cfg original
DST = Path("Yolov4/yolov4-tiny-clean.cfg")    # cfg limpio

# Claves comunes que rompen en esa lib:
DROP_KEYS = {
    "subdivisions", "batch", "momentum", "decay", "angle",
    "saturation", "exposure", "hue", "learning_rate", "burn_in",
    "max_batches", "policy", "steps", "scales", "power",
    # yolo heads / augmentación
    "jitter", "random", "ignore_thresh", "truth_thresh", "scale_x_y",
    "cls_normalizer", "iou_normalizer", "obj_normalizer", "label_smooth_eps",
    "mosaic", "mixup", "letter_box"
}

lines = SRC.read_text().splitlines()
clean = []
for ln in lines:
    s = ln.strip()
    if not s or s.startswith("#") or s.startswith(";"):
        clean.append(ln); continue
    if "=" in s:
        k = s.split("=",1)[0].strip().lower()
        if k in DROP_KEYS:
            continue
    clean.append(ln)

# Asegura que [net] tenga lo mínimo requerido
out = []
in_net = False
for ln in clean:
    if ln.strip().startswith("["):
        in_net = (ln.strip().lower() == "[net]")
        out.append(ln); continue
    if in_net:
        # Mantén solo width/height/channels (ajústalos si quieres)
        key = ln.split("=",1)[0].strip().lower() if "=" in ln else ""
        if key in {"width","height","channels"} or not "=" in ln:
            out.append(ln)
    else:
        out.append(ln)

DST.write_text("\n".join(out))
print(f"Escrito cfg limpio en: {DST}")