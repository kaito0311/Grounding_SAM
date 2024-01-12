import os 

import cv2
import numpy as np
import supervision as sv
import argparse
import torch
import torchvision
from tqdm import tqdm 

from groundingdino.util.inference import Model
from segment_anything import SamPredictor
from EfficientSAM.MobileSAM.setup_mobile_sam import setup_model

def get_object_instance(ls_mask, image):

    ls_mask = ls_mask 

    ls_image = []

    for idx, mask in enumerate(ls_mask):
        print(mask.shape)
        print(image.shape)
        print(np.max(mask))
        print(np.min(mask))

        exit()
        ls_row = np.where(np.sum(mask, axis=1) > 0)[0]
        if len(ls_row) == 0: 
            ls_image.append(None)
            continue 
        start_row, end_row = ls_row[0], ls_row[-1]
        ls_col = np.where(np.sum(mask, axis=0) > 0)[0]
        if len(ls_col) == 0: 
            ls_image.append(None)
            continue
        start_col, end_col = ls_col[0], ls_col[-1]

        mask = np.expand_dims(mask, axis=2)
        mask = np.repeat(mask, 3, axis=2)
        # print(mask.shape)
        assert mask.shape == image.shape
        cut_image = np.concatenate([image[start_row:end_row, start_col:end_col, :], mask[:, :, :1][start_row:end_row, start_col:end_col, :]], axis= 2)
        ls_image.append(
            np.copy(cut_image)
        )
    return ls_image


def get_full_size_mask(ls_mask, image):
    '''
    Return mask has same shape with image
    '''
    list_mask = [] 
    for idx, mask in enumerate(ls_mask):
        mask = np.expand_dims(mask, 2)
        mask = np.repeat(mask, 3, axis=2)
        print("mask shape: ", mask)
        mask = np.array(mask * 255.0, dtype= np.uint8)
        list_mask.append(mask.copy()) 
    
    return list_mask


def process_segment(path_image, output_dir = "output_dir"): 




   # load image
    image = cv2.imread(path_image)

    # detect objects
    detections = grounding_dino_model.predict_with_classes(
        image=image,
        classes=CLASSES,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD
    )

    # annotate image with detections
    box_annotator = sv.BoxAnnotator()
    labels = [
        f"{CLASSES[class_id]} {confidence:0.2f}"
        for _, _, confidence, class_id, _
        in detections]
    annotated_frame = box_annotator.annotate(
        scene=image.copy(), detections=detections, labels=labels)

    nms_idx = torchvision.ops.nms(
        torch.from_numpy(detections.xyxy),
        torch.from_numpy(detections.confidence),
        NMS_THRESHOLD
    ).numpy().tolist()

    detections.xyxy = detections.xyxy[nms_idx]
    detections.confidence = detections.confidence[nms_idx]
    detections.class_id = detections.class_id[nms_idx]

    # Prompting SAM with detected boxes
    def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
        sam_predictor.set_image(image)
        result_masks = []
        for box in xyxy:
            masks, scores, logits = sam_predictor.predict(
                box=box,
                multimask_output=True
            )
            index = np.argmax(scores)
            result_masks.append(masks[index])
        return np.array(result_masks)

    # convert detections to masks
    detections.mask = segment(
        sam_predictor=sam_predictor,
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        xyxy=detections.xyxy
    )

    ls_occlu_image = get_object_instance(
        detections.mask,
        image= image
    )
    # list_mask = get_full_size_mask(
    #     detections.mask, 
    #     image= image
    # )

    os.makedirs(output_dir, exist_ok=True)
    ls_idx_classes = detections.class_id.tolist()
    if len(ls_idx_classes) == 0: 
        return 

    count = 0 

    for object_img, idx_class in zip(ls_occlu_image, ls_idx_classes):
        if object_img is None: 
            continue
        base_name = os.path.basename(path_image).split(".")[0] + "_" +  str(CLASSES[idx_class]) + "_" + str(count) + ".npy"
        if np.min(object_img.shape[:2]) < 48:
            continue  
        # np.save(
        #     os.path.join(output_dir, base_name),
        #     object_img
        # )
        cv2.imwrite(
            os.path.join(output_dir, base_name[:-4] + ".jpg"),
            object_img
        )
        count +=  1




# Predict classes and hyper-param for GroundingDINO
SOURCE_IMAGE_PATH = "/home/data2/tanminh/Detic/downloads/mask face people/3.cropped-16813235942023-03-13t005419z_2037021470_rc2nsz97hxhz_rtrmadp_3_health-coronavirus-japan-mask.jpg"
CLASSES = ["mask", "ahaha eyeglasses", "watch", "sunglasses", "hat"]
BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.25
NMS_THRESHOLD = 0.8

# GroundingDINO config and checkpoint
GROUNDING_DINO_CONFIG_PATH = "/home/data2/tanminh/Detic/newGSAM/Grounded-Segment-Anything/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT_PATH = "./groundingdino_swint_ogc.pth"

# Building GroundingDINO inference model
grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH,
                                model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH, device= "cpu")

# grounding_dino_model.to("cpu")
# Building MobileSAM predictor
MOBILE_SAM_CHECKPOINT_PATH = "/home/data2/tanminh/Detic/newGSAM/Grounded-Segment-Anything/mobile_sam.pt"
checkpoint = torch.load(MOBILE_SAM_CHECKPOINT_PATH)
mobile_sam = setup_model()
mobile_sam.load_state_dict(checkpoint, strict=True)
mobile_sam.to(device="cpu")

sam_predictor = SamPredictor(mobile_sam)




# root_dir = "/home/data2/tanminh/Detic/downloads/mask_face_people"
# coco_class = open("/home/data2/tanminh/Detic/coco.txt", "r").readlines()
# coco_class = [i.strip("\n") for i in coco_class][48:]
# process_segment(
#     path_image="/home/data2/tanminh/Detic/downloads/mask_face_people/13.women-wearing-facemasks-while-walking-outdoors-milan-italy-february-2020-coronavirus-covid-19.jpg",
#     output_dir= "output/"
# )
for cls in ["mask_face_people"]:
    root_dir = f"/home/data2/tanminh/Detic/downloads/{cls}/"
    if not os.path.isdir(root_dir):
        print(f"[INFO] {cls} is not exist.")
        continue
    output_dir = f"coco_cls_extract/{cls}"
    os.makedirs(output_dir, exist_ok=True)
    CLASSES.append(cls)
    for path_image in tqdm(os.listdir(root_dir)):
        if not str(path_image).endswith(".jpg"):
            continue
        process_segment(os.path.join(root_dir, path_image), output_dir= output_dir)
    CLASSES.remove(cls)
