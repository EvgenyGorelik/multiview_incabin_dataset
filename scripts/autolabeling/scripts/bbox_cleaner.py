import cv2, json, glob, os
import numpy as np

from argparse import ArgumentParser


def process_segmentation_folder(folder, save_file):
    with open(save_file, 'r') as f:
        results = json.load(f)
    valid_masks = []
    for k, v in results.items():
        for box_id, valid in enumerate(v):
            if valid:
                valid_masks.append(f"{k}_mask_{box_id}.png")
    for fname in os.listdir(folder):
        if fname.endswith('.png') and fname not in valid_masks:
            # Add 'unmatched' tag to filename
            file = os.path.join(folder, fname)
            os.remove(file)
            print(f"Removed unmatched mask: {file}")
        if fname.endswith('.json'):
            json_path = os.path.join(folder, fname)
            with open(json_path, 'r') as f:
                data = json.load(f)
            new_data = []
            for i in range(len(data)):
                if data[i]['seg_fname'] not in valid_masks:
                    print(f"Removing box {i} from {json_path} because mask {data[i]['seg_fname']} is not valid")
                else:
                    new_data.append(data[i])
            with open(json_path, 'w') as f:
                json.dump(new_data, f, indent=2)

# mouse handler for area selection
def _area_mouse(event, x, y, flags, param):
    global SELECTING, _sel_start, _sel_rect, AUTO_AREAS
    if event == cv2.EVENT_LBUTTONDOWN:
        _sel_start = (x, y)
        _sel_rect = (x, y, x, y)
    elif event == cv2.EVENT_MOUSEMOVE and _sel_start is not None:
        sx, sy = _sel_start
        _sel_rect = (min(sx, x), min(sy, y), max(sx, x), max(sy, y))
    elif event == cv2.EVENT_LBUTTONUP and _sel_start is not None:
        sx, sy = _sel_start
        ex, ey = x, y
        area = (min(sx, ex), min(sy, ey), max(sx, ex), max(sy, ey))
        AUTO_AREAS.append(area)
        _sel_start = None
        _sel_rect = None
        SELECTING = False
        cv2.setMouseCallback("viewer", _ORIGINAL_MOUSE)



if __name__ == "__main__":
    parser = ArgumentParser(description="Bounding box cleaner tool")
    parser.add_argument("root", type=str, help="root directory of the dataset, should contain cameras and segmentations folders")
    parser.add_argument("--filter", type=float, default=0.99, help="automatically filter out boxes that are almost identical (IoU > 0.99), keeping the largest one")
    parser.add_argument("--cam", required=True, help="camera name")
    parser.add_argument("--save_file", default="cleaned_boxes", help="filename to save cleaned boxes (relative to each segmentation folder)")
    parser.add_argument("--clean", action="store_true", help="if set, will clean the segmentations folder by removing masks that do not have a corresponding box in the _results.json files")
    parser.epilog = "Tool for manually cleaning bounding boxes.\n\rClick on a box and press 'd' to delete it, 'u' to undo last deletion, 'a' to start area selection (drag mouse to select an area, all boxes fully inside will be removed), 'r' to remove last selected area, 'n' to save and go to next image, 'b' to save and go back to previous image. 'q' to quit without saving."
    args = parser.parse_args()

    IMG_FILES = sorted(glob.glob(os.path.join(args.root, "cameras", "color", args.cam, "images", "*.jpg")))
    SEG_FOLDER = os.path.join(args.root, "segmentations", args.cam)

    save_path = f"{args.save_file}_{args.cam}.json"

    results = {}
    if os.path.exists(save_path):
        with open(save_path, "r") as f:
            results = json.load(f)

    if args.clean:
        process_segmentation_folder(SEG_FOLDER, save_path)
        print(f"Segmentations folder cleaned based on {args.cam} camera info")
        exit()
    
    mouse = [0, 0]

    def move(event, x, y, flags, param):
        mouse[0], mouse[1] = x, y

    cv2.namedWindow("viewer")
    cv2.setMouseCallback("viewer", move)

    undo_box = None
    AUTO_AREAS = []         # list of (x1,y1,x2,y2) areas
    SELECTING = False       # are we in area-selection mode
    _sel_start = None       # temporary start point while selecting
    _sel_rect = None        # temporary rect while selecting
    _ORIGINAL_MOUSE = move  # save original mouse callback
    FILTER_MODE = args.filter > 0  # if True, automatically filter out boxes that are almost identical (IoU > 0.99), keeping the largest one


    i = 0
    while i < len(IMG_FILES):
        img_path = IMG_FILES[i]
        name = os.path.splitext(os.path.basename(img_path))[0]

        bbox_path = os.path.join(SEG_FOLDER, f"{name}.json")

        boxes = json.load(open(bbox_path))
        valid_boxes = results.get(name, np.ones(len(boxes), dtype=bool))
        if len(boxes) == 0:
            valid_boxes = np.array([], dtype=bool)
        
        img = cv2.imread(img_path)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.6, img.shape[1] / 1000.0)
        thickness = 2
        pad = 6
        x0, y0 = 10, 10
        text_size = cv2.getTextSize(name, font, scale, thickness)[0]
        rect_tl = (x0, y0)
        rect_br = (x0 + text_size[0] + pad * 2, y0 + text_size[1] + pad * 2)
        cv2.rectangle(img, rect_tl, rect_br, (0, 0, 0), -1)
        text_org = (x0 + pad, y0 + text_size[1] + pad)
        cv2.putText(img, name, text_org, font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

        if FILTER_MODE:
            # remove smaller boxes that largely overlap bigger ones (keep biggest)
            def _area(box):
                x1,y1,x2,y2 = map(int, box)
                return max(0, x2 - x1) * max(0, y2 - y1)

            def _inter_area(b1, b2):
                x1,y1,x2,y2 = map(int, b1)
                x3,y3,x4,y4 = map(int, b2)
                ix1, iy1 = max(x1, x3), max(y1, y3)
                ix2, iy2 = min(x2, x4), min(y2, y4)
                if ix2 <= ix1 or iy2 <= iy1:
                    return 0
                return (ix2 - ix1) * (iy2 - iy1)

            OVERLAP_THRESH = args.filter  # fraction of smaller box that must be covered to remove it

            # sort by area desc so larger boxes are kept preferentially
            sorted_boxes = np.argsort([-_area(b["box"]) for b in boxes])
            kept = []
            for idx in sorted_boxes:
                b_a = boxes[idx]
                a_b = _area(b_a["box"])
                skip = False
                for k in kept:
                    b_k = boxes[k]
                    a_k = _area(b_k["box"])
                    inter = _inter_area(b_a["box"], b_k["box"])
                    min_area = min(a_b, a_k)
                    if min_area > 0 and (inter / min_area) > OVERLAP_THRESH:
                        skip = True
                        break
                if not skip:
                    kept.append(idx)
            valid_boxes = valid_boxes & np.array([idx in kept for idx in range(len(boxes))], dtype=bool)

            


        while True:
            vis = img.copy()

            for b_ind, b in enumerate(boxes):
                if not valid_boxes[b_ind]:
                    continue
                x1,y1,x2,y2 = map(int,b["box"])
                cv2.rectangle(vis,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.putText(vis, f"ID: {b.get('obj_id', 'N/A')}", (x1, y1 - 10), font, scale * 0.8, (0, 255, 0), thickness, cv2.LINE_AA)

            # apply auto-deletion: remove boxes fully inside any saved area
            if AUTO_AREAS:
                valid_boxes = valid_boxes & np.array([not any(_box_inside_area(b, a) for a in AUTO_AREAS) for b in boxes], dtype=bool)

            # visualize saved areas and current selection rectangle
            for a in AUTO_AREAS:
                cv2.rectangle(vis, (a[0], a[1]), (a[2], a[3]), (255, 0, 0), 2)
            if _sel_rect is not None:
                cv2.rectangle(vis, (int(_sel_rect[0]), int(_sel_rect[1])),
                              (int(_sel_rect[2]), int(_sel_rect[3])), (0, 0, 255), 1)


            cv2.imshow("viewer", vis)
            k = cv2.waitKey(10)

            if k == ord("d"):
                mx,my = mouse
                hits = [b for b in boxes if
                        b["box"][0] <= mx <= b["box"][2] and
                        b["box"][1] <= my <= b["box"][3]]
                if hits:
                    def area(b):
                        x1,y1,x2,y2 = map(int, b["box"])
                        return max(0, x2 - x1) * max(0, y2 - y1)
                    sid = np.argmin([area(b) for b in hits])
                    undo_box = hits[sid]
                    valid_boxes = valid_boxes & np.array([b is not undo_box for b in boxes], dtype=bool)


            if k == ord("n"):
                with open(save_path, "w") as f:
                    results[name] = valid_boxes.astype(int).tolist()
                    json.dump(results, f, indent=2)
                break
            if k == ord("b"):
                with open(save_path, "w") as f:
                    results[name] = valid_boxes.astype(int).tolist()
                    json.dump(results, f, indent=2)
                i = i-2
                break
            if k == ord(" "):
                i = i+10
                break

            if k == ord("u") and undo_box is not None:
                boxes.append(undo_box)
                valid_boxes[len(boxes)-1] = True
                undo_box = None
                

            # toggle selection mode
            if k == ord("a"):
                SELECTING = True
                cv2.setMouseCallback("viewer", _area_mouse)

            # remove last area
            if k == ord("r") and AUTO_AREAS:
                AUTO_AREAS.pop()

            # helper to test box inside area (box fully inside area)
            def _box_inside_area(b, area):
                x1, y1, x2, y2 = map(int, b["box"])
                ax1, ay1, ax2, ay2 = area
                return x1 >= ax1 and y1 >= ay1 and x2 <= ax2 and y2 <= ay2
            
            if k == ord("q"):
                exit()
        i += 1