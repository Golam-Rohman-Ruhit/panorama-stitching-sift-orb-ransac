import cv2
import numpy as np


def detect_and_match(gray_left, gray_right, method="SIFT"):
    """
    gray_left, gray_right: দুইটা grayscale image
    method: "SIFT" বা "ORB"
    return: kp_left, kp_right, good_matches
    """
    method = method.upper()

    if method == "SIFT":
        # SIFT: local feature detector + descriptor
        detector = cv2.SIFT_create()
        norm_type = cv2.NORM_L2
    elif method == "ORB":
        # ORB: FAST keypoint + BRIEF-like binary descriptor
        detector = cv2.ORB_create(nfeatures=5000)
        norm_type = cv2.NORM_HAMMING
    else:
        raise ValueError("method must be 'SIFT' or 'ORB'")

    # keypoint + descriptor বের করা
    kp_left, des_left = detector.detectAndCompute(gray_left, None)
    kp_right, des_right = detector.detectAndCompute(gray_right, None)

    if des_left is None or des_right is None:
        raise RuntimeError(f"{method}: descriptors পাওয়া যায়নি, অন্য ছবি চেষ্টা করো।")

    print(f"[{method}] keypoints: left = {len(kp_left)}, right = {len(kp_right)}")

    # BF matcher + kNN (k=2) + Lowe ratio test
    bf = cv2.BFMatcher(norm_type, crossCheck=False)
    raw_matches = bf.knnMatch(des_left, des_right, k=2)

    good = []
    ratio = 0.75  # Lowe's ratio
    for m, n in raw_matches:
        if m.distance < ratio * n.distance:
            good.append(m)

    print(f"[{method}] good matches after ratio test = {len(good)}")

    if len(good) < 4:
        raise RuntimeError(f"{method}: homography করার মত match পাওয়া যায়নি (need >= 4).")

    return kp_left, kp_right, good


def estimate_homography(kp_left, kp_right, matches, method="SIFT"):
    """
    BF + ratio test শেষে পাওয়া good match থেকে Homography estimate
    """
    # left image এর পয়েন্ট
    pts_left = np.float32(
        [kp_left[m.queryIdx].pt for m in matches]
    ).reshape(-1, 1, 2)

    # right image এর পয়েন্ট
    pts_right = np.float32(
        [kp_right[m.trainIdx].pt for m in matches]
    ).reshape(-1, 1, 2)

    # আমরা চাই: pts_left ≈ H * pts_right  → right → left mapping
    H, mask = cv2.findHomography(pts_right, pts_left, cv2.RANSAC, 5.0)

    if H is None:
        raise RuntimeError(f"{method}: homography estimate করা যায়নি।")

    inliers = int(mask.sum())
    print(f"[{method}] inliers after RANSAC = {inliers} / {len(mask)}")
    print(f"[{method}] Homography matrix:\n{H}")

    return H, mask


def warp_and_stitch(img_left, img_right, H):
    """
    img_right কে img_left এর coordinate-এ warp করে stitched panorama বানায়
    """
    h1, w1 = img_left.shape[:2]
    h2, w2 = img_right.shape[:2]

    # right image এর চার corner কে warp করব
    corners_right = np.float32(
        [[0, 0], [0, h2], [w2, h2], [w2, 0]]
    ).reshape(-1, 1, 2)
    warped_corners_right = cv2.perspectiveTransform(corners_right, H)

    # left image এর corner
    corners_left = np.float32(
        [[0, 0], [0, h1], [w1, h1], [w1, 0]]
    ).reshape(-1, 1, 2)

    # দুইটার সব corner মিলিয়ে canvas বের করি
    all_corners = np.concatenate((warped_corners_right, corners_left), axis=0)

    [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    tx, ty = -xmin, -ymin  # translation so that all coords are positive
    T = np.array([[1, 0, tx],
                  [0, 1, ty],
                  [0, 0, 1]], dtype=float)

    # প্রথমে right image warp + translate
    stitched = cv2.warpPerspective(
        img_right, T @ H, (xmax - xmin, ymax - ymin)
    )

    # তারপর canvas এ left image বসিয়ে দেই
    stitched[ty:ty + h1, tx:tx + h1 if False else tx + w1] = img_left

    return stitched


def run_stitch(left_path, right_path, method="SIFT", prefix="sift"):
    """
    পুরো pipeline: read → feature extract + match → homography → stitch → save
    """
    method = method.upper()
    print("\n==============================")
    print(f"  {method} stitching started")
    print("==============================")

    # 1) image read
    img_left = cv2.imread(left_path)
    img_right = cv2.imread(right_path)

    if img_left is None or img_right is None:
        raise FileNotFoundError("left.jpg / right.jpg পাওয়া যায়নি, path ঠিক করো।")

    gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    # 2) Feature extraction + matching
    kp_left, kp_right, good_matches = detect_and_match(
        gray_left, gray_right, method=method
    )

    # 3) Homography
    H, mask = estimate_homography(kp_left, kp_right, good_matches, method=method)

    # 4) Matching visualization
    matches_mask = mask.ravel().tolist()
    matches_vis = cv2.drawMatches(
        img_left, kp_left,
        img_right, kp_right,
        good_matches, None,
        matchColor=(0, 255, 0),
        singlePointColor=(255, 0, 0),
        matchesMask=matches_mask,
        flags=cv2.DrawMatchesFlags_DEFAULT,
    )

    # 5) Warp + stitch
    stitched = warp_and_stitch(img_left, img_right, H)

    # 6) Save all results for the report
    cv2.imwrite(f"{prefix}_input_left.jpg", img_left)
    cv2.imwrite(f"{prefix}_input_right.jpg", img_right)
    cv2.imwrite(f"{prefix}_matches.jpg", matches_vis)
    cv2.imwrite(f"{prefix}_stitched.jpg", stitched)

    print(f"[{method}] saved files:")
    print(f"  {prefix}_input_left.jpg   (original left)")
    print(f"  {prefix}_input_right.jpg  (original right)")
    print(f"  {prefix}_matches.jpg      (intermediate: inlier matches)")
    print(f"  {prefix}_stitched.jpg     (final stitched panorama)")


if __name__ == "__main__":
    # নিজের তোলা দুইটা ছবি – এই নামেই রাখো
    left_image_path = "left.jpg"
    right_image_path = "right.jpg"

    # 1) SIFT-based image stitching (reference method)
    run_stitch(left_image_path, right_image_path,
               method="SIFT", prefix="sift")

    # 2) ORB-based image stitching (Part III: অন্য feature detector/descriptor)
    run_stitch(left_image_path, right_image_path,
               method="ORB", prefix="orb")
