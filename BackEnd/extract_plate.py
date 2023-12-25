# %%
from skimage import io
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_holes
from skimage.measure import find_contours
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle
import imutils

# %% [markdown]
# # Utilities

# %%


def show_with_contours(image, contours, title=None):
    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap='gray')

    for contour in contours:
        min_row, min_col = int(min(contour[:, 0])), int(min(contour[:, 1]))
        max_row, max_col = int(max(contour[:, 0])), int(max(contour[:, 1]))

        rect_height = max_row - min_row
        rect_width = max_col - min_col

        rectangle = plt.Rectangle((min_col, min_row), rect_width, rect_height,
                                  edgecolor='green', linewidth=1, facecolor='none')
        plt.gca().add_patch(rectangle)

    plt.axis('off')
    if title is not None:
        plt.title('Rectangles around Contours')
    plt.show()

# %%


def show_with_contours_dict(image, contours_dict, title=None, axes=None):
    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap='gray')

    for c in contours_dict:
        rectangle = plt.Rectangle((c["x"][1], c["y"][1]), c["w"], c["h"],
                                  edgecolor='green', linewidth=2, facecolor='none')
        plt.gca().add_patch(rectangle)

    plt.axis('off')
    if title is not None:
        plt.title('Rectangles around Contours')
    plt.show()

# %%


def show_plate_chars(char_list):
    fig, axes = plt.subplots(1, len(char_list))
    for i, char in enumerate(char_list):
        axes[i].imshow(char, cmap="gray")
        axes[i].axis("off")
    plt.show()

# %%
# extracting the hog


def extract_hog_features(img):
    if img.dtype == "bool":
        img = np.where(img, 255, 0).astype(np.uint8)
    img = cv2.resize(img, (32, 32))
    win_size = (32, 32)
    cell_size = (4, 4)
    block_size_in_cells = (2, 2)

    block_size = (block_size_in_cells[1] * cell_size[1],
                  block_size_in_cells[0] * cell_size[0])
    block_stride = (cell_size[1], cell_size[0])
    nbins = 9  # Number of orientation bins
    hog = cv2.HOGDescriptor(win_size, block_size,
                            block_stride, cell_size, nbins)
    h = hog.compute(img)
    h = h.flatten()
    return h.flatten()

# %%
# extract contours in skimage


def get_contours(img):
    contours_dict = []
    contours = find_contours(img)

    for contour in contours:
        min_row, min_col = int(min(contour[:, 0])), int(min(contour[:, 1]))
        max_row, max_col = int(max(contour[:, 0])), int(max(contour[:, 1]))

        rect_height = max_row - min_row
        rect_width = max_col - min_col

        ratio = 99999
        if rect_height != 0:
            ratio = rect_width/rect_height

        contours_dict.append({"area": rect_height*rect_width, "w/h": ratio, "y": (
            max_row, min_row), "x": (max_col, min_col), "h": rect_height, "w": rect_width})
    return contours, contours_dict

# %% [markdown]
# # Testing an image


# %%
# img = cv2.imread('dataset/images/Cars3.png')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))
# plt.show()

# %% [markdown]
# # Preprocess

# %%


def get_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def preprocess(gray):

    new_img = cv2.equalizeHist(gray)  # enhance contrast
    new_img = cv2.bilateralFilter(gray, 11, 17, 17)  # smooth image
#     new_img = new_img < threshold_otsu(new_img) # thresholding
#     new_img = new_img < threshold_sauvola(new_img) # thresholding
    return new_img


# gray = get_gray(img)
# preprocessed = preprocess(gray)
# io.imshow(preprocessed)
# io.show()

# %% [markdown]
# # Get Edges

# %%


def get_edges(preprocessed):
    edges = cv2.Canny(preprocessed, 30, 200)  # Edge detection
    return edges


# edges = get_edges(preprocessed)
# io.imshow(edges)
# io.show()

# %% [markdown]
# # Get Suspected Plates

# %%


def get_plates(edges, gray):
    keypoints = cv2.findContours(
        edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]

    locations = []
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 20, True)
        if len(approx) == 4:
            locations.append(approx)
            break

    for contour in contours:
        approx = cv2.approxPolyDP(contour, 11, True)
        if len(approx) == 4:
            locations.append(approx)
            break

    if len(locations) == 0:
        return False

    plates = []
    for location in locations:
        mask = np.zeros(gray.shape, np.uint8)
        new_image = cv2.drawContours(mask, [location], 0, 255, -1)
        new_image = cv2.bitwise_and(gray, gray, mask=mask)

        (x, y) = np.where(mask == 255)
        (x1, y1) = (np.min(x), np.min(y))
        (x2, y2) = (np.max(x), np.max(y))
        plates.append(gray[x1:x2+1, y1:y2+1])

    return plates


# plates = get_plates(edges, gray)
# for plate in plates:
#     io.imshow(plate)
#     io.show()

# %% [markdown]
# # Process Plates

# %%


def preprocess_plates(plates):
    pre_plates = []
    for plate in plates:
        temp_img = plate < threshold_otsu(plate)  # thresholding
        pre_plates.append(temp_img)
    return pre_plates


# pre_plates = preprocess_plates(plates)
# for image in pre_plates:
#     io.imshow(image)
#     io.show()

# %% [markdown]
# # Extract Plates Contours

# %%


def get_plate_contours(pre_plates):
    plate_contours = []
    plate_contours_dict = []

    for pre_plate in pre_plates:
        plt_cont, plt_cont_dict = get_contours(pre_plate)
        plate_contours.append(plt_cont)
        plate_contours_dict.append(plt_cont_dict)

    return plate_contours, plate_contours_dict


# plate_contours, plate_contours_dict = get_plate_contours(pre_plates)

# for i in range(len(pre_plates)):
#     show_with_contours_dict(pre_plates[i], plate_contours_dict[i])

# %% [markdown]
# # 1st Filter

# %%


def filter_size(plate_contours_dict):
    AREA_THRESHOLD = 74
    WIDTH_THRESHOLD = 2
    HEIGHT_THRESHOLD = 7
    MIN_RATIO = 0.16
    MAX_RATIO = 1

    # TODO: filter with relative size to the window
    # TODO: filter with the amount of ones in the contour

    filtered_plate_contours_dict = []
    # TODO: check for contours that are inside other contours here
    for contours_dict in plate_contours_dict:
        filtered_contours_dict = []

        for c in contours_dict:
            if c['area'] > AREA_THRESHOLD and c["w"] > WIDTH_THRESHOLD and c['h'] > HEIGHT_THRESHOLD and MIN_RATIO < c['w/h'] < MAX_RATIO:
                filtered_contours_dict.append(c)

        # sort the contours based from left tot right
        filtered_contours_dict = sorted(
            filtered_contours_dict, key=lambda x: x['x'][0])
        filtered_plate_contours_dict.append(filtered_contours_dict)
    return filtered_plate_contours_dict


# filtered_plate_contours_dict = filter_size(plate_contours_dict)

# for i in range(len(filtered_plate_contours_dict)):
#     show_with_contours_dict(pre_plates[i], filtered_plate_contours_dict[i])

# %% [markdown]
# # 2nd Filter

# %%


def filter_alignment(filtered_plate_contours_dict):
    W_THRESHOLD = 0.8
    H_THRESHOLD = 0.2
    ANGLE_THRESHOLD = np.radians(12)
    AREA_DIFF_THRESHOLD = 1
    DIAGONAL_THRESHOLD = 1.1

    plate_aligned_dict = []

    for filtered_contours_dict in filtered_plate_contours_dict:
        final_contours = []

        for c1 in filtered_contours_dict:
            diagonal = (c1["w"] ** 2 + c1["h"] ** 2)**0.5
            for c2 in filtered_contours_dict:
                if c1 == c2:  # make sure they're different contours
                    continue

                dx = abs(c1['x'][0] - c2['x'][0])  # x difference
                dy = abs(c1['y'][0] - c2['y'][0])  # y difference

                # filter based on  angle
                if dx == 0 or np.arctan(dy/dx) > ANGLE_THRESHOLD:
                    continue

                if abs(c1["area"] - c2["area"]) / c1["area"] > AREA_DIFF_THRESHOLD or abs(c1["w"] - c2["w"]) / c1["w"] > W_THRESHOLD or abs(c1["h"] - c2["h"]) / c1["h"] > H_THRESHOLD or dx > diagonal * DIAGONAL_THRESHOLD:
                    continue

                final_contours.append(c2)

        final_contours = [dict(t)
                          for t in {tuple(sorted(d.items())) for d in final_contours}]

        # sort the contours based from left tot right
        final_contours = sorted(final_contours, key=lambda x: x['x'][0])
        plate_aligned_dict.append(final_contours)
    return plate_aligned_dict


# plate_aligned_dict = filter_alignment(filtered_plate_contours_dict)
# for i in range(len(plate_aligned_dict)):
#     show_with_contours_dict(pre_plates[i], plate_aligned_dict[i])

# %% [markdown]
# # Extract Characters

# %%


def get_chars(pre_plates, plate_aligned_dict):
    plate_chars = []

    for i in range(len(plate_aligned_dict)):
        chars_img = []
        if len(plate_aligned_dict[i]) != 0:
            for char in plate_aligned_dict[i]:
                left = char["x"][1]
                right = char["x"][0]
                top = char["y"][1]
                bottom = char["y"][0]

                cropped_char = pre_plates[i][top:bottom, left:right]

                # TODO: character level filtering here
                cropped_morph_char = remove_small_holes(cropped_char, 4)

                cropped_morph_char = np.pad(
                    cropped_morph_char, pad_width=1, mode="constant", constant_values=0)
                chars_img.append(cropped_morph_char)
            plate_chars.append(chars_img)
    return plate_chars


# plate_chars = get_chars(pre_plates, plate_aligned_dict)

# for chars_list in plate_chars:
#     show_plate_chars(chars_list)

# %% [markdown]
# # Load OCR and Extract Features

# %%
OCR = pickle.load(open('modelKNN.OCR', 'rb'))


def ocr(plate_chars):
    total_features = []
    predictions = []
    for chars_list in plate_chars:
        test_features = []
        for char in chars_list:
            test_features.append(extract_hog_features(char))
        total_features.append(test_features)
#         show_plate_chars(chars_list)
        if len(test_features) == 0:
            return [[False]], [[False]]
        predictions.append(OCR.predict(test_features))
    return total_features, predictions


# %%
# total_features, predictions = ocr(plate_chars)
# print(predictions)

# %% [markdown]
# # ANPR

# %%


def plate_extractor(image):
    img = np.copy(image)

    gray = get_gray(img)

    preprocessed = preprocess(gray)

    edges = get_edges(preprocessed)

    plates = get_plates(edges, gray)

    if type(plates) == bool:
        return False

    pre_plates = preprocess_plates(plates)

    plate_contours, plate_contours_dict = get_plate_contours(pre_plates)

    filtered_plate_contours_dict = filter_size(plate_contours_dict)

    plate_aligned_dict = filter_alignment(filtered_plate_contours_dict)

    plate_chars = get_chars(pre_plates, plate_aligned_dict)

    total_features, predictions = ocr(plate_chars)

    prediction = [""]
    plate = None

    if predictions != []:
        prediction = max(predictions, key=len)

    if plate_aligned_dict != []:
        max_length = max(plate_aligned_dict, key=len)
        plate_index = plate_aligned_dict.index(max_length)
    predictionString =  ''.join(prediction)

    return plates[plate_index], predictionString


# %%
# plate, prediction = plate_extractor(cv2.imread("D:\IP_Team7\Tests\Cars7.png"))
# io.imshow(plate)
# io.show()
# print(prediction)

# %%
