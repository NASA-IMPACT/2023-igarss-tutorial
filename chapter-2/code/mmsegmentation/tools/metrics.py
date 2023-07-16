import cv2
import numpy as np
from sklearn.metrics import jaccard_score, accuracy_score, balanced_accuracy_score, f1_score, precision_score, \
    recall_score, fbeta_score, precision_recall_fscore_support


def metrics(y_true, y_pred) -> []:
    """
    Returns metrics
    """
    y_true, y_pred = y_true.flatten(), y_pred.flatten()
    y_true = np.uint8(y_true)
    ignore_index = np.argwhere(y_true == 2)
    # TODO: How to deal with completely noise image?
    if len(ignore_index) == 512 * 512:
        ignore_index = ignore_index[:-2]
        if y_true[512 * 512 - 1] == 2:
            y_true[512 * 512 - 1] = 0
            y_true[512 * 512 - 2] = 1
    y_true = np.delete(y_true, ignore_index)
    y_pred = np.delete(y_pred, ignore_index)

    # metrics
    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
    bal_accuracy = balanced_accuracy_score(y_true=y_true, y_pred=y_pred)
    precision = precision_score(y_true=y_true, y_pred=y_pred, average="binary")
    precision_weighted = precision_score(y_true=y_true, y_pred=y_pred, average="weighted")
    recall = recall_score(y_true=y_true, y_pred=y_pred, average="binary")
    recall_weighted = recall_score(y_true=y_true, y_pred=y_pred, average="weighted")
    iou_score = jaccard_score(y_true=y_true, y_pred=y_pred, average="weighted")
    f1 = f1_score(y_true=y_true, y_pred=y_pred, average="weighted")
    f1_micro = f1_score(y_true=y_true, y_pred=y_pred, average="micro")
    f1_macro = f1_score(y_true=y_true, y_pred=y_pred, average="macro")
    f0_5 = fbeta_score(y_true, y_pred, average='weighted', beta=0.5)
    f0_1 = fbeta_score(y_true, y_pred, average='weighted', beta=0.1)
    f10 = fbeta_score(y_true, y_pred, average='weighted', beta=10)
    precision_per_class, recall_per_class, fscore_per_class, support_per_class = precision_recall_fscore_support(y_true, y_pred)
    iou_score_per_class = jaccard_score(y_true=y_true, y_pred=y_pred, average=None)


    return accuracy, bal_accuracy, precision, precision_weighted, recall, recall_weighted, iou_score, f1, \
           f1_micro, f1_macro, f0_5, f0_1, f10, precision_per_class, recall_per_class, fscore_per_class, \
           support_per_class, iou_score_per_class


# Augmentations
def shift_image(img, shift_pnt):
    M = np.float32([[1, 0, shift_pnt[0]], [0, 1, shift_pnt[1]]])
    res = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REFLECT_101)
    return res


def rotate_image(image, angle, scale, rot_pnt):
    rot_mat = cv2.getRotationMatrix2D(rot_pnt, angle, scale)
    result = cv2.warpAffine(image, rot_mat, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REFLECT_101)  # INTER_NEAREST
    return result


def gauss_noise(img, var=30):
    row, col, ch = img.shape
    mean = var
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    gauss = (gauss - np.min(gauss)).astype(np.uint8)
    return np.clip(img.astype(np.int32) + gauss, 0, 255).astype('uint8')


def clahe(img, clipLimit=2.0, tileGridSize=(5, 5)):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_LAB2RGB)
    return img_output


def _blend(img1, img2, alpha):
    return np.clip(img1 * alpha + (1 - alpha) * img2, 0, 255).astype('uint8')


_alpha = np.asarray([0.114, 0.587, 0.299]).reshape((1, 1, 3))


def _grayscale(img):
    return np.sum(_alpha * img, axis=2, keepdims=True)


def saturation(img, alpha):
    gs = _grayscale(img)
    return _blend(img, gs, alpha)


def brightness(img, alpha):
    gs = np.zeros_like(img)
    return _blend(img, gs, alpha)


def contrast(img, alpha):
    gs = _grayscale(img)
    gs = np.repeat(gs.mean(), 3)
    return _blend(img, gs, alpha)


def change_hsv(img, h, s, v):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = hsv.astype(int)
    hsv[:, :, 0] += h
    hsv[:, :, 0] = np.clip(hsv[:, :, 0], 0, 255)
    hsv[:, :, 1] += s
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    hsv[:, :, 2] += v
    hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
    hsv = hsv.astype('uint8')
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img


def shift_channels(img, b_shift, g_shift, r_shift):
    img = img.astype(int)
    img[:, :, 0] += b_shift
    img[:, :, 0] = np.clip(img[:, :, 0], 0, 255)
    img[:, :, 1] += g_shift
    img[:, :, 1] = np.clip(img[:, :, 1], 0, 255)
    img[:, :, 2] += r_shift
    img[:, :, 2] = np.clip(img[:, :, 2], 0, 255)
    img = img.astype('uint8')
    return img


def invert(img):
    return 255 - img


def channel_shuffle(img):
    ch_arr = [0, 1, 2]
    np.random.shuffle(ch_arr)
    img = img[..., ch_arr]
    return img


#######

def preprocess_inputs(x):
    x = np.asarray(x, dtype='float32')
    x /= 127
    x -= 1
    return x


def dice(im1, im2, empty_score=1.0):
    """
    Computes the Dice coefficient, a measure of set similarity.
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0
        Both are empty (sum eq to zero) = empty_score

    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.
    """
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / im_sum


def iou(im1, im2, empty_score=1.0):
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    union = np.logical_or(im1, im2)
    im_sum = union.sum()
    if im_sum == 0:
        return empty_score

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return intersection.sum() / im_sum
