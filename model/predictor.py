"""
predictor.py
============
Module for making predictions with pre-trained neural networks,
including semantic segmentation model.
Modified from predictor.py by Maxim Ziatdinov (maxim.ziatdinov@ai4microscopy.com)
"""

import time
import cv2
from typing import Dict, List, Tuple, Type, Union
from skimage.segmentation import watershed
import numpy as np
from scipy import ndimage
import torch
import torch.nn.functional as F


def set_train_rng(seed: int = 1):
    """
    For reproducibility
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def torch_format_image(image_data: np.ndarray,
                       norm: bool = True) -> torch.Tensor:
    """
    Reshapes (if needed), normalizes and converts image data
    to pytorch format for model training and prediction
    Args:
        image_data (3D or 4D numpy array):
            Image stack with dimensions (n_batches x height x width)
            or (n_batches x 1 x height x width)
        norm (bool):
            Normalize to (0, 1) (Default: True)
    """
    if image_data.ndim not in [3, 4]:
        raise AssertionError(
            "Provide image(s) as 3D (n, h, w) or 4D (n, 1, h, w) tensor")
    if np.ndim(image_data) == 3:
        image_data = np.expand_dims(image_data, axis=1)
    elif np.ndim(image_data) == 4 and image_data.shape[1] != 1:
        raise AssertionError(
            "4D image tensor must have (n, 1, h, w) dimensions")
    else:
        pass
    if norm:
        image_data = (image_data - image_data.min()) / image_data.ptp()
    image_data = torch.tensor(image_data).float()
    return image_data


def cv_resize(img: np.ndarray, rs: Tuple[int],
              round_: bool = False) -> np.ndarray:
    """
    Wrapper for open-cv resize function
    Args:
        img (2D numpy array): input 2D image
        rs (tuple): target height and width
        round_(bool): rounding (in case of labeled pixels)
    Returns:
        Resized image
    """
    if img.shape == rs:
        return img
    rs = (rs[1], rs[0])
    rs_method = cv2.INTER_AREA if img.shape[0] < rs[0] else cv2.INTER_CUBIC
    img_rs = cv2.resize(img, rs, interpolation=rs_method)
    if round_:
        img_rs = np.round(img_rs)
    return img_rs


def cv_thresh(imgdata: np.ndarray,
              threshold: float = .5):
    """
    Wrapper for opencv binary threshold method.
    Returns thresholded image.
    """
    _, thresh = cv2.threshold(
        imgdata,
        threshold, 1,
        cv2.THRESH_BINARY)
    return thresh


def img_pad(image_data: np.ndarray, pooling: int) -> np.ndarray:
    """
    Pads the image if its size (w, h)
    is not divisible by :math:`2^n`, where *n* is a number
    of pooling layers in a network
    Args:
        image_data (3D numpy array):
            Image stack with dimensions (n_batches x height x width)
        pooling (int):
            Downsampling factor (equal to :math:`2^n`, where *n* is a number
            of pooling operations)
    """
    pooling = 2
    # Pad image rows (height)
    while image_data.shape[1] % pooling != 0:
        d0, _, d2 = image_data.shape
        image_data = np.concatenate(
            (image_data, np.zeros((d0, 1, d2))), axis=1)
    # Pad image columns (width)
    while image_data.shape[2] % pooling != 0:
        d0, d1, _ = image_data.shape
        image_data = np.concatenate(
            (image_data, np.zeros((d0, d1, 1))), axis=2)
    return image_data


def img_resize(image_data: np.ndarray, rs: Tuple[int],
               round_: bool = False) -> np.ndarray:
    """
    Resizes a stack of images
    Args:
        image_data (3D numpy array):
            Image stack with dimensions (n_batches x height x width)
        rs (tuple):
            Target height and width
        round_(bool):
            rounding (in case of labeled pixels)
    Returns:
        Resized stack of images
    """
    if rs[0] != rs[1]:
        rs = (rs[1], rs[0])
    if image_data.shape[1:3] == rs:
        return image_data.copy()
    image_data_r = np.zeros(
        (image_data.shape[0], rs[0], rs[1]))
    for i, img in enumerate(image_data):
        img = cv_resize(img, rs, round_)
        image_data_r[i, :, :] = img
    return image_data_r


def find_com(image_data: np.ndarray) -> np.ndarray:
    """
    Find atoms via center of mass methods
    Args:
        image_data (2D numpy array):
            2D image (usually an output of neural network)
    """
    labels, nlabels = ndimage.label(image_data)
    coordinates = np.array(
        ndimage.center_of_mass(
            image_data, labels, np.arange(nlabels) + 1))
    coordinates = coordinates.reshape(coordinates.shape[0], 2)
    return coordinates


def mapint(mat):
    data = (mat - mat.min()) / (mat.max() - mat.min())
    return (data * 255).astype(np.uint8)


class BasePredictor:
    """
    Base predictor class
    """

    def __init__(self,
                 model: Type[torch.nn.Module] = None,
                 use_gpu: bool = False,
                 **kwargs: Union[bool, str]) -> None:
        """
        Initialize predictor
        Args:
            model: trained pytorch model
            use_gpu: Use GPU accelerator (Default: False)
            **device: CUDA device, e.g. 'cuda:0'
        """
        self.model = model
        self.device = "cpu"
        if use_gpu and torch.cuda.is_available():
            if kwargs.get("device") is None:
                self.device = "cuda"
            else:
                self.device = kwargs.get("device")
        if self.model is not None:
            self.model.to(self.device)
        self.verbose = kwargs.get("verbose", False)

    def preprocess(self,
                   data: Union[torch.Tensor, np.ndarray]
                   ) -> None:
        """
        Preprocess input data
        """
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()
        return data

    def _model2device(self, device: str = None) -> None:
        if device is None:
            device = self.device
        self.model.to(device)

    def _data2device(self,
                     data: torch.Tensor,
                     device: str = None) -> torch.Tensor:
        if device is None:
            device = self.device
        data = data.to(device)
        return data

    def forward_(self, xnew: torch.Tensor) -> torch.Tensor:
        """
        Pass data through a trained neural network
        """
        self.model.eval()
        with torch.no_grad():
            out = self.model(xnew.to(self.device))
        return out

    def batch_predict(self,
                      data: torch.Tensor,
                      out_shape: Tuple[int],
                      num_batches: int) -> torch.Tensor:
        """
        Make a prediction batch-by-batch (for larger datasets)
        """
        batch_size = len(data) // num_batches
        if batch_size < 1:
            num_batches = batch_size = 1
        # prediction_all = np.zeros(shape=out_shape)
        prediction_all = torch.zeros(out_shape)
        for i in range(num_batches):
            if self.verbose:
                print("\rBatch {}/{}".format(i + 1, num_batches), end="")
            data_i = data[i * batch_size:(i + 1) * batch_size]
            prediction_i = self.forward_(data_i)
            # We put predictions on cpu since the major point of batch-by-batch
            # prediction is to not run out of the GPU memory
            prediction_all[i * batch_size:(i + 1) * batch_size] = prediction_i.cpu()
        data_i = data[(i + 1) * batch_size:]
        if len(data_i) > 0:
            prediction_i = self.forward_(data_i)
            prediction_all[(i + 1) * batch_size:] = prediction_i.cpu()
        return prediction_all

    def predict(self,
                data: torch.Tensor,
                out_shape: Tuple[int] = None,
                num_batches: int = 1) -> torch.Tensor:
        """
        Make a prediction on the new data with a trained model
        """
        if out_shape is None:
            out_shape = data.shape
        else:
            out_shape = (data.shape[0], *out_shape)
        data = self.preprocess(data)
        prediction = self.batch_predict(data, out_shape, num_batches)
        return prediction


class SegPredictor(BasePredictor):
    """
    Prediction with a trained fully convolutional neural network
    Args:
        trained_model:
            Trained pytorch model (skeleton+weights)
        resize:
            Target dimensions for optional image(s) resizing
        use_gpu:
            Use gpu device for inference
        logits:
            Indicates that the image data is passed through
            a softmax/sigmoid layer when set to False
            (logits=True for AtomAI models)
        **thresh (float):
            value between 0 and 1 for thresholding the NN output
            (Default: 0.2)
        **glfilter_thresh (float):
            value between 0 and 1 for thresholding the NN output for laplacian gaussian filter
            (Default: 0.02)
        **glfilter_sigma (float):
            value at least 1 for sigma of laplacian gaussian filter. Use larger value to reduce false positives. 
            (Default: 1)
        **ostu_thresh ([min_int, max_int])
            min and max value for the ostu filter within range 0-255
        **nnfilter (string):
            name of filter to be applied on the NN output: binarize or gaussian_laplace. 
            (Default: binarize)
        **d (int):
            half-side of a square around each atomic position used
            for refinement with 2d Gaussian peak fitting. Defaults to 1/4
            of average nearest neighbor atomic distance
        **nb_classes (int):
            Number of classes in the model
        **downsampling (int or float):
            Downsampling factor (equal to :math:`2^n` where *n* is a number
            of pooling operations)
    Example:
        >>> # Here we load new experimental data (as 2D or 3D numpy array)
        >>> expdata = np.load('expdata-test.npy')
        >>> # Get prediction from a trained model
        >>> pseg = atomnet.SegPredictor(trained_model)
        >>> nn_output, coords = pseg.run(expdata)
    """

    def __init__(self,
                 trained_model: Type[torch.nn.Module],
                 resize: Union[Tuple, List] = None,
                 use_gpu: bool = False,
                 logits: bool = True,
                 **kwargs: Union[int, float, bool, str]) -> None:
        """
        Initializes predictive object
        """
        super(SegPredictor, self).__init__(trained_model, use_gpu)
        set_train_rng(1)
        self.nb_classes = kwargs.get('nb_classes', None)
        if self.nb_classes is None:
            self.nb_classes = 1

        self.resize = resize
        self.logits = logits
        self.d = kwargs.get("d", None)
        self.thresh = kwargs.get("thresh", .5)
        self.glfilter_thresh = kwargs.get("glfilter_thresh", 0.02)
        self.nnfilter = kwargs.get("nnfilter", 'binarize')
        self.glfilter_sigma = kwargs.get("glfilter_sigma", '3')
        self.ostu_thresh = kwargs.get("ostu_thresh", [30, 100])
        self.use_gpu = use_gpu
        self.verbose = kwargs.get("verbose", True)

    def preprocess(self,
                   image_data: np.ndarray,
                   norm: bool = True) -> torch.Tensor:
        """
        Prepares an input for a neural network
        """
        if image_data.ndim == 2:
            image_data = image_data[np.newaxis, ...]
        elif image_data.ndim == 4:
            if image_data.shape[-1] == 1:
                image_data = image_data[..., 0]
            elif image_data.shape[1] == 1:
                image_data = image_data[:, 0, ...]
        if self.resize is not None:
            image_data = img_resize(image_data, self.resize)
        image_data = img_pad(image_data, 2)
        image_data = torch_format_image(image_data, norm)
        return image_data

    def forward_(self, images: torch.Tensor) -> np.ndarray:
        """
        Returns 'probability' of each pixel
        in image(s) belonging to an atom/defect
        """
        images = images.to(self.device)
        self.model.eval()
        with torch.no_grad():
            prob = self.model(images)
        if self.logits:
            if self.nb_classes > 1:
                prob = F.softmax(prob, dim=1)
            else:
                prob = torch.sigmoid(prob)
        else:
            if self.nb_classes > 1:
                prob = torch.exp(prob)
            else:
                pass
        prob = prob.permute(0, 2, 3, 1)  # reshape to have channel as a last dim
        images = images.cpu()
        prob = prob.cpu()
        return prob

    def predict(self,
                image_data: np.ndarray,
                return_image: bool = False,
                **kwargs: int) -> Tuple[np.ndarray]:
        """
        Make prediction
        Args:
            image_data:
                3D image stack or a single 2D image (all greyscale)
            return_image:
                Returns images used as input into NN
            **num_batches (int): number of batches
            **norm (bool): Normalize data to (0, 1) during pre-processing
        """
        image_data = self.preprocess(
            image_data, kwargs.get("norm", True))
        n, _, w, h = image_data.shape
        num_batches = kwargs.get("num_batches")
        if num_batches is None:
            if w >= 256 or h >= 256:
                num_batches = len(image_data)
            else:
                num_batches = 10
        segmented_imgs = self.batch_predict(
            image_data, (n, w, h, self.nb_classes), num_batches)
        if return_image:
            image_data = image_data.permute(0, 2, 3, 1).numpy()
            return image_data, segmented_imgs.numpy()
        return segmented_imgs.numpy()

    def run(self,
            image_data: np.ndarray,
            compute_coords=True,
            **kwargs: int) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
        """
        Make prediction with a trained model and calculate coordinates
        Args:
            image_data:
                Image stack or a single image (all greyscale)
            compute_coords:
                Computes centers of the mass of individual blobs
                in the segmented images (Default: True)
            **num_batches (int):
                number of batches for batch-by-batch prediction
                which ensures that one doesn't run out of memory
                (Default: 10)
            **norm (bool): Normalize data to (0, 1) during pre-processing
        """
        start_time = time.time()
        if not compute_coords:
            decoded_imgs = self.predict(image_data, **kwargs)
            return decoded_imgs
        images, decoded_imgs = self.predict(
            image_data, return_image=True, **kwargs)
        loc = Locator(self.thresh, d=self.d, nnfilter=self.nnfilter,
                      glfilter_thresh=self.glfilter_thresh, glfilter_sigma=self.glfilter_sigma,
                      ostu_thresh=self.ostu_thresh)
        coordinates = loc.run(decoded_imgs, images)
        if self.verbose:
            n_images_str = " image was " if decoded_imgs.shape[0] == 1 else " images were "
            print("\n" + str(decoded_imgs.shape[0])
                  + n_images_str + "decoded in approximately "
                  + str(np.around(time.time() - start_time, decimals=4))
                  + ' seconds')
        return decoded_imgs, coordinates


class Locator:
    """
    Transforms pixel data from NN output into coordinate data
    Args:
        decoded_imgs:
            Output of a neural network
        threshold:
            Value at which the neural network output is thresholded
        dist_edge:
            Distance within image boundaries not to consider
        dim_order:
            'channel_last' or 'channel_first' (Default: 'channel last')
    Example:
        >>> # Transform output of atomnet.predictor to atomic classes and coordinates
        >>> coordinates = locator(dist_edge=10, refine=False).run(nn_output)
    """

    def __init__(self,
                 threshold: float = 0.2,
                 dist_edge: int = 5,
                 dim_order: str = 'channel_last',
                 **kwargs: Union[bool, float, str]) -> None:
        """
        Initialize locator parameters
        """
        self.dim_order = dim_order
        self.threshold = threshold
        self.dist_edge = dist_edge
        self.d = kwargs.get("d")
        self.glfilter_thresh = kwargs.get("glfilter_thresh", 0.02)
        self.glfilter_sigma = kwargs.get("glfilter_sigma", '1')
        self.nnfilter = kwargs.get("nnfilter", 'binarize')
        self.ostu_thresh = kwargs.get("ostu_thresh", [30, 100])

    def preprocess(self, nn_output: np.ndarray) -> np.ndarray:
        """
        Prepares data for coordinates extraction
        """
        if nn_output.shape[-1] == 1:  # Add background class for 1-channel data
            nn_output_b = 1 - nn_output
            nn_output = np.concatenate(
                (nn_output, nn_output_b), axis=3)
        if self.dim_order == 'channel_first':  # make channel dim the last dim
            nn_output = np.transpose(nn_output, (0, 2, 3, 1))
        elif self.dim_order == 'channel_last':
            pass
        else:
            raise NotImplementedError(
                'For dim_order, use "channel_first"',
                'or "channel_last" (e.g. tensorflow)')
        return nn_output

    def run(self, nn_output: np.ndarray, *args: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Extract all atomic coordinates in image
        via CoM method & store data as a dictionary
        (key: frame number)
        Args:
            nn_output (4D numpy array):
                Output (prediction) of a neural network
            *args: 4D input into a neural network (experimental data)
        """
        nn_output = self.preprocess(nn_output)
        d_coord = {}
        for i, decoded_img in enumerate(nn_output):
            coordinates = np.empty((0, 2))
            category = np.empty((0, 1))
            # we assume that class 'background' is always the last one
            for ch in range(decoded_img.shape[2] - 1):

                if self.nnfilter == 'binarize':
                    decoded_img_c = cv_thresh(decoded_img[:, :, ch], self.threshold)
                    coord = find_com(decoded_img_c)
                elif self.nnfilter == 'gaussian_laplace':
                    decoded_img_c = decoded_img[:, :, ch]
                    decoded_img_c[decoded_img_c < self.glfilter_thresh] = 0
                    # sx = ndimage.sobel(decoded_img_c, axis=0, mode='constant')
                    # sy = ndimage.sobel(decoded_img_c, axis=1, mode='constant')
                    lag = ndimage.gaussian_laplace(decoded_img_c, sigma=self.glfilter_sigma)
                    coord = find_com(lag < -0.001)
                elif self.nnfilter == 'ostu':
                    decoded_img_c = decoded_img[:, :, ch]

                    decoded_img_c[decoded_img_c < 0] = 0
                    decoded_img_c = mapint(decoded_img_c)
                    elevation_map = ndimage.sobel(decoded_img_c)  # Sobel filter, a gradient filter for edge detection
                    markers = np.zeros_like(decoded_img_c)

                    min_thre = self.ostu_thresh[0]
                    max_thre = self.ostu_thresh[1]
                    markers[decoded_img_c < min_thre] = 1
                    markers[decoded_img_c > max_thre] = 2
                    seg_1 = watershed(elevation_map, markers)
                    filled_regions = ndimage.binary_fill_holes(seg_1 - 1)
                    coord = find_com(filled_regions)
                    del elevation_map
                    del markers, seg_1, filled_regions

                else:
                    raise AssertionError("Use nnfilter = binarize or gaussian_laplace or ostu")

                coord_ch = self.rem_edge_coord(coord, *nn_output.shape[1:3])
                category_ch = np.zeros((coord_ch.shape[0], 1)) + ch
                coordinates = np.append(coordinates, coord_ch, axis=0)
                category = np.append(category, category_ch, axis=0)
            d_coord[i] = np.concatenate((coordinates, category), axis=1)

        return d_coord

    def rem_edge_coord(self, coordinates: np.ndarray, h: int, w: int) -> np.ndarray:
        """
        Removes coordinates at the image edges
        """

        def coord_edges(coordinates, h, w):
            return [coordinates[0] > h - self.dist_edge,
                    coordinates[0] < self.dist_edge,
                    coordinates[1] > w - self.dist_edge,
                    coordinates[1] < self.dist_edge]

        coord_to_rem = [
            idx for idx, c in enumerate(coordinates)
            if any(coord_edges(c, h, w))
        ]
        coord_to_rem = np.array(coord_to_rem, dtype=int)
        coordinates = np.delete(coordinates, coord_to_rem, axis=0)
        return coordinates
