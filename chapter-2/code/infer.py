import os
import matplotlib.pyplot as plt
import mmcv
import mmengine
import logging
import rasterio
import torch

from mmseg.datasets.pipelines.compose import Compose

from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint
from mmseg.models import build_segmentor
from post_process import PostProcess

CONFIG_DIR = "/opt/mmsegmentation/configs/{experiment}_config/geospatial_fm_config.py"
DOWNLOAD_FOLDER = "/opt/downloads"

class LoadImage:
    """A simple pipeline to load image."""

    def __call__(self, results):
        """Call function to load images into results.

        Args:
            results (dict): A result dict contains the file name
                of the image to be read.

        Returns:
            dict: ``results`` will be returned containing loaded image.
        """

        if isinstance(results["img"], str):
            results["filename"] = results["img"]
            results["ori_filename"] = results["img"]
        else:
            results["filename"] = None
            results["ori_filename"] = None
        img = mmcv.imread(results["img"])
        results["img"] = img
        results["img_shape"] = img.shape
        results["ori_shape"] = img.shape
        return results


class Infer:
    def __init__(self):
        self.initialized = False

    # copied over from https://github.com/aws/amazon-sagemaker-examples/blob/main/advanced_functionality/multi_model_bring_your_own/container/model_handler.py
    def initialize(self, context):
        """
        Initialize model. This will be called during model loading time
        :param context: Initial context contains model server system properties.
        :return:
        """
        self.initialized = True
        properties = context.system_properties
        # Contains the url parameter passed to the load request
        model_path = properties.get("model_dir")
        checkpoint_filename = context.model_name
        logging.info("Model directory: {}, {}".format(checkpoint_filename, properties))
        logging.info(f"Configuration::: {list(os.walk(model_path))}" )
        # gpu_id = properties.get("gpu_id")
        self.load_model_config_file(model_path, checkpoint_filename)

        # load model
        self.load_model()

    def load_model(self):
        """
        Load model based on configuration loaded configuration files.
        """
        self.config = mmengine.Config.fromfile(self.config_filename)
        self.config.model.pretrained = None
        self.config.model.train_cfg = None
        self.model = build_segmentor(
            self.config.model, test_cfg=self.config.get("test_cfg")
        )
        if self.checkpoint_filename is not None:
            self.checkpoint = load_checkpoint(
                self.model, self.checkpoint_filename, map_location="cpu"
            )
            self.model.CLASSES = self.checkpoint["meta"]["CLASSES"]
            self.model.PALETTE = self.checkpoint["meta"]["PALETTE"]
        self.model.cfg = self.config  # save the config in the model for convenience
        self.model.to("cuda:0")
        self.model.eval()
        self.device = next(self.model.parameters()).device


    def infer(self, images):
        """
        Infer on provided images
        Args:
            images (list): List of images
        """
        test_pipeline = self.config.data.test.pipeline
        test_pipeline = Compose(test_pipeline)
        data = []
        if type(images) != list:
            images = [images]

        for image in images:
            image_data = dict(img_info=dict(filename=image))
            image_data['seg_fields'] = []
            image_data['img_prefix'] = DOWNLOAD_FOLDER
            image_data['seg_prefix'] = DOWNLOAD_FOLDER 
            image_data = test_pipeline(image_data)
            data.append(image_data)
        data = collate(data, samples_per_gpu=len(images))
        if next(self.model.parameters()).is_cuda:
            # scatter to specified GPU
            data = scatter(data, [self.device])[0]
        else:
            data["img_metas"] = [i.data[0] for i in data["img_metas"]]

        # forward the model
        with torch.no_grad():
            result = self.model(return_loss=False, rescale=True, **data)
        return result

    def load_model_config_file(self, model_path, model_name):
        """
        Get the model config based on the selected model filename.
        This assume config exist in the CONFIG_DIR

        :param model_dir: Path to the directory with model artifacts
        :return: model config file
        """
        from glob import glob
        model_files = glob(f"{model_path}/*")
        logging.info(f"Configuration:: {model_files}")
        model_name = model_files[-1]
        splits = os.path.basename(model_name).replace('.pth', '').split('_')
        username = splits[0] 
        experiment = "_".join(splits[1:])

        self.config_filename = CONFIG_DIR.format(experiment=experiment)
        self.checkpoint_filename = f"{model_path}/{model_name}"
        logging.info("Model config for user {}: {}".format(username, self.config_filename))

    def postprocess(self, results, files):
        """
        Postprocess results to prepare geojson based on the images

        :param results: list of results from infer method
        :param files: list of files on which the inference was performed

        :return: GeoJSON of detected features
        """
        transforms = list()
        geojson_list = list()
        for tile in files:
            with rasterio.open(tile) as raster:
                transforms.append(raster.transform)
        for index, result in enumerate(results):
            detections = PostProcess.extract_shapes(result, transforms[index])
            detections = PostProcess.remove_intersections(detections)
            geojson = PostProcess.convert_to_geojson(detections)
            for geometry in geojson:
                updated_geometry = PostProcess.convert_geojson(geometry)
                geojson_list.append(updated_geometry)
        return {"type": "FeatureCollection", "features": geojson_list}
    
    def handle(self, data, context):
        """
        Call preprocess, inference and post-process functions
        :param data: input data
        :param context: mms context
        """
        model_out = self.infer([data])
        return self.postprocess([model_out], [data])


_service = Infer()

def handle(data, context):
    logging.info(f"Data to be processed: {data}")
    if not _service.initialized:
        _service.initialize(context)
    if data is None:
        return None
    return _service.handle(data, context)
