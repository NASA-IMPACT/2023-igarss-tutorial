import cv2
import numpy as np
import rasterio
import rasterio.warp

from geojson import Feature, Polygon
from PIL import Image, ImageDraw
from rasterio.crs import CRS
from scipy.interpolate import splprep, splev
from shapely import geometry

AREA_THRESHOLD = 0.05
MIN_POINTS = 3
PREDICT_THRESHOLD = 0.5
BLUR_FACTOR = (15, 15)
BLUR_THRESHOLD = 127


class PostProcess:
    @classmethod
    def prepare_bitmap(cls, predictions, width, height):
        predictions = predictions.reshape((height, width))
        # return reshaped raw array instead of bitmap
        return predictions

    @classmethod
    def extract_shapes(cls, predictions, transform):
        bitmap = (predictions > PREDICT_THRESHOLD).astype(dtype="uint8") * 255
        img_blurred = cv2.blur(bitmap, BLUR_FACTOR)
        thresholded_img = (img_blurred > BLUR_THRESHOLD).astype(dtype="uint8") * 255
        contours, _ = cv2.findContours(
            np.asarray(thresholded_img, dtype="uint8"),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        shape = bitmap.shape
        smoothened = list()
        for contour in contours:
            length = len(contour)
            if length > MIN_POINTS:
                y, x = contour.T
                x = x.tolist()[0]
                y = y.tolist()[0]
                knots_vector, params = splprep([x, y], s=3.0, quiet=1)
                new_params = np.linspace(params.min(), params.max(), 25)
                x_new, y_new = splev(new_params, knots_vector, der=0, ext=3)
                res_array = list()
                # calculate score here
                new_polygon = list()
                for pair in zip(x_new, y_new):
                    pair = list(pair)
                    # hack to make sure the shapes are inside a boundary
                    # Working on a fix, until then this is the reality
                    # we live in
                    if pair[0] > shape[0]:
                        pair[0] = shape[0]
                    if pair[1] > shape[1]:
                        pair[1] = shape[1]
                    if pair[0] < 0:
                        pair[0] = 0
                    if pair[1] < 0:
                        pair[1] = 0
                    new_polygon.append((pair[0], pair[1]))
                    res_array.append(
                        cls.convert_xy_to_latlon(pair[0], pair[1], transform)
                    )
                img = Image.new("L", (shape[0], shape[1]), 0)
                ImageDraw.Draw(img).polygon(new_polygon, outline=1, fill=1)
                mask = np.where(np.array(img).T > 0)
                score = predictions[mask]
                score_length = len(score)
                score = sum(score) / score_length
                res_array.append(res_array[0])
                smoothened.append([np.asarray(res_array), score])
        return smoothened

    @classmethod
    def convert_to_geojson(cls, shapes):
        geojson_dict = []
        for id_, shape in enumerate(shapes):
            geojson_dict.append(
                Feature(
                    properties={
                        "score": shape[1],
                    },
                    geometry=Polygon(
                        [[[float(lon), float(lat)] for lon, lat in shape[0]]]
                    ),
                )
            )
        # just return the list of features, wrapping is done in main.py
        return geojson_dict

    @classmethod
    def convert_geojson(cls, results):
        feature = results["geometry"]
        feature_proj = rasterio.warp.transform_geom(
            CRS.from_epsg(3857), CRS.from_epsg(4326), feature
        )
        results["geometry"] = feature_proj
        return results

    @classmethod
    def remove_intersections(cls, shapes):
        computed_polygons = list()
        selected_indices = list()
        selected_shapes = list()
        areas = list()
        for shape in shapes:
            polygon = geometry.Polygon(shape[0])
            if polygon.is_valid:
                area = polygon.area
                if area > AREA_THRESHOLD:
                    computed_polygons.append(polygon)
                    areas.append(area)
                    selected_shapes.append(shape)
        if len(areas) > 0:
            computed_polygons = np.asarray(computed_polygons)
            polygon_indices = list(np.argsort(areas))
            while len(polygon_indices) > 0:
                selected_index = polygon_indices[-1]
                selected_polygon = computed_polygons[selected_index]
                selected_indices.append(selected_index)
                polygon_indices.remove(selected_index)
                indices_holder = polygon_indices.copy()
                for index in indices_holder:
                    if computed_polygons[index].intersects(selected_polygon):
                        polygon_indices.remove(index)
        return np.array(selected_shapes)[selected_indices]

    @classmethod
    def convert_xy_to_latlon(cls, row, col, transform):
        """
        uses rasterio transform module to convert row, col of an image to
        its respective lat, lon coordinates
        """
        transform = rasterio.transform.guard_transform(transform)
        return rasterio.transform.xy(transform, row, col, offset="center")
