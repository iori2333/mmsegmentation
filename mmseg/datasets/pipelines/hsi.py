import os.path as osp
from mmcv.parallel import DataContainer as DC
import mmcv
import numpy as np
import torch

from ..builder import PIPELINES
from .formating import to_tensor
from .transforms import PhotoMetricDistortion, RandomCrop, RandomRotate, Resize


@PIPELINES.register_module()
class LoadHSIFromFile:

    @staticmethod
    def _read_hsd(data_dict):
        height = data_dict['height']
        width = data_dict['width']
        SR = data_dict['SR']
        average = data_dict['average']
        coeff = data_dict['coeff']
        scoredata = data_dict['scoredata']

        temp = torch.mm(scoredata, coeff)
        data = (temp + average).reshape((height, width, SR))

        return data.numpy()

    def __init__(self,
                 to_float32=True,
                 color_type='color',
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='cv2') -> None:
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        hsiname = osp.join(results['img_prefix'],
                           results['img_info']['hsiname'])
        hsi_dict = torch.load(hsiname)
        hsi = self._read_hsd(hsi_dict)
        # hsi = torch.rand((1889, 1422, 128))
        results['filename'] = hsiname
        results['ori_filename'] = results['img_info']['hsiname']
        results['img'] = hsi
        results['img_shape'] = hsi.shape
        results['ori_shape'] = hsi.shape
        results['pad_shape'] = hsi.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(hsi.shape) < 3 else hsi.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32},'
        repr_str += f"color_type='{self.color_type}')"
        return repr_str


@PIPELINES.register_module()
class PhotoMetricDistortionHSI(PhotoMetricDistortion):

    def __init__(self, *args, **kwargs):
        super(PhotoMetricDistortionHSI, self).__init__(*args, **kwargs)

    def convert(self, img, alpha=1, beta=0):
        img = img * alpha + beta / 255.0
        return img

    def __call__(self, results):
        hsi = results['hsi_img']
        hsi = (hsi - hsi.min()) / (hsi.max() - hsi.min())
        # random brightness
        hsi = self.brightness(hsi)

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        hsi = self.contrast(hsi)

        results['hsi_img'] = hsi
        return results


@PIPELINES.register_module()
class DefaultFormatBundleHSI:

    def __call__(self, results):
        """Call function to transform and format common fields in results.
        Args:
            results (dict): Result dict contains the data to convert.
        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """
        hsi = results['hsi_img']
        hsi = np.ascontiguousarray(hsi.transpose(2, 0, 1))
        results['hsi_img'] = DC(to_tensor(hsi), stack=True)
        return results

    def __repr__(self):
        return self.__class__.__name__


@PIPELINES.register_module()
class ReplaceHSI:

    def __call__(self, result):
        for key in [
                'filename', 'ori_filename', 'img', 'img_shape', 'ori_shape',
                'pad_shape', 'scale_factor', 'img_norm_cfg'
        ]:
            result[f'hsi_{key}'] = result[key]
        return result

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__ + '()'
        return repr_str


@PIPELINES.register_module()
class RandomCropHSI(RandomCrop):

    def __call__(self, results):
        img = results['img']
        hsi = results['hsi_img']
        crop_bbox = self.get_crop_bbox(img)
        if self.cat_max_ratio < 1.:
            # Repeat 10 times
            for _ in range(10):
                seg_temp = self.crop(results['gt_semantic_seg'], crop_bbox)
                labels, cnt = np.unique(seg_temp, return_counts=True)
                cnt = cnt[labels != self.ignore_index]
                if len(cnt) > 1 and np.max(cnt) / np.sum(
                        cnt) < self.cat_max_ratio:
                    break
                crop_bbox = self.get_crop_bbox(img)

        # crop the image
        img = self.crop(img, crop_bbox)
        hsi = self.crop(hsi, crop_bbox)
        img_shape = img.shape
        results['img'] = img
        results['img_shape'] = img_shape
        results['hsi_img'] = hsi
        results['hsi_img_shape'] = hsi.shape

        # crop semantic seg
        for key in results.get('seg_fields', []):
            results[key] = self.crop(results[key], crop_bbox)

        return results


@PIPELINES.register_module()
class RandomRotateHSI(RandomRotate):

    def __call__(self, results):
        rotate = True if np.random.rand() < self.prob else False
        degree = np.random.uniform(min(*self.degree), max(*self.degree))
        if rotate:
            # rotate image
            results['img'] = mmcv.imrotate(
                results['img'],
                angle=degree,
                border_value=self.pal_val,
                center=self.center,
                auto_bound=self.auto_bound)

            results['hsi_img'] = mmcv.imrotate(
                results['hsi_img'],
                angle=degree,
                border_value=self.pal_val,
                center=self.center,
                auto_bound=self.auto_bound)

            # rotate segs
            for key in results.get('seg_fields', []):
                results[key] = mmcv.imrotate(
                    results[key],
                    angle=degree,
                    border_value=self.seg_pad_val,
                    center=self.center,
                    auto_bound=self.auto_bound,
                    interpolation='nearest')
        return results


@PIPELINES.register_module()
class HSIPCA:

    @staticmethod
    def _pca(hsi: np.ndarray, n_components=32):
        h, w, c = hsi.shape
        hsi = hsi - np.expand_dims(hsi.mean(axis=0), axis=0)
        hsi = hsi.reshape(h * w, c)
        _, _, v = np.linalg.svd(hsi, full_matrices=False)
        r = np.matmul(hsi, v.T[:, :n_components])
        return r.reshape(h, w, n_components)

    def __init__(self, n_components: int) -> None:
        self.n_components = n_components

    def __call__(self, result):
        hsi = result['hsi_img']
        hsi = self._pca(hsi, self.n_components)
        result['hsi_img'] = hsi
        return result


@PIPELINES.register_module()
class NormalizeHSI:

    def __call__(self, result):
        key = 'hsi_img' if 'hsi_img' in result else 'img'
        img = result[key]
        mod = torch.sqrt(torch.sum(img**2, axis=2))
        mod = mod.unsqueeze(-1)
        img = img / mod
        img = (img - img.min()) / (img.max() - img.min())
        result[key] = img.numpy()
        return result

    def __repr__(self) -> str:
        return self.__class__.__name__ + '()'
