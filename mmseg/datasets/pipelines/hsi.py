import os.path as osp
from mmcv.parallel import DataContainer as DC
import mmcv
import numpy as np
import torch

from ..builder import PIPELINES
from .formating import to_tensor, DefaultFormatBundle
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

        return data

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
        '''
        HSI images are loaded to torch.Tensor to increase the performance.
        Any transformation applied to HSI images should be written in torch APIs 
        instead of numpy APIs.
        '''
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
        key = 'hsi_img' if 'hsi_img' in results else 'img'
        hsi = results[key]
        hsi = (hsi - hsi.min()) / (hsi.max() - hsi.min())
        # random brightness
        hsi = self.brightness(hsi)

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        hsi = self.contrast(hsi)

        results[key] = hsi
        return results


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
class RandomCropMixed(RandomCrop):

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
class RandomRotateMixed(RandomRotate):

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
class PCAHSI:

    @staticmethod
    def _pca(hsi: torch.Tensor, n_components=32):
        h, w, c = hsi.shape
        hsi = hsi - hsi.mean(0).unsqueeze(0)
        hsi = hsi.view(h * w, c)
        _, _, v = torch.linalg.svd(hsi, full_matrices=False)
        r = torch.mm(hsi, v.T[:, :n_components])
        return r.view(h, w, n_components)

    def __init__(self, n_components: int) -> None:
        self.n_components = n_components

    def __call__(self, result):
        key = 'hsi_img' if 'hsi_img' in result else 'img'
        result[key] = self._pca(result[key], self.n_components)
        return result

    def __repr__(self) -> str:
        return self.__class__.__name__ + f'(n_components={self.n_components})'


@PIPELINES.register_module()
class NormalizeHSI:

    def __call__(self, result):
        key = 'hsi_img' if 'hsi_img' in result else 'img'
        img = result[key]
        mod = torch.sqrt(torch.sum(img**2, axis=2))
        mod = mod.unsqueeze(-1)
        img = img / mod
        result[key] = img
        return result

    def __repr__(self) -> str:
        return self.__class__.__name__ + '()'


@PIPELINES.register_module()
class RandomFlipMixed:

    def __init__(self, prob=None, direction='horizontal'):
        self.prob = prob
        self.direction = direction
        if prob is not None:
            assert prob >= 0 and prob <= 1
        assert direction in ['horizontal', 'vertical']

    def __call__(self, results):
        """Call function to flip bounding boxes, masks, semantic segmentation
        maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Flipped results, 'flip', 'flip_direction' keys are added into
                result dict.
        """

        if 'flip' not in results:
            flip = True if np.random.rand() < self.prob else False
            results['flip'] = flip
        if 'flip_direction' not in results:
            results['flip_direction'] = self.direction
        axis = 1 if self.direction == 'horizontal' else 0
        if results['flip']:
            # flip image
            results['img'] = torch.flip(results['img'], dims=[axis])

            # flip segs
            for key in results.get('seg_fields', []):
                # use copy() to make numpy stride positive
                results[key] = np.flip(results[key], axis=axis)

            # flip hsi
            for key in results.get('hsi_img', []):
                # use copy() to make numpy stride positive
                results[key] = torch.flip(results[key], dims=[axis])
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(prob={self.prob})'


@PIPELINES.register_module()
class DefaultFormatBundleMixed(DefaultFormatBundle):

    def __call__(self, results):
        if 'img' in results:
            img = results['img']
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            if isinstance(img, torch.Tensor):
                img = img.permute(2, 0, 1)
            else:
                img = np.ascontiguousarray(img.transpose(2, 0, 1))
            results['img'] = DC(to_tensor(img), stack=True)
        if 'hsi_img' in results:
            img = results['hsi_img']
            img = img.permute(2, 0, 1)
            results['img'] = DC(img, stack=True)
        if 'gt_semantic_seg' in results:
            # convert to long
            results['gt_semantic_seg'] = DC(
                to_tensor(results['gt_semantic_seg'][None,
                                                     ...].astype(np.int64)),
                stack=True)
        return results
