import os
import math
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate
from collections import OrderedDict
from utils.camera_utils import loadCam
from concurrent.futures import ThreadPoolExecutor,as_completed
from utils.graphics_utils import getProjectionMatrix, getWorld2ViewCUDA

def construct_cam_info(image_height, image_width, extrinsic, intrinsic, disturb=None, trans=None, scale=1.0, znear=0.01, zfar=100.0):
    extrinsic = extrinsic.float()
    intrinsic = intrinsic.float()
    if trans is None:
        trans = torch.zeros(3, dtype=extrinsic.dtype, device=extrinsic.device)
    R = extrinsic[:3, :3]
    T = extrinsic[:3, 3]

    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    FoVx = 2 * math.atan(image_width / (2 * fx))
    FoVy = 2 * math.atan(image_height / (2 * fy))

    world_view_transform = getWorld2ViewCUDA(R, T, trans, scale, disturb).permute(1, 0)
    projection_matrix = getProjectionMatrix(znear=znear, zfar=zfar, fovX=FoVx, fovY=FoVy)
    projection_matrix = projection_matrix.to(device=world_view_transform.device, dtype=torch.float32).transpose(0, 1)
    full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
    camera_center = world_view_transform.inverse()[3,:3]


    x = {
        "FoVx": torch.tensor([FoVx], dtype=torch.float32),
        "FoVy": torch.tensor([FoVy], dtype=torch.float32),
        "image_name": 0,
        "image_height": torch.tensor(image_height),
        "image_width": torch.tensor(image_width),
        "camera_center": camera_center.unsqueeze(0),               # (1,3) on device
        "world_view_transform": world_view_transform.unsqueeze(0), # (1,4,4) on device
        "full_proj_transform": full_proj_transform.unsqueeze(0),   # (1,4,4) on device
        "intrinsic": intrinsic.to("cpu").float(),
        "extrinsic": world_view_transform.permute(1, 0).to("cpu").float(),  # Rw2c on CPU (matches prior)
    }

    return x,None,None


class GSDataset(Dataset):
    def __init__(
        self,
        cameras,
        scale=1,
        no_dynamic_res=False,
        preload=False,
        cache_size=0,
        max_resolution_level=5,
        num_preload_workers=8,
        znear=0.01,
        zfar=100.0,
        gpu_resize_cache=False,
        gpu_resize_mode="bicubic",
        gpu_cache_after_resize=False,
    ):
        self.cameras = cameras
        self.scale = scale
        self.no_dynamic_res = no_dynamic_res
        self.image_scale = scale
        self.image_ini_scale = scale
        self.max_resolution_level = max_resolution_level
        self.cache_size = max(0, int(cache_size))
        self.num_preload_workers = max(1, int(num_preload_workers))
        self.znear = znear
        self.zfar = zfar
        self.gpu_resize_cache = bool(gpu_resize_cache)
        self.gpu_resize_mode = str(gpu_resize_mode)
        self.gpu_cache_after_resize = bool(gpu_cache_after_resize)
        self._cache = OrderedDict()
        if self.no_dynamic_res:
            self.image_re_scale = self.max_resolution_level
        else:
            self.image_re_scale = 1
        self.view_count = len(cameras)
        self.preload = preload
        if self.preload:
            self.views_data_dict = self._preload_data()
    
    def _preload_data(self, resolution_level=None):
        """Preload CPU image tensors.

        This is intentionally opt-in and limited to one resolution level at a
        time. Preloading every image at every dynamic resolution can use a
        large amount of host memory on city-scale scenes. Prefer lazy loading
        with a small LRU cache for normal training.
        """
        num_views = len(self.cameras)
        views_data_dict = {}
        with ThreadPoolExecutor(max_workers=self.num_preload_workers) as executor:
            res = self.image_re_scale if resolution_level is None else int(resolution_level)
            views_data = [None] * num_views
            image_scale = self.image_ini_scale * res / self.max_resolution_level
            futures = {
                executor.submit(self._load_single_sample, camera, image_scale): idx
                for idx, camera in enumerate(self.cameras)
            }
            for future in as_completed(futures):
                idx = futures[future]
                views_data[idx] = future.result()
            views_data_dict[res] = views_data
        return views_data_dict
    
    def _load_single_sample(self, camera, image_scale=None):
        if image_scale is None:
            image_scale = self.image_ini_scale * self.image_re_scale / self.max_resolution_level
        viewpoint_cam = loadCam(
            camera,
            image_scale,
            znear=self.znear,
            zfar=self.zfar,
            gpu_resize=self.gpu_resize_cache,
            gpu_resize_mode=self.gpu_resize_mode,
            gpu_cache_after_resize=self.gpu_cache_after_resize,
        )
        x = {
            "FoVx": viewpoint_cam.FoVx,
            "FoVy": viewpoint_cam.FoVy,
            "image_name": viewpoint_cam.image_name,
            "image_height": viewpoint_cam.image_height,
            "image_width": viewpoint_cam.image_width,
            "camera_center": viewpoint_cam.camera_center,
            "world_view_transform": viewpoint_cam.world_view_transform,
            "full_proj_transform": viewpoint_cam.full_proj_transform,
            "intrinsic": torch.as_tensor(camera.intrinsic, dtype=torch.float32),
            "extrinsic": viewpoint_cam.world_view_transform.permute(1, 0).float() #Rw2c
        }
        y = viewpoint_cam.original_image
        z = viewpoint_cam.depth_inv
        return x,y,z

    def _get_cached_sample(self, idx):
        cache_key = (idx, self.image_re_scale)
        if cache_key in self._cache:
            self._cache.move_to_end(cache_key)
            return self._cache[cache_key]

        sample = self._load_single_sample(self.cameras[idx])
        if self.cache_size > 0:
            self._cache[cache_key] = sample
            self._cache.move_to_end(cache_key)
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return sample
    
    def __getitem__(self, index):
        idx = index % len(self.cameras)
        if not self.preload:
            return self._get_cached_sample(idx)
        else:
            return self.views_data_dict[self.image_re_scale][idx]
    
    def __len__(self):
        return len(self.cameras)

    def reset_down_ratio(self, res_ratio):
        self.image_re_scale = int(res_ratio)
        self._cache.clear()
        if self.preload and self.image_re_scale not in self.views_data_dict:
            self.views_data_dict.clear()
            self.views_data_dict.update(self._preload_data(self.image_re_scale))


class ChunkedCacheDataLoader:
    """Cache a fixed-size image chunk and reuse it for several train steps.

    This avoids decoding and resizing a large source image on every iteration.
    A chunk is cached at the dataset's current dynamic-resolution level; when
    the dataset level changes, the next iterator pass rebuilds the cache.
    """

    def __init__(
        self,
        dataset,
        cache_size,
        iterations_per_cache,
        seed=0,
        shuffle=True,
        num_workers=4,
    ):
        self.dataset = dataset
        self.cache_size = max(1, int(cache_size))
        self.iterations_per_cache = max(1, int(iterations_per_cache))
        self.shuffle = bool(shuffle)
        self.num_workers = max(1, int(num_workers))
        self.generator = torch.Generator()
        self.generator.manual_seed(int(seed))
        self.indices = list(range(len(dataset)))
        self.cursor = 0
        self.cached = []
        self.cached_indices = []
        self.cached_resolution_level = None

    def __len__(self):
        return len(self.dataset)

    def _reshuffle_indices(self):
        if self.shuffle:
            order = torch.randperm(len(self.indices), generator=self.generator).tolist()
            self.indices = [self.indices[i] for i in order]

    def _next_chunk_indices(self):
        if len(self.indices) == 0:
            return []
        if self.cursor == 0:
            self._reshuffle_indices()

        chunk = []
        while len(chunk) < min(self.cache_size, len(self.indices)):
            remaining = len(self.indices) - self.cursor
            take = min(min(self.cache_size, len(self.indices)) - len(chunk), remaining)
            chunk.extend(self.indices[self.cursor : self.cursor + take])
            self.cursor += take
            if self.cursor >= len(self.indices):
                self.cursor = 0
                self._reshuffle_indices()
        return chunk

    def _cache_data(self, indices):
        cached = []
        desc = (
            f"caching {len(indices)} images "
            f"(level {self.dataset.image_re_scale}, first {indices[0]})"
        )
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            for sample in tqdm(
                executor.map(self.dataset.__getitem__, indices),
                total=len(indices),
                desc=desc,
            ):
                cached.append(sample)
        return cached

    def _rebuild_cache(self):
        self.cached_indices = self._next_chunk_indices()
        self.cached = self._cache_data(self.cached_indices)
        self.cached_resolution_level = self.dataset.image_re_scale

    def __iter__(self):
        while True:
            if len(self.dataset) == 0:
                return
            if (
                len(self.cached) == 0
                or self.cached_resolution_level != self.dataset.image_re_scale
            ):
                self._rebuild_cache()

            for _ in range(self.iterations_per_cache):
                sample_idx = torch.randint(
                    len(self.cached),
                    size=(1,),
                    generator=self.generator,
                ).item()
                yield default_collate([self.cached[sample_idx]])

            self.cached = []
            self.cached_indices = []

class CacheDataLoader(torch.utils.data.DataLoader):
    def __init__(
            self,
            dataset: torch.utils.data.Dataset,
            max_cache_num: int,
            shuffle: bool,
            seed: int = -1,
            distributed: bool = False,
            world_size: int = -1,
            global_rank: int = -1,
            **kwargs,
    ):
        assert kwargs.get("batch_size", 1) == 1, "only batch_size=1 is supported"

        self.dataset = dataset

        super().__init__(dataset=dataset, **kwargs)

        self.shuffle = shuffle
        self.max_cache_num = max_cache_num

        # image indices to use
        self.indices = list(range(len(self.dataset)))
        if distributed is True and self.max_cache_num != 0:
            assert world_size > 0
            assert global_rank >= 0
            image_num_to_use = math.ceil(len(self.indices) / world_size)
            start = global_rank * image_num_to_use
            end = start + image_num_to_use
            indices = self.indices[start:end]
            indices += self.indices[:image_num_to_use - len(indices)]
            self.indices = indices

            print("#{} distributed indices (total: {}): {}".format(os.getpid(), len(self.indices), self.indices))

        # cache all images if max_cache_num > len(dataset)
        if self.max_cache_num >= len(self.indices):
            self.max_cache_num = -1

        self.num_workers = kwargs.get("num_workers", 0)

        if self.max_cache_num < 0:
            # cache all data
            print("cache all images")
            self.cached = self._cache_data(self.indices)

        # use dedicated random number generator foreach dataloader
        if self.shuffle is True:
            assert seed >= 0, "seed must be provided when shuffle=True"
            self.generator = torch.Generator()
            self.generator.manual_seed(seed)
            print("#{} dataloader seed to {}".format(os.getpid(), seed))

    def _cache_data(self, indices: list):
        # TODO: speedup image loading
        cached = []
        if self.num_workers > 0:
            with ThreadPoolExecutor(max_workers=self.num_workers) as e:
                for i in tqdm(
                        e.map(self.dataset.__getitem__, indices),
                        total=len(indices),
                        desc="#{} caching images (1st: {})".format(os.getpid(), indices[0]),
                ):
                    cached.append(i)
        else:
            for i in tqdm(indices, desc="#{} loading images (1st: {})".format(os.getpid(), indices[0])):
                cached.append(self.dataset.__getitem__(i))

        return cached

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset.__getitem__(idx)

    def __iter__(self):
        # TODO: support batching
        if self.max_cache_num < 0:
            if self.shuffle is True:
                indices = torch.randperm(len(self.cached), generator=self.generator).tolist()  # shuffle for each epoch
                # print("#{} 1st index: {}".format(os.getpid(), indices[0]))
            elif self.shuffle is None and self.sampler is not None:
                indices = list(self.sampler)
            else:
                indices = list(range(len(self.cached)))

            for i in indices:
                yield self.cached[i]
        else:
            if self.shuffle is True:
                indices = torch.randperm(len(self.indices), generator=self.generator).tolist()  # shuffle for each epoch
                # print("#{} 1st index: {}".format(os.getpid(), indices[0]))
            elif self.shuffle is None and self.sampler is not None:
                indices = list(self.sampler)
            else:
                indices = self.indices.copy()

            # print("#{} self.max_cache_num={}, indices: {}".format(os.getpid(), self.max_cache_num, indices))

            if self.max_cache_num == 0:
                # no cache
                for i in indices:
                    yield self.__getitem__(i)
            else:
                # cache
                # the list contains the data have not been cached
                not_cached = indices.copy()

                while not_cached:
                    # select self.max_cache_num images
                    to_cache = not_cached[:self.max_cache_num]
                    del not_cached[:self.max_cache_num]

                    # cache
                    try:
                        del cached
                    except:
                        pass
                    cached = self._cache_data(to_cache)

                    for i in cached:
                        yield i
