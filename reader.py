import os
import cv2
import numpy as np
import subprocess as sp

from collections import deque, defaultdict, Counter
from pathlib import Path
from itertools import count, repeat, starmap
from functools import cached_property
from operator import itemgetter, attrgetter, sub

import config

if config.PROCESSING_FP_DTYPE == 'float16':
    PROCESSING_FP_DTYPE = np.float16
elif config.PROCESSING_FP_DTYPE == 'float32':
    PROCESSING_FP_DTYPE = np.float32
elif config.PROCESSING_FP_DTYPE == 'float64':
    PROCESSING_FP_DTYPE = np.float64
else:
    print('config any of input data type: float16/float32/float64')
    
class CarouselBuffer:
    def __init__(self, 
                 raw_frame_shape=config.FULL_HD_FRAME_SHAPE, 
                 letterbox=None, 
                 num_of_frames=config.NUM_OF_MODEL_INPUTS,
                 align_to=16, 
                 dtype=PROCESSING_FP_DTYPE, 
                 normalizer=config.NORMALIZER):
        self.raw_frame_shape = raw_frame_shape
        self.letterbox = letterbox
        self.num_of_frames = num_of_frames
        self.align_to = align_to
        self.dtype = dtype
        self.normalizer = normalizer
        
    @cached_property
    def buffers(self):
        aligned_shape = self.align_shape(self.content_shape, align_to=self.align_to)
        buffers = np.zeros((self.num_of_frames, 1, *aligned_shape, 3), dtype=self.dtype)
        buffers = np.ascontiguousarray(buffers)
        buffers = deque(buffers, maxlen=self.num_of_frames)
        return buffers

    @property
    def buffer(self, n=-1):
        return self.buffers[n]
    
    @property
    def letterbox(self):
        return self._letterbox
    
    @letterbox.setter
    def letterbox(self, list_of_slices):
        if list_of_slices:
            self._letterbox = list_of_slices
        else:
            self._letterbox = self.slices_from_shape(self.raw_frame_shape)
        self.reset()
        
    @cached_property
    def content_shape(self):
        return self.shape_from_slices(self.letterbox)
    
    @property
    def content_height(self):
        return self.content_shape[0]    
    
    @property
    def content_width(self):
        return self.content_shape[1]
        
    @property
    def frame(self):
        return self.buffer[0, :self.content_height, :self.content_width]
    
    @frame.setter
    def frame(self, raw_frame_from_video):
        if raw_frame_from_video is not None:
            self._rotate()
            frame = self.frame
            frame[:] = raw_frame_from_video.__getitem__((*self.letterbox,))
            frame += config.BIAS
            frame /= config.SCALE 
            
    @staticmethod
    def shape_from_slices(list_of_slices, container=tuple):
        return container(starmap(sub, map(attrgetter('stop', 'start'), list_of_slices)))
    
    @staticmethod
    def slices_from_shape(shape, shifts=(0, 0), container=list):
        return container(slice(shift, shift + size, None) for size, shift in zip(shape[:2], shifts))
    
    @staticmethod
    def align_shape(shape, align_to=16, dtype=np.int32):
        return type(shape)(dtype(np.ceil(dtype(shape) / align_to) * align_to))
        
    def _rotate(self, n=-1):
        self.buffers.rotate(n)
        
    def reset(self, cached_properties=('buffers', 'content_shape')):
        for attr in cached_properties:
            if hasattr(self, attr):
                delattr(self, attr)
                
class VideoStream():
    def __init__(self, path_to_video, 
                 num_of_frames_in_sequence=config.NUM_OF_MODEL_INPUTS,
                 unet_depth=config.MODEL_POOLING_DEPTH, 
                 upscale_by=config.UPSCALE_BY, 
                 dtype=PROCESSING_FP_DTYPE):
        self.path_to_video = Path(path_to_video)
        self.num_of_frames_in_sequence = num_of_frames_in_sequence
        self.required_multiple_of_2 = 2 ** (unet_depth - 1)
        self.upscale_by = upscale_by
        self.dtype = dtype
        
    def __enter__(self):
        self.video = cv2.VideoCapture(self.path_to_video.as_posix(), cv2.CAP_FFMPEG)
        ffmpeg_cmd = [ 'ffmpeg', 
                      '-i', self.path_to_video.as_posix(), 
                      '-r', str(self.fps), # FPS
                      '-vf', 'scale=in_color_matrix=bt709:flags=full_chroma_int+accurate_rnd,format=rgb24', # read BT709
                      '-pix_fmt', 'rgb24',      # opencv requires bgr24 pixel format.
                      '-vcodec', 'rawvideo',
                      '-an','-sn',              # disable audio processing
                      '-f', 'image2pipe', 
                      '-'
                     ]
        print(' '.join(ffmpeg_cmd))
        self.pipe = sp.Popen(ffmpeg_cmd, stdout = sp.PIPE, stdin = sp.DEVNULL, stderr = sp.DEVNULL, bufsize=10**9)
        self.bytes_per_frame = self.raw_frame_height * self.raw_frame_width * 3
        self.fbuffer = np.zeros((self.raw_frame_height, self.raw_frame_width, 3), dtype=np.uint8)
        self.cbuffer = CarouselBuffer(
            raw_frame_shape = self.fbuffer.shape,
            letterbox = None,
            num_of_frames = self.num_of_frames_in_sequence,
            align_to = self.required_multiple_of_2,
            dtype = self.dtype
        )
        return self
        
    def __exit__(self, exc_type, exc_value, traceback):
        self.video.release()
        self.pipe.stdout.close()
        
    @cached_property
    def fps(self):
        return self.video.get(cv2.CAP_PROP_FPS)
    
    @cached_property
    def num_of_frames_in_video(self):
        return self.video.get(cv2.CAP_PROP_FRAME_COUNT)
   
    @cached_property
    def idx_of_mid_frame_in_sequence(self):
        return self.num_of_frames_in_sequence // 2
    
    @cached_property
    def raw_frame_width(self):
        return int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))

    @cached_property
    def raw_frame_height(self):
        return int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    @cached_property    
    def up_frame_height(self):
        return self.raw_frame_height*self.upscale_by
    
    @cached_property    
    def up_frame_width(self):
        return self.raw_frame_width*self.upscale_by
    
    @property
    def position(self):
        return self.video.get(cv2.CAP_PROP_POS_FRAMES)
    
    @position.setter
    def position(self, zero_based_frame_number):
        self.video.set(cv2.CAP_PROP_POS_FRAMES, zero_based_frame_number)
        
    @property
    def telomeres(self):
        return repeat(self.cbuffer.buffer, self.idx_of_mid_frame_in_sequence)
    
    def rewind(self, zero_based_frame_number=0):
        self.position = zero_based_frame_number

    def stream_of_padded_frames(self, telomeres=True, loop=False, cacher=False):
        self.rewind()
        for frame_id in count(0):
            
            if cacher:
                # чтобы быстро бегать по файлу при анализе каше читаем с помощью opencv 
                ret, frame = self.video.read(self.fbuffer)
                if ret:
                    self.cbuffer.frame = frame[...,::-1]
            else:
                # в остальных случаях используем pipe, который дает возможность правильно читать цвета bt709
                try:
                    frame = self.pipe.stdout.read(self.bytes_per_frame)
                    frame =  np.fromstring(frame, dtype='uint8') # convert read bytes to np
                    self.cbuffer.frame = frame.reshape((self.raw_frame_height, self.raw_frame_width, 3))
                    ret = True
                except:
                    ret = False
                    
            if ret:
                if frame_id == 0 and telomeres:
                    yield from self.telomeres
                yield self.cbuffer.buffer
            elif loop:
                self.rewind()
            else:
                if frame_id > 0 and telomeres:
                    yield from self.telomeres
                #self.pipe.stdout.flush()
                break      
    
    def stream_of_padded_sequences(self, loop=False):
        for frame_id, _ in enumerate(self.stream_of_padded_frames(loop=loop), 1):
            if frame_id < self.num_of_frames_in_sequence:
                continue
            yield self.cbuffer.buffers
            
    def _detect_letterbox_coarse(self, grayscale_frame, trim_top_and_bottom=True, trim_left_and_right=True,
                                 profiler=np.max, threshold=32/config.NORMALIZER, compare=np.less_equal):
        # Дефолтные значения слайсов покрывают 100% площади кадра
        letterbox = self.cbuffer.slices_from_shape(grayscale_frame.shape)
        # Обрабатываем горизонтальное и вертикальное сечения кадра, чтобы подрезать с выбранных сторон
        for axis, trim_along_this_axis in enumerate((trim_top_and_bottom, trim_left_and_right)):
            if not trim_along_this_axis:
                continue
            # Профиль (подобие гистограммы) считается функцией profiler
            profile_across_axis = profiler(grayscale_frame, axis=1-axis)
            mask = compare(profile_across_axis, threshold)
            print(f'proportion of pixels above cacher threshold {np.sum(mask)/mask.shape}')
            profile_across_axis = np.ma.masked_array(profile_across_axis, mask=mask)
            try:
                # Индексы первой и последней строк (столбцов) полезного содержимого кадра
                start, stop = np.ma.notmasked_edges(profile_across_axis)
                # Для перехода от индексов к слайсу надо увеличить stop на единицу
                content = slice(start, stop + 1, None)
            except:
                raise 'Too high cacher threshold'
            else:
                letterbox[axis] = content
        return letterbox
        
    def detect_letterbox(self, skip_credits_ratio=0.0, num_of_probes=config.NUM_OF_PROBES):
        # Буфер для усреднённого накопленного мультикадра
        aggregated_frame = np.zeros_like(self.cbuffer.frame)
        # Вычисляем относительные (относительно длительности видео) позиции за исключением начальных и конечных титров,
        # доля которых от начала и конца задаётся skip_credits_ratio. С количеством проб (num_of_probes) лучше не мелочиться,
        # и задавать не менее 100 проб, а лучше 200+. Встречаются фильмы, где сужение кадра (расширение чёрных полос) в
        # отдельных эпизодах является режиссёрским приёмом, и нужно отличать такие случаи от постоянных чёрных полос.
        probe_positions = np.linspace(skip_credits_ratio, 1.0 - skip_credits_ratio, num_of_probes)
        # Логичнее для self.position использовать cv2.CAP_PROP_POS_AVI_RATIO, но там баг в OpenCV,
        # поэтому втыкаем костыль и переходим от относительных позиций к абсолютным (номера кадров начиная с 0)
        probe_positions *= (self.num_of_frames_in_video - 1)  # преобразовывать в int не обязательно
        # Будем использовать стандартную читалку видео, но косвенным способом
        frame_reading_trigger = self.stream_of_padded_frames(telomeres=False, loop=False, cacher=True)        
        # Проматываем видео ко всем интересующим позициям
        for self.position in probe_positions:
            print(f'collecting cacher for position {self.position}', end='\r')
            # Неявно триггерим чтение кадра
            next(frame_reading_trigger)
            # Накапливаем в цветовых каналах самые яркие пиксели
            np.maximum(aggregated_frame, self.cbuffer.frame, out=aggregated_frame)
        print('\ndone')
        # Схлопываем цветовые каналы в grayscale, выбирая максимальные значения
        aggregated_frame = aggregated_frame.max(axis=-1)
        print(np.sum(aggregated_frame[0])/aggregated_frame[0].shape)
        # Теперь на основе интегрального кадра можно весьма надёжно вычислить слайсы полезного контента
        letterbox = self._detect_letterbox_coarse(aggregated_frame)
        return letterbox
    
    def remove_padding(self, padded_frame, scale=1):
        frame = padded_frame.squeeze()[
            :self.cbuffer.content_height * scale,
            :self.cbuffer.content_width * scale, 
            :        ]
        return frame
    
    def postprocess(self, frame, clip=True, apply_color_corr=True):
        # Вычисления с float16 очень медленные
        if frame.dtype == np.float16:
            frame = np.core.float32(frame)
        
        for ch in range(frame.shape[-1]):
            if apply_color_corr:
                frame[..., ch] -= config.COLOR_CORR[ch][0]
                frame[..., ch] *= config.SCALE * config.COLOR_CORR[ch][1]
            else:
                frame[..., ch] *= config.SCALE
        frame -= config.BIAS
        if clip:    
            frame = np.clip(frame, 0, 255)
         
        return np.core.uint8(frame)
        #return np.core.uint8(frame[:,:,::-1])
    
    def add_letterbox(self, frame):
        frame_h, frame_w = frame.shape[:2]
        up = (self.up_frame_height - frame_h)//2
        left = (self.up_frame_width - frame_w)//2
        letterbox = np.zeros((self.up_frame_height, self.up_frame_width, 3), dtype=np.uint8)
        letterbox[up:up+frame_h, left:left+frame_w, :] = frame
        return letterbox
    
