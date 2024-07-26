import subprocess as sp
import os
import cv2

class FFMPEG():
    def __init__(self, dst_frame_height, dst_frame_width, fps, Mbitrate, dst_path='', 
                 pix_fmt='rgb24', vcodec='h264_nvenc'):
        self.dst_path = dst_path
        self.params = dict(
                           #vsync='0',
                           #avoid_negative_ts='1',
                           #sws_flags='spline+accurate_rnd+full_chroma_int',
                           #color_trc='2',
                           #colorspace='2',
                           #color_primaries='2',
                           pix_fmt=pix_fmt,
                           s=f'{dst_frame_width}x{dst_frame_height}',
                           r=f'{fps}',
                           i='-',
                           vcodec=vcodec,
                           #vf='colorspace=all=bt709:iall=bt601-6-625:fast=1',
                           #vf='format=rgb24',
                           #profile=':v high',
                           #profile=':v high444',
                           #preset='medium',
                           b=f':v {Mbitrate}M'
                           )
        
    def __init_cmd(self, rewrite=True):
        self.command = f'ffmpeg '
        if rewrite:
            self.command += '-y '
        
    def __finish_cmd(self):
        self.command += self.dst_path
            
    def __add_items(self, items):
        for key in items.keys():
            value = items[key]
            self.command += '-' + key
            if not value.startswith((':')):
                self.command += ' '
            self.command += value + ' ' 
            
    def __build_cmd(self, prefix, rewrite=True):
        self.__init_cmd(rewrite)
        self.__add_items(prefix)
        self.__add_items(self.params)
        self.__finish_cmd()
         
    @property
    def stdout(self):
        source = dict(f='rawvideo',
                      vcodec='rawvideo'
                     )
        self.__build_cmd(source)
        return self.command
    
        

class FFMPEG_BAK():
    def __init__(self, dst_frame_height, dst_frame_width, fps, Mbitrate, dst_path='', 
                 pix_fmt='rgb24', vcodec='h264_nvenc'):
        self.dst_path = dst_path
        self.params = dict(
                           #vsync='0',
                           #avoid_negative_ts='1',
                           #sws_flags='spline+accurate_rnd+full_chroma_int',
                           #color_trc='2',
                           #colorspace='2',
                           #color_primaries='2',
                           
                           pix_fmt=pix_fmt,
                           #vf='colorspace=all=bt709:iall=bt601-6-625:fast=1',
                           s=f'{dst_frame_width}x{dst_frame_height}',
                           r=f'{fps}',
                           i='-',
                           vcodec=vcodec,
                           #profile=':v high',
                           #profile=':v high444',
                           #preset='medium',
                           b=f':v {Mbitrate}M'
                           )
        
    def __init_cmd(self, rewrite=True):
        self.command = f'ffmpeg '
        if rewrite:
            self.command += '-y '
        
    def __finish_cmd(self):
        self.command += self.dst_path
            
    def __add_items(self, items):
        for key in items.keys():
            value = items[key]
            self.command += '-' + key
            if not value.startswith((':')):
                self.command += ' '
            self.command += value + ' ' 
            
    def __build_cmd(self, prefix, rewrite=True):
        self.__init_cmd(rewrite)
        self.__add_items(prefix)
        self.__add_items(self.params)
        self.__finish_cmd()
         
    @property
    def stdout(self):
        source = dict(f='rawvideo',
                      vcodec='rawvideo'
                     )
        self.__build_cmd(source)
        return self.command
    
    
def num_frames(vid_path):
    cap = cv2.VideoCapture(vid_path)
    if cap.isOpened():
        num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        return int(num_frames)
    else:
        return None
        
