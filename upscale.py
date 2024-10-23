import argparse

import os
import shutil
import cv2
import numpy as np
import subprocess as sp
from multiprocessing import Process, Queue, Lock
from time import time, sleep

import onnx
import onnxruntime as ort

import config 
from reader import VideoStream
from utils import FFMPEG
from utils import num_frames


NORM_METHOD = config.NORM_METHOD
REPO_DIR = config.REPO_DIR

def equalize_mean_area(q_in, q_out, iolock):
    NORM_MODE = config.NORM_MODE
    NORM_UP = config.NORM_AREA_SIZE
    NORM_DOWN = 1/NORM_UP
    
    while True:
        hires, lores = q_in.get()
        if hires is None:
            with iolock:
                q_out.put(None)
            break
        
        with iolock:
            lores = np.core.float32(lores.squeeze())
            hires = np.core.float32(hires.squeeze())
             
            # Результат записывается в заранее аллоцированный массив frame_dst_mean_as_1x1 (интерполяция linear)
            lodown = cv2.resize(lores, None, fx=NORM_DOWN*2, fy=NORM_DOWN*2, interpolation=cv2.INTER_LINEAR)
            frame_dst_mean_as_1x1 = np.zeros_like(lodown)
            cv2.resize(hires, None, frame_dst_mean_as_1x1, NORM_DOWN, NORM_DOWN, cv2.INTER_LINEAR)
                    
            if NORM_MODE=='scale':
                # Вычисляем каоэффициенты масштабирования по сетке 1х1 для FullHD кадра, причём нули на выходе
                # оставляем нулями, чтобы избежать деления на ноль, поэтому используем np.where()
                coeffs_grid_1x1 = np.where(frame_dst_mean_as_1x1 <= 0.0, 0.0, lodown / frame_dst_mean_as_1x1)
            
                # Растягиваем сетку коэффициентов в 2 раза, чтобы получить разрешение 4K (интерполяция nearest) и нормируем
                coeffs_grid = np.zeros_like(hires)
                hires *= cv2.resize(coeffs_grid_1x1, None, coeffs_grid, NORM_UP, NORM_UP, cv2.INTER_NEAREST)
                    
            elif NORM_MODE=='bias':
                coeffs_grid_1x1 = lodown - frame_dst_mean_as_1x1
            
                # Растягиваем сетку коэффициентов в 2 раза, чтобы получить разрешение 4K (интерполяция nearest) и нормируем
                coeffs_grid = np.zeros_like(hires)
                hires += cv2.resize(coeffs_grid_1x1, None, coeffs_grid, NORM_UP, NORM_UP, cv2.INTER_NEAREST)
                    
            q_out.put(hires)

def equalize_gauss(q_in, q_out, iolock):
    NORM_MODE = config.NORM_MODE
    GRAY_NORM = config.GRAY_NORM
    KERNEL_SIZE = config.NORM_AREA_SIZE + 1
    
    while True:
        hires, lores = q_in.get()
        if hires is None:
            with iolock:
                q_out.put(None)
            break
        
        with iolock:
            lores = np.core.float32(lores.squeeze())
            hires = np.core.float32(hires.squeeze())
             
            # увеличиваем размер исходного изображения до размера апскейла
            lores_mean = cv2.resize(lores, hires.shape[:2][::-1], interpolation=cv2.INTER_LINEAR)
            hires_mean = hires.copy()
            # если вдруг надумаем нормализовать интенсивность не по каждму цвету индивидуально
            # а в сером канале  
            if GRAY_NORM:
                lores_mean = cv2.cvtColor(lores_mean, cv2.COLOR_RGB2GRAY)
                hires_mean = cv2.cvtColor(hires_mean, cv2.COLOR_RGB2GRAY)
            # усредняем интенсивность по каждому цвету с помощью гауссового фильтра на исходном и апскейле
            lores_mean = cv2.GaussianBlur(lores_mean,(KERNEL_SIZE,KERNEL_SIZE),0)
            hires_mean = cv2.GaussianBlur(hires_mean,(KERNEL_SIZE,KERNEL_SIZE),0)
            
            if NORM_MODE=='scale':
                # вычисляем нормировочный множитель для каждого пикселя
                coeffs_grid_1x1 = np.where(hires_mean <= 0.0, 0.0, lores_mean/hires_mean)
            
                if GRAY_NORM:
                    for i in range(3):
                        hires[...,i] *= coeffs_grid_1x1
                else:
                    hires *= coeffs_grid_1x1
                    
            elif NORM_MODE=='bias':
                coeffs_grid_1x1 = lores_mean - hires_mean
                if GRAY_NORM:
                    for i in range(3):
                        hires[...,i] += coeffs_grid_1x1
                else:
                    hires += coeffs_grid_1x1
                    
            q_out.put(hires)
            
            
def postprocess(q_in, q_out, iolock, vs):
    with vs as video_stream:
        while True:
            hires = q_in.get()
            if hires is None:
                with iolock:
                    q_out.put(None)
                break
            with iolock:
                hires = video_stream.remove_padding(hires, scale=config.UPSCALE_BY)
                hires = video_stream.postprocess(hires)
                letterbox = video_stream.add_letterbox(hires)
                q_out.put(letterbox)
            
def dump(q_in, iolock, command):
    pipe = sp.Popen(command.split(), stdin=sp.PIPE, stdout=sp.DEVNULL, stderr=sp.DEVNULL)
    while True:
        hires = q_in.get()
        if hires is None:
            print('Got None input for out pipe')
            pipe.wait()
            print('wait out pipe to finish')
            pipe.stdin.close()
            print('out pipe stdin close')
            if pipe.returncode !=0: 
                raise sp.CalledProcessError(pipe.returncode, command)
            #pipe.terminate()
            #print('out pipe terminated')
            break
        with iolock:
            pipe.stdin.write(hires.tobytes())

def upscale(src_vid_path, dst_vid_path):
    # if supported, tensorrt much faster than other providers
    # to get maximum performance use fp16 models with fp16_enable flag 
    if ('TensorrtExecutionProvider') in config.PROVIDERS:
        os.environ["ORT_TENSORRT_FP16_ENABLE"] = "1" 
    
    # init onnxruntime
    ort_session = ort.InferenceSession(config.ONNX_MODEL, providers=config.PROVIDERS)\
    # the order of inputs must correspond to the order of frames, from older to newer
    inputs = sorted(node.name for node in ort_session.get_inputs())
    
    # init video stream 
    vs = VideoStream(src_vid_path)
    # detect letterbox
    with vs as video_stream:
        video_stream.cbuffer.letterbox = video_stream.detect_letterbox()
        print('FPS:', video_stream.fps)
        print('Letterbox:', video_stream.cbuffer.letterbox)
        print('Content:', video_stream.cbuffer.content_shape)

    # build command for ffmpeg to write video
    ff_cmd = FFMPEG(video_stream.up_frame_height, 
                    video_stream.up_frame_width, 
                    video_stream.fps, 
                    config.MBITRATE,
                    dst_vid_path,
                    config.PIX_FMT,
                    config.VCODEC
                   )

    command = ff_cmd.stdout
    print(command)
    
    # parallelize equalizing, post processing and dumping
    q0 = Queue(maxsize=config.QUE_MAXSIZE)
    q1 = Queue(maxsize=config.QUE_MAXSIZE)
    q2 = Queue(maxsize=config.QUE_MAXSIZE)

    iolock = Lock()
    
    if NORM_METHOD=='area':
        proc_equal = Process(target=equalize_mean_area, args=(q0, q1, iolock))
    elif NORM_METHOD=='gauss':
        proc_equal = Process(target=equalize_gauss, args=(q0, q1, iolock))
        
    proc_post = Process(target=postprocess, args=(q1, q2, iolock, vs))
    proc_dump = Process(target=dump, args=(q2, iolock, command))

    proc_equal.start()
    proc_post.start()
    proc_dump.start()
    
    # inference loop
    
    start = time()
    with vs as video_stream:
        print('start polling')
        for frame_id, sequence_of_padded_frames in enumerate(video_stream.stream_of_padded_sequences(loop=False)):
            if frame_id < video_stream.fps * config.START_SEC:
                continue
            elif frame_id > video_stream.fps * (config.START_SEC+config.DUMP_INTERVAL):
                break
            tic = time()
            hires = ort_session.run(None, dict(zip(inputs, sequence_of_padded_frames)))[0]
            print(time()-tic)
            
            tic = time()
            lores = sequence_of_padded_frames[video_stream.idx_of_mid_frame_in_sequence] 
            
            q0.put((hires.copy(), lores.copy()))
            
            print(time()-tic)
            print(f'video time {frame_id/vs.fps:.2f} s, \
                    processing time {time()-start:.3f}, \
                    completed {(frame_id+1)/vs.num_of_frames_in_video*100:.3f}%')
            
        print('end of source video_stream')
        
        print('None to queues')
        q0.put((None, None))
    
    sleep(60)
    # stop all processes
    proc_equal.terminate()
    proc_post.terminate()
    proc_dump.terminate()
    print('processes terminated')
        

def add_sound(src_vid_path, up_vid_path, dst_vid_path):
    ff_cmd = 'ffmpeg -y '
    ff_cmd += f'-i {src_vid_path} '
    ff_cmd += f'-i {up_vid_path} '
    ff_cmd += '-map 1:0 -map 0:1 -c:a copy -c:v copy '
    ff_cmd += f'{dst_vid_path}'
    sp.run(ff_cmd.split())
    
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source', help='cloud folder video path')
    parser.add_argument('-o', '--out_dir', default=None, help='folder to store upscaled video')
    
    args = parser.parse_args()
    
    # load from mts cloud
    print('source', args.source)
    src_vid_path = args.source
    
    # set path to store result
    up_vid_path = src_vid_path.replace('.mp4', f'_v40_4K_{config.MBITRATE}M_nosound.mp4')
    dst_vid_path = src_vid_path.replace('.mp4', f'_superres.mp4')
    
    # upscaling routine
    upscale(src_vid_path, up_vid_path)
    
    # no way for missed frames
    assert num_frames(src_vid_path)==num_frames(up_vid_path), f'different length of {src_vid_path} and  of {up_vid_path}'
        
    # return sound back to upscaled video
    if config.ADD_SOUND:
        add_sound(src_vid_path, up_vid_path, dst_vid_path)
    
    # move and remove
    dst_base_name = os.path.basename(dst_vid_path)
    repo_path = os.path.join(REPO_DIR, dst_base_name)
    shutil.move(dst_vid_path, repo_path)
    print(f'{dst_base_name} stored in {repo_path}')
    
    os.remove(up_vid_path)
    
    print('mission completed')
    

    
