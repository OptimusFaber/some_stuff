# reader
UPSCALE_BY = 2
NUM_OF_MODEL_INPUTS = 5
MODEL_POOLING_DEPTH = 5
NORMALIZER = 256.0
PROCESSING_FP_DTYPE = 'float16'
FULL_HD_FRAME_SHAPE = (1080, 1920, 3)
NUM_OF_PROBES = 100

BIAS = 16 * 255 / 219 # set to range [16/256, 235/256] instead of [0, 1] before inference
SCALE = 256 * 255 / 219 # 16, 235 and 256 are from TOPAZ dll
#COLOR_CORR = ((0.034491650760, 0.961110234261),
#              (0.033822555095, 0.964093148708),
#              (0.035013835877, 0.951422631741)
#             )
COLOR_CORR = ((0, 1),
              (0, 1),
              (0, 1)
             )


#COLOR_CORR = (255/250, 255/248, 255/251) # color correction per channel
#COLOR_CORR = (255/251, 255/250, 255/252) # color correction per channel

# onnxruntime
PROVIDERS = ['TensorrtExecutionProvider']
#PROVIDERS = ['CUDAExecutionProvider']

ONNX_MODEL = '/ffmpeg/superres/some_stuff/ghq-v6-gaia-fp16-anysize-2x-batch.onnx'

# ffmpeg 
MBITRATE = 50
VCODEC = 'h264_nvenc'
#VCODEC = 'hevc_nvenc'
#VCODEC = 'libx264'
PIX_FMT = 'rgb24'
#PIX_FMT = 'yuv444p'
#PIX_FMT = 'yuv420p'

# upscale
NORM_METHOD = 'area' # 'area', 'gauss'
NORM_MODE = 'bias' # 'scale', 'bias'
NORM_AREA_SIZE = 2 # int value
GRAY_NORM = False
START_SEC = 0 # start of video in seconds
DUMP_INTERVAL = 60*60*4 #video interval in seconds
QUE_MAXSIZE = 16
ADD_SOUND = True #True

REPO_DIR = '/superres' 
