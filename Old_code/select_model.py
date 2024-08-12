"""
Select the models defined in config.py.
"""

from segmenter import vit_model
# from hd_segformer import hd_vit_model
import config

import segmentation_models_pytorch as smp

model_version = config.model_version
ENCODER = config.ENCODER
ENCODER_WEIGHTS = config.ENCODER_WEIGHTS
ACTIVATION = config.ACTIVATION
use_decoder = config.use_decoder
DECODER = config.DECODER


# if ENCODER == "hd_model":
#     MODEL = hd_vit_model

if ENCODER == "segmenter":
    MODEL = vit_model

if use_decoder == True:
   if DECODER == "FPN":
        MODEL = smp.FPN(
            encoder_name=ENCODER, 
            encoder_weights=ENCODER_WEIGHTS, 
            classes=len(config.wound_classes), 
            activation=ACTIVATION
        )
   if DECODER == "UNET":
        MODEL = smp.UNET(
            encoder_name=ENCODER, 
            encoder_weights=ENCODER_WEIGHTS, 
            classes=len(config.wound_classes), 
            activation=ACTIVATION,
        )
      
else:
      print("Encoder ", ENCODER, " or decoder ", DECODER, "not recognized or use_decoder=False. Use_decoder: ", use_decoder)
