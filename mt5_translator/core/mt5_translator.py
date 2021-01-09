from simpletransformers.t5 import T5Model, T5Args
import torch
import logging
from typing import List
from mt5_translator import config

logging.basicConfig(level=config.LOGGING_LEVEL)

class MT5_Translator:

    def __init__(self,
                 model_path: str = config.DEFAULT_MODEL_PATH,
                 model_architecture: str = config.MODEL_ARCHITECTURE,
                 use_cuda: bool = config.GPU):
        '''
        Constructs all the necessary attributes for the MT5_Translator object.
        Parameters
        ----------
            model_path : str
                path to the mt5_translator model
            model_architecture : str
                model architecture (mt5, t5 ...)
            use_cuda : bool
                whether to use CUDA or not (if available)
        '''
        logging.info("Loading model...")
        self.model_path = model_path
        self.use_cuda = use_cuda
        self.device = True \
            if torch.cuda.is_available() and self.use_cuda else False
        self.model_args = T5Args()
        self.model_args.max_length = 512
        self.model_args.length_penalty = 1
        self.model_args.num_beams = 10
        self.model = T5Model("mt5", self.model_path, args=self.model_args, use_cuda=self.device)
        logging.info(f"Use CUDA: {self.device}")
        logging.info(f"Num GPUs Available: {torch.cuda.device_count()}")
        logging.info(f"Model loaded")

    def translate(self, 
                  sentences: List[str],
                  max_length: int = 512,
                  min_length: int = None,
                  num_beams: int = 4):
        '''
        Generate translations from sentences

        Parameters
        ----------
            sentences : list of str
                list of sentences. Each sentence gets its own translation
        '''
        self.model_args = T5Args()
        self.model_args.max_length = max_length if max_length else max_length
        self.model_args.min_length = min_length if min_length else min_length
        self.model_args.num_beams = num_beams if num_beams else num_beams
        self.model.args = self.model_args
        return self.model.predict(sentences)