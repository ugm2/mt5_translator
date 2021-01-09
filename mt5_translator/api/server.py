from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List
from mt5_translator.core.mt5_translator import MT5_Translator

app = FastAPI()
translator = MT5_Translator()

first_example = \
"¿Podemos controlar nuestros sueños?"
second_example = \
"Can we control our dreams?"

class Payload(BaseModel):
    sentences: List[str] = Field([first_example, 
                                  second_example],
                                 title="Input sentences")
    max_length: int = Field(None,
                            title="Maximum translation length")
    min_length: int = Field(None,
                            title="Minimum translation length")
    num_beams: int = Field(4,
                           title="Number of exploratory beams")

class Translations(BaseModel):
    translations: List[str] = Field(None, title="Translations")

async def translate_async(sentences, max_length, min_length, num_beams):
    return translator.translate(sentences, max_length, min_length, num_beams)

@app.post("/translate", response_model=Translations, status_code=200, name="translate")
async def translate(payload: Payload):
    translations = await translate_async(payload.sentences,
                                         payload.max_length,
                                         payload.min_length,
                                         payload.num_beams)
    translations = Translations(translations=translations)
    return translations