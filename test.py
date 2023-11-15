from collections import deque

import numpy as np
import torch
from pyboy import PyBoy
from transformers import (
    pipeline,
    AutoImageProcessor,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    VisionEncoderDecoderModel,
)

FPS = 48
VIDEO_LEN = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    # load pretrained processor, tokenizer, and model
    image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = VisionEncoderDecoderModel.from_pretrained(
        "Neleac/timesformer-gpt2-video-captioning"
    ).to(DEVICE)

    qa_model_name = "deepset/roberta-base-squad2"
    nlp = pipeline("question-answering", model=qa_model_name, tokenizer=qa_model_name)

    with PyBoy("pokemonred.gb") as pyboy:
        screen = pyboy.botsupport_manager().screen()
        i = 0
        frames = []
        while not pyboy.tick():
            if i % FPS == 0:
                frame = screen.screen_ndarray()
                frames.append(
                    np.pad(
                        frame,
                        (
                            ((480 - frame.shape[0]) // 2, (640 - frame.shape[1]) // 2),
                            ((480 - frame.shape[0]) // 2, (640 - frame.shape[1]) // 2),
                            (0, 0),
                        ),
                    )
                )
                if len(frames) == VIDEO_LEN:
                    # generate caption
                    gen_kwargs = {
                        "min_length": 10,
                        "max_length": 120,
                        "num_beams": 16,
                    }
                    pixel_values = image_processor(frames, return_tensors="pt").pixel_values.to(
                        DEVICE
                    )
                    tokens = model.generate(pixel_values, **gen_kwargs)
                    caption = tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]
                    print(
                        f"caption: {caption}"
                    )

                    instruction = nlp(
                        {
                            "question": "What should I do next among the options: A, B, UP, DOWN, LEFT, RIGHT, START?",
                            "context": f"I am currently playing a video game on a GameBoy. "
                            "A GameBoy has options A, B, UP, DOWN, LEFT, RIGHT, START. "
                            f"Most recently, I {caption}.",
                        }
                    )

                    print(f"instruction: {instruction}")

                    # Now remove half the frames
                    frames = frames[VIDEO_LEN // 2 :]

            i = i + 1
