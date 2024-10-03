# import librosa
# import torch
# from transformers import AutoProcessor, HubertModel
import pandas as pd


def get_features(path_to_file):
    # Due to limited local processing capacity, embeddings are generated via Google Colab and only loaded here.
    # Recreating features for NIFM models is only included for completeness of code and should not be used anytime.
    # The code used to generate the embeddings is included for reproducibility:

    # processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft")
    # model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")
    #
    # sr = 16000
    # x, _ = librosa.core.load(path_to_file, sr=sr)
    # input_values = processor(x, return_tensors="pt", sampling_rate=sr).input_values
    # with torch.no_grad():
    #     hidden_states = model(input_values).last_hidden_state
    # return hidden_states.detach().numpy()
    raise Exception("NIFM features cannot be recreated locally due to limited processing capacity. \n "
                    "Please set 'recreate_features' to False.")

