import torch
import os
import librosa
import numpy as np
from torch.utils.data import DataLoader
from transformers import MarianMTModel, MarianTokenizer
from transformers import AutoFeatureExtractor

class TextAudioLoader(torch.utils.data.Dataset):
    def __init__(self):
        super(TextAudioLoader, self).__init__()
        self.train_txt_file_path = '/home/bubee/data/nas05/bubee/translation_data/train_list_clean.txt'
        self.train_list = self.get_train_list(self.train_txt_file_path)
        self.tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
    def get_train_list(self, train_txt_file_path):
        train_list = []
        with open(train_txt_file_path, 'r')as f:
            for line in f.readlines():
                line = line.replace('\n', '')
                if(line != ''):
                    data = line.split('|')
                    train_list.append(data)
        return train_list

    def get_text_audio_target(self, data):
        # audio
        audio_path = data[0]
        sample_rate = None
        if(audio_path[-4:]=='.wav'):
            sample_rate = 44100
        else:
            sample_rate = 48000
            
        audio, sr = librosa.load(audio_path, sr=sample_rate)
        audio = librosa.resample(y=audio, orig_sr=sample_rate, target_sr=16000)
        audio = torch.FloatTensor(audio.astype(np.float32))
        
        # audio = audio.unsqueeze(dim=0)
        #text
        src_txt = data[1]
        tgt_txt = data[2]

        src_txt = self.tokenizer(src_txt, add_special_tokens=False, return_tensors="pt").input_ids
        tgt_txt = self.tokenizer(tgt_txt, add_special_tokens=False, return_tensors="pt").input_ids

        src_txt = src_txt.squeeze()
        try:
            a = len(src_txt)
        except:
            src_txt = src_txt.unsqueeze(dim=0)
        # tgt_txt = tgt_txt.squeeze(dim=0)

        bos_token = self.tokenizer(self.tokenizer.pad_token, add_special_tokens=False, return_tensors="pt").input_ids
        eos_token = self.tokenizer(self.tokenizer.eos_token, add_special_tokens=False, return_tensors="pt").input_ids
        
        decoder_input_ids = torch.cat((bos_token, tgt_txt), -1)
        decoder_input_ids = decoder_input_ids.squeeze()

        target_ids = torch.cat((tgt_txt, eos_token), -1)
        target_ids = target_ids.squeeze()
        
        return audio, src_txt, decoder_input_ids, target_ids
        
    
    def __getitem__(self, index):
        return self.get_text_audio_target(self.train_list[index])

    def __len__(self):
        return len(self.train_list)

class TextAudioCollate():

    def __init__(self, return_ids=False):
        self.return_ids = return_ids
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")

    def __call__(self, batch):
        # for x in batch:
        #     print(x.shape)
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        # print('ids_sorted_decreasing:',ids_sorted_decreasing)

        

        max_audio_len = max([len(x[0]) for x in batch])
        max_src_text_len = max([len(x[1]) for x in batch])
        max_decoder_input_ids_len = max([len(x[2]) for x in batch])
        max_target_ids_len = max([len(x[3]) for x in batch])

        

        audio_len = torch.IntTensor(len(batch))
        src_text_len = torch.IntTensor(len(batch))
        decoder_input_ids_len = torch.IntTensor(len(batch))
        target_ids_len = torch.IntTensor(len(batch))
        

        audio_padded = torch.FloatTensor(len(batch), max_audio_len)
        src_text_padded = torch.IntTensor(len(batch), max_src_text_len)
        decoder_input_ids_padded = torch.IntTensor(len(batch), max_decoder_input_ids_len)
        target_ids_padded = torch.IntTensor(len(batch), max_target_ids_len)
        
        src_text_mask = torch.IntTensor(len(batch), max_src_text_len)
        tgt_text_mask = torch.IntTensor(len(batch), max_decoder_input_ids_len)
        
        
        audio_padded.zero_()
        src_text_padded = src_text_padded.fill_(65000)
        decoder_input_ids_padded = decoder_input_ids_padded.fill_(65000)
        target_ids_padded = target_ids_padded.fill_(65000)
        src_text_mask.zero_()
        tgt_text_mask.zero_()

        audio_array = [None]*len(ids_sorted_decreasing)
        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            audio = row[0]
            audio_array[i] = audio
            
            audio_padded[i, : audio.size(0)] = audio
            # audio_len[i] = audio.size(0)

            # audio_ones = torch.IntTensor(audio.size(0))
            # audio_ones = audio_ones.fill_(1)
            # audio_mask[i, : audio.size(0)] = audio_ones

            src_text = row[1]
            src_text_padded[i, :src_text.size(0)] = src_text
            src_text_len[i] = src_text.size(0)

            # text encoder input mask
            src_text_ones = torch.IntTensor(src_text.size(0))
            src_text_ones = src_text_ones.fill_(1)
            src_text_mask[i, : src_text.size(0)] = src_text_ones


            decoder_input_ids = row[2]
            decoder_input_ids_padded[i, :decoder_input_ids.size(0)] = decoder_input_ids
            decoder_input_ids_len[i] = decoder_input_ids.size(0)

            # decoder input mask
            decoder_input_ids_ones = torch.IntTensor(decoder_input_ids.size(0))
            decoder_input_ids_ones = decoder_input_ids_ones.fill_(1)
            tgt_text_mask[i, : decoder_input_ids.size(0)] = decoder_input_ids_ones

            target_ids = row[3]
            target_ids_padded[i, :target_ids.size(0)] = target_ids
            target_ids_len[i] = target_ids.size(0)


        audio_batch = [x.numpy() for x in audio_array]
        audio_feature = self.feature_extractor(audio_batch, sampling_rate=16000, return_tensors="pt", return_attention_mask=True, padding='longest')
        audio_feature_padded = audio_feature['input_features']
        audio_feature_mask = audio_feature['attention_mask']

        if self.return_ids: #text_padded, text_lengths,
            return audio_padded, audio_feature_padded, audio_feature_mask, src_text_padded, src_text_len, src_text_mask, decoder_input_ids_padded, decoder_input_ids_len, tgt_text_mask, target_ids_padded, target_ids_len, ids_sorted_decreasing
        return audio_padded, audio_feature_padded, audio_feature_mask, src_text_padded, src_text_len, src_text_mask, decoder_input_ids_padded, decoder_input_ids_len, tgt_text_mask, target_ids_padded, target_ids_len