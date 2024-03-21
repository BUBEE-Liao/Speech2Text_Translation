import torch
from data_loader import *
from model import *
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from transformers import AutoFeatureExtractor
from transformers import MarianMTModel, MarianTokenizer
from torch.cuda.amp import GradScaler, autocast

def main():
    # model
    model = SpeechToTextTranslationModel()

    # optimizer
    learning_rate=1e-8
    weight_decay=1e-6
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=weight_decay)

    # data lodaer
    batch_size=1
    train_dataset = TextAudioLoader('TrainList.txt')
    collate_fn = TextAudioCollate()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, pin_memory=True, num_workers=4)

    # start training
    train(model, optimizer, train_loader, learning_rate)


def train(model, optimizer, train_loader, learning_rate):
    scaler = GradScaler()
    learning_rate = learning_rate
    tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
    # processor = AutoFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
    device = torch.device('cuda')
    # device = torch.device("cuda:1,3" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(log_dir='./logs')
    s2t_model = model
    # s2t_model.text_encoder.eval()
    # s2t_model.text_decoder.train()
    # s2t_model.speech_encoder.train()
    # s2t_model.lm_head.train()
    s2t_model.train()
    for param in s2t_model.text_encoder.parameters():
        param.requires_grad = False 
    s2t_model = s2t_model.to(device)

    ### define loss ###
    logSoftmax = nn.LogSoftmax(dim=-1)
    nllloss = nn.NLLLoss(reduction='mean', ignore_index=65000)
    softmax = torch.nn.Softmax(dim=-1)
    kl = torch.nn.KLDivLoss(reduction='mean')

    iteration = 1
    accumulation_steps = 4
    epochs = 1000000 #where to stop training
    print_iteration = 1000
    eval_iteration = 1000
    store_iteration = 100000

    for epoch in range(0, epochs):
        print('------epoch{}------'.format(epoch))
        for batch_idx, (audio_feature_padded, audio_feature_mask, src_text_padded, src_text_len, src_text_mask, decoder_input_ids_padded, decoder_input_ids_len, tgt_text_mask, target_ids_padded, target_ids_len) in enumerate(train_loader):
            #print(audio_feature_padded.shape)


            audio_feature_padded = audio_feature_padded.to(device)
            audio_feature_mask = audio_feature_mask.to(device)

            src_text_padded = src_text_padded.to(device)
            src_text_mask = src_text_mask.to(device)
            
            decoder_input_ids_padded = decoder_input_ids_padded.to(device)
            tgt_text_mask = tgt_text_mask.to(device)
            target_ids_padded = target_ids_padded.to(device)

            with autocast():
                text_encoder_output = s2t_model.forward_text(src_text_padded, src_text_mask)
                text_encoder_output = text_encoder_output
                speech_encoder_output, attention_mask_adapter = s2t_model.forward_speech(audio_feature_padded, audio_feature_mask)
                speech_encoder_output = speech_encoder_output
                attention_mask_adapter = attention_mask_adapter.to(device)
                # speech_encoder_output = speech_encoder_output.to(device)
    
                text_decoder_output = s2t_model.text_decoder(input_ids=decoder_input_ids_padded, attention_mask=tgt_text_mask, encoder_hidden_states=text_encoder_output, encoder_attention_mask=src_text_mask).last_hidden_state
    
                speech_decoder_output = s2t_model.text_decoder(input_ids=decoder_input_ids_padded, attention_mask=tgt_text_mask, encoder_hidden_states=speech_encoder_output, encoder_attention_mask=attention_mask_adapter).last_hidden_state
    
                text_logits = s2t_model.lm_head(text_decoder_output)
                speech_logits = s2t_model.lm_head(speech_decoder_output)
                
                # calculate loss
                text_logits_logSoftmax = logSoftmax(text_logits) 
                speech_logits_logSoftmax = logSoftmax(speech_logits)
    
                text_logits_logSoftmax = text_logits_logSoftmax.permute(0, 2, 1)
                speech_logits_logSoftmax = speech_logits_logSoftmax.permute(0, 2, 1)
                target_ids_long_tensor = target_ids_padded.type(torch.LongTensor)
                target_ids_long_tensor = target_ids_long_tensor.to(device)
                
    
                loss_text = nllloss(text_logits_logSoftmax, target_ids_long_tensor)
                loss_speech = nllloss(speech_logits_logSoftmax, target_ids_long_tensor)
    
                text_logits_softmax = softmax(text_logits)
                speech_logits_softmax = softmax(speech_logits)
                loss_kl = kl(speech_logits_softmax, text_logits_softmax)

                total_loss = loss_text + loss_speech + loss_kl
                total_loss = total_loss/accumulation_steps
                # backward
                scaler.scale(total_loss).backward()

            if(iteration%print_iteration==0):
                print('iteration{}: '.format(iteration), 'text_nllloss_loss:', loss_text.item(), ', speech_nllloss_loss:', loss_speech.item(), 'kl_loss:', loss_kl.item())


            if(iteration%eval_iteration==0):
                evaluate(s2t_model, audio_feature_padded[0,:,:], audio_feature_mask[0, :], target_ids_padded[0, :], tokenizer, iteration, writer, device)
            
                
            if(iteration%accumulation_steps==0):
                scaler.step(optimizer)
                scaler.update()
    
                # zero grad
                optimizer.zero_grad()
    
    
            if(iteration%store_iteration==0):
                ckpt_path = os.path.join('/home/bubee/seamless/s2t/3000_ckpt/', "iteration_{}.pth".format(iteration))
                state_dict = s2t_model.state_dict()
                torch.save({'model': state_dict,'iteration': iteration,'optimizer': optimizer.state_dict(),'learning_rate': learning_rate}, ckpt_path)
    
            ### release memory
            del text_encoder_output
            del speech_encoder_output
            del text_decoder_output
            del speech_decoder_output
            del text_logits
            del speech_logits
            del text_logits_logSoftmax
            del speech_logits_logSoftmax
            del target_ids_long_tensor
            del text_logits_softmax
            del speech_logits_softmax
            del attention_mask_adapter
                
            del loss_text
            del loss_speech
            del loss_kl
            del total_loss
                
                
    # remember to add del
def evaluate(s2t_model, audio_feature_padded, audio_feature_mask, target_ids_padded, tokenizer, iteration, writer, device):
    s2t_model.eval()
    # audio = audio.squeeze()

    audio_feature_padded = audio_feature_padded.unsqueeze(dim=0)
    audio_feature_mask = audio_feature_mask.unsqueeze(dim=0)
    speech_encoder_output, attention_mask_adapter = s2t_model.forward_speech(speech_input=audio_feature_padded, attention_mask=audio_feature_mask)
    res, ids = s2t_model.inference(encoder_hidden_state=speech_encoder_output, attention_mask_adapter=attention_mask_adapter, tokenizer=tokenizer, device=device)
    tgt_text = tokenizer.decode(target_ids_padded, skip_special_tokens=True)

    # writer.add_audio('input_audio_{}'.format(iteration), audio, global_step=iteration, sample_rate=16000)
    writer.add_text('model_output_{}'.format(iteration), res[0], global_step=iteration)
    writer.add_text('ground_truth_{}'.format(iteration), tgt_text, global_step=iteration)

    del speech_encoder_output, attention_mask_adapter, res, ids
    s2t_model.train()
    for param in s2t_model.text_encoder.parameters():
        param.requires_grad = False 
            
if __name__ == '__main__':
    main()
