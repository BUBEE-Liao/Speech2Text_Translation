import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
from transformers import AutoFeatureExtractor, Wav2Vec2BertModel
import librosa
import numpy as np
from transformers import MarianMTModel, MarianTokenizer
################################################################################################
# (intermediate_ffn): SeamlessM4Tv2ConformerFeedForward(
#       (intermediate_dropout): Dropout(p=0.0, inplace=False)
#       (intermediate_dense): Linear(in_features=1024, out_features=4096, bias=True)
#       (intermediate_act_fn): ReLU()
#       (output_dense): Linear(in_features=4096, out_features=1024, bias=True)
#       (output_dropout): Dropout(p=0.0, inplace=False)
# )
class SeamlessM4Tv2ConformerFeedForward(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.intermediate_dropout = nn.Dropout(p=dropout, inplace=True)
        self.intermediate_dense = nn.Linear(in_features=1024, out_features=4096, bias=True)
        self.intermediate_act_fn = nn.ReLU()
        self.output_dense = nn.Linear(in_features=4096, out_features=1024, bias=True)
        self.output_dropout = nn.Dropout(p=dropout, inplace=True)
    def forward(self, hidden_states):
        hidden_states = self.intermediate_dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.intermediate_dropout(hidden_states)

        hidden_states = self.output_dense(hidden_states)
        hidden_states = self.output_dropout(hidden_states)
        return hidden_states
################################################################################################
class SeamlessM4Tv2ConformerSelfAttention(nn.Module):
    def __init__(self, use_position_embeddings=True):
        super().__init__()
        self.hidden_size = 1024
        self.speech_encoder_attention_heads = 16
        self.position_embeddings_type = "relative_key"
        self.right_max_position_embeddings = 8
        self.left_max_position_embeddings = 64
        
        
        
        self.head_size = self.hidden_size // self.speech_encoder_attention_heads
        self.num_heads = self.speech_encoder_attention_heads
        self.position_embeddings_type = self.position_embeddings_type if use_position_embeddings else None

        self.linear_q = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_k = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_v = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_out = nn.Linear(self.hidden_size, self.hidden_size)

        self.dropout = nn.Dropout(p=0.0, inplace=False)

        if self.position_embeddings_type == "relative_key":
            self.left_max_position_embeddings = self.left_max_position_embeddings
            self.right_max_position_embeddings = self.right_max_position_embeddings
            num_positions = self.left_max_position_embeddings + self.right_max_position_embeddings + 1
            self.distance_embedding = nn.Embedding(num_positions, self.head_size)
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        # self-attention mechanism
        batch_size, sequence_length, hidden_size = hidden_states.size()

        # make sure query/key states can be != value states
        query_key_states = hidden_states
        value_states = hidden_states

        # project query_key_states and value_states
        query = self.linear_q(query_key_states).view(batch_size, -1, self.num_heads, self.head_size)
        key = self.linear_k(query_key_states).view(batch_size, -1, self.num_heads, self.head_size)
        value = self.linear_v(value_states).view(batch_size, -1, self.num_heads, self.head_size)

        # => (batch, head, time1, d_k)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        attn_weights = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_size)

        if self.position_embeddings_type == "relative_key":
            query_length, key_length = query.shape[2], key.shape[2]

            position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_r - position_ids_l
            distance = torch.clamp(distance, -self.left_max_position_embeddings, self.right_max_position_embeddings)

            positional_embedding = self.distance_embedding(distance + self.left_max_position_embeddings)
            positional_embedding = positional_embedding.to(dtype=query.dtype)  # fp16 compatibility

            relative_position_attn_weights = torch.einsum("bhld,lrd->bhlr", query, positional_embedding)
            attn_weights = attn_weights + (relative_position_attn_weights / math.sqrt(self.head_size))

        # apply attention_mask if necessary
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # => (batch, head, time1, time2)
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # => (batch, head, time1, d_k)
        attn_output = torch.matmul(attn_weights, value)

        # => (batch, time1, hidden_size)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, -1, self.num_heads * self.head_size)
        attn_output = self.linear_out(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights
        
# (adapter): SeamlessM4Tv2ConformerAdapter(
#       (layers): ModuleList(
#         (0): SeamlessM4Tv2ConformerAdapterLayer(
#           (residual_layer_norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
#           (residual_conv): Conv1d(1024, 2048, kernel_size=(8,), stride=(8,), padding=(4,))
#           (activation): GLU(dim=1)
#           (self_attn_layer_norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
#           (self_attn_conv): Conv1d(1024, 2048, kernel_size=(8,), stride=(8,), padding=(4,))
#           (self_attn): SeamlessM4Tv2ConformerSelfAttention(
#             (linear_q): Linear(in_features=1024, out_features=1024, bias=True)
#             (linear_k): Linear(in_features=1024, out_features=1024, bias=True)
#             (linear_v): Linear(in_features=1024, out_features=1024, bias=True)
#             (linear_out): Linear(in_features=1024, out_features=1024, bias=True)
#             (dropout): Dropout(p=0.0, inplace=False)
#           )
#           (self_attn_dropout): Dropout(p=0.1, inplace=False)
#           (ffn_layer_norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
#           (ffn): SeamlessM4Tv2ConformerFeedForward(
#             (intermediate_dropout): Dropout(p=0.1, inplace=False)
#             (intermediate_dense): Linear(in_features=1024, out_features=4096, bias=True)
#             (intermediate_act_fn): ReLU()
#             (output_dense): Linear(in_features=4096, out_features=1024, bias=True)
#             (output_dropout): Dropout(p=0.1, inplace=False)
#           )
#         )
#       )
#     )

class SeamlessM4Tv2ConformerAdapterLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_dim = 1024
        self.kernel_size = 8
        self.stride = 8

        # 1. residual convolution
        self.residual_layer_norm = nn.LayerNorm(self.embed_dim)
        self.residual_conv = nn.Conv1d(
            self.embed_dim,
            2 * self.embed_dim,
            self.kernel_size,
            stride=self.stride,
            padding=self.stride // 2,
        )
        self.activation = nn.GLU(dim=1)

        # 2. Self-Attention
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.self_attn_conv = nn.Conv1d(
            self.embed_dim,
            2 * self.embed_dim,
            self.kernel_size,
            stride=self.stride,
            padding=self.stride // 2,
        )
        self.self_attn = SeamlessM4Tv2ConformerSelfAttention()
        self.self_attn_dropout = nn.Dropout(p=0.1, inplace=False)

        # 3. Feed-forward
        self.ffn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.ffn = SeamlessM4Tv2ConformerFeedForward(dropout=0.1)
        
    def _compute_sub_sample_lengths_from_attention_mask(self, attention_mask):
        pad = self.kernel_size // 2
        seq_lens = attention_mask.size(1) - (1 - attention_mask.int()).sum(1)

        seq_lens = ((seq_lens + 2 * pad - self.kernel_size) / self.stride) + 1

        return seq_lens.floor()

    def forward(
        self,
        hidden_states,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ):
        residual = self.residual_layer_norm(hidden_states)

        # Apply pooling to the residual to match the sequence length of the
        # multi-head attention output.
        # (batch, seq_len, feature_dim) -> (batch, feature_dim, seq_len)
        residual = residual.transpose(1, 2)
        residual = self.residual_conv(residual)
        residual = self.activation(residual)
        # (batch, feature_dim, seq_len) -> (batch, seq_len, feature_dim)
        residual = residual.transpose(1, 2)

        hidden_states = self.self_attn_layer_norm(hidden_states)
        # Apply pooling before feeding to the multihead-attention layer.
        # (batch, seq_len, feature_dim) -> (batch, feature_dim, seq_len)
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.self_attn_conv(hidden_states)
        hidden_states = self.activation(hidden_states)
        # (batch, feature_dim, seq_len) -> (batch, seq_len, feature_dim)
        hidden_states = hidden_states.transpose(1, 2)

        if attention_mask is not None:
            sub_sampled_lengths = self._compute_sub_sample_lengths_from_attention_mask(attention_mask).to(
                hidden_states.device
            )
            attention_mask = _compute_new_attention_mask(hidden_states=hidden_states, seq_lens=sub_sampled_lengths)
            attention_mask = _prepare_4d_attention_mask(
                attention_mask,
                hidden_states.dtype,
            )

        # The rest of the computation is identical to a vanilla Transformer
        # encoder layer.
        hidden_states, attn_weigths = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = self.self_attn_dropout(hidden_states)
        hidden_states = hidden_states + residual

        residual = hidden_states

        hidden_states = self.ffn_layer_norm(hidden_states)
        hidden_states = self.ffn(hidden_states) + residual

        return hidden_states
    
class SeamlessM4Tv2ConformerAdapter(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.ModuleList(
            [SeamlessM4Tv2ConformerAdapterLayer()]
        )

    def forward(self, hidden_states, attention_mask):
        # down project hidden_states if necessary

        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)

        return hidden_states
        
################################################################################################




class SeamlessM4Tv2SpeechEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # self.feature_projection = AutoFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0") # from huggingface
        self.encoder = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0") # from huggingface
        self.intermediate_ffn = SeamlessM4Tv2ConformerFeedForward(dropout=0.0)
        self.adapter = SeamlessM4Tv2ConformerAdapter()
        self.inner_layer_norm = nn.LayerNorm(1024)
        self.final_projection = nn.Linear(1024, 512)
    def forward(
        self,
        inputs,
        attention_mask,
        # output_attentions : bool = None,
        # output_hidden_states: bool = None,
        # return_dict: bool = None,
        # **kwargs,
    ):
        hidden_states = self.encoder(input_features=inputs, attention_mask=attention_mask)

        expanded_hidden_states = self.intermediate_ffn(hidden_states.last_hidden_state)
        hidden_states = hidden_states.last_hidden_state + 0.5 * expanded_hidden_states

        hidden_states = self.inner_layer_norm(hidden_states)
        output = self.final_projection(hidden_states)
        return output

################################################################################################
def get_text_encoder():
    model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
    return model.base_model.encoder
def get_text_decoder():
    model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
    return model.base_model.decoder
def get_embedding():
    model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
    embeddings = model.get_input_embeddings()
    return embeddings.weight
def get_bias():
    model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
    return model.final_logits_bias

def get_lm_head():
    model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
    linear = torch.nn.Linear(512, 65001)
    embeddings = model.get_input_embeddings()
    linear.weight = torch.nn.Parameter(embeddings.weight)
    linear.bias = torch.nn.Parameter(model.final_logits_bias)
    return linear
    
class SpeechToTextTranslationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_encoder = get_text_encoder()
        self.text_decoder = get_text_decoder()
        self.lm_head = get_lm_head()

        self.speech_encoder = SeamlessM4Tv2SpeechEncoder()


    def forward_text(self, text_input_ids, attention_mask=None):
        return self.text_encoder(input_ids=text_input_ids, attention_mask=attention_mask).last_hidden_state

    def forward_speech(self, speech_input, attention_mask=None):
        return self.speech_encoder(inputs=speech_input, attention_mask=attention_mask)

    def inference(self, encoder_hidden_state, tokenizer, device):
        seq_len = 1
        eos_ids = tokenizer(tokenizer.eos_token, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        decoder_input_ids = tokenizer("<pad>", add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        
        while(seq_len<512):
            decoder_output_vectors = self.text_decoder(decoder_input_ids, encoder_hidden_states=encoder_hidden_state).last_hidden_state
            lm_logits = self.lm_head(decoder_output_vectors)
            next_decoder_input_ids = torch.argmax(lm_logits[:, -1:], axis=-1)
            decoder_input_ids = torch.cat([decoder_input_ids, next_decoder_input_ids], axis=-1)
            if(next_decoder_input_ids==eos_ids):
                break
            else:
                seq_len += 1
            
        return tokenizer.batch_decode(decoder_input_ids, skip_special_tokens=True), decoder_input_ids
        

    
        
    