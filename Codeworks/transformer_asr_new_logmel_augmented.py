# -*- coding: utf-8 -*-
"""transformer_asr-keras_sanskrit-working
 recognition.
"""





import os
import librosa    

import random
from glob import glob
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

import matplotlib.pyplot as plt
from tokenizers import Tokenizer
# from tokenizers.models import BPE
import soundfile as sf
from datetime import datetime
import sys
from contextlib import redirect_stdout
# from werpy import wer
from jiwer import wer
# -------------------------------*************************--------------------------------
# -------------------------------change here the parameters ------------------------------
# -------------------------------*************************--------------------------------
#choose train mode or inference mode/demo mode
if 10:
    train_enable = True
else:
    train_enable = False    

native_scheduler = True  #one given in the Vaswani's paper else default one given with the code

target_start_token_idx=7 
target_end_token_idx=8

batch_size = 64
max_target_len = 50  # all transcripts after tokenization should have < 50 tokens 
# tokenizer_file = "tokenizer-sanskrit-2.5k.json"
tokenizer_file = "tokenizer-sanskrit-450.json"



#Spectrogram hyperparameters
# srate = 22050 # sampling rate
srate = 16000 # sampling rate
dur = 10 # 10 sec audio sequences assumed
frame_length = 512
frame_step = 256
fft_length = 512


#Transformer hyperparameters
num_hid=128
num_head=8
num_feed_forward=512
source_maxlen=100
target_maxlen=max_target_len
num_layers_enc=6
num_layers_dec=6
drop_out_enc = 0.3
drop_out_dec = 0.3
drop_out_cross = 0.3
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++




"""## Define the Transformer Input Layer

When processing past target tokens for the decoder, we compute the sum of
position embeddings and token embeddings.

When processing audio features, we apply convolutional layers to downsample
them (via convolution strides) and process local relationships.
"""

class EarlyStoppingAtMinLoss(keras.callbacks.Callback):
    """Stop training when the loss is at its min, i.e. the loss stops decreasing.

    Arguments:
        patience: Number of epochs to wait after min has been hit. After this
        number of no improvement, training stops.
    """

    def __init__(self, patience=0):
        super().__init__()
        self.patience = patience
        # best_weights to store the weights at which the minimum loss occurs.
        self.best_weights = None
        

    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        
        if epoch % 5 != 0 or epoch == 0:
            return
        
        # Evaulating the model on the test set and logging the loss in results.txt
                
        curr_wer = compute_WER(val_ds)
        recon_str1 = ' WER on test set after epoch -' + \
          str(epoch) + ' is ' + str("{:.3f}".format(curr_wer)) + ' : \n'
        # loss = model.evaluate(val_ds, verbose=2)
        # recon_str = '; validation loss after epoch -' + \
        #   str(epoch) + ' is ' + str("{:.3f}".format(loss)) + '; '
        with open('results/results.txt', 'a') as file:
          # file.write(recon_str)    
          file.write(recon_str1)  
        
        if np.less(curr_wer, self.best):
            self.best = curr_wer
            self.wait = 0
            # Record the best weights if current results is better (less).
            # self.best_weights = self.model.get_weights()
            self.model.save_weights("MyModel_tf.weights.h5")

        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print("Restoring model weights from the end of the best epoch.")
                # self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print(f"Epoch {self.stopped_epoch + 1}: early stopping")



# def positional_encoding(length, depth):
#   depth = depth/2

#   positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
#   depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)

#   angle_rates = 1 / (10000**depths)         # (1, depth)
#   angle_rads = positions * angle_rates      # (pos, depth)

#   pos_encoding = np.concatenate(
#   [np.sin(angle_rads), np.cos(angle_rads)],
#   axis=-1) 

#   return tf.cast(pos_encoding, dtype=tf.float32)


class TokenEmbedding(layers.Layer):
    def __init__(self, num_vocab, maxlen, d_model):
        super().__init__()
        self.emb = tf.keras.layers.Embedding(num_vocab, d_model)
        self.pos_emb = layers.Embedding(input_dim=maxlen+1, output_dim=d_model)
# TODO: check the maxlen+1 logic in above line
    def call(self, x):
        maxlen = tf.shape(x)[-1]
        x = self.emb(x)
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        return x + positions

# class TokenEmbedding(tf.keras.layers.Layer):
#   def __init__(self, num_vocab, maxlen, d_model):
#     super().__init__()
#     self.d_model = d_model
#     self.embedding = tf.keras.layers.Embedding(num_vocab, d_model, mask_zero=True) 
#     self.pos_encoding = positional_encoding(length=maxlen, depth=d_model)

#   def compute_mask(self, *args, **kwargs):
#     return self.embedding.compute_mask(*args, **kwargs)


#   def call(self, x):
#       length = tf.shape(x)[1]
#       x = self.embedding(x)
#       # This factor sets the relative scale of the embedding and positonal_encoding.
#       x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
#       x = x + self.pos_encoding[tf.newaxis, :length, :]
#       return x

class SpeechFeatureEmbedding(layers.Layer):
    def __init__(self, num_hid, maxlen):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv1D(
            num_hid, 11, strides=2, padding="same", data_format='channels_last', activation="gelu"
        )
        self.conv2 = tf.keras.layers.Conv1D(
            num_hid, 11, strides=2, padding="same", data_format='channels_last', activation="gelu"
        )
        self.conv3 = tf.keras.layers.Conv1D(
             num_hid, 11, strides=2, padding="same", activation="gelu"
         )

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        
        # return x
        return self.conv3(x)

"""## Transformer Encoder Layer"""

class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, num_heads, feed_forward_dim, drop_out_enc):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [
                layers.Dense(feed_forward_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(drop_out_enc)
        self.dropout2 = layers.Dropout(drop_out_enc)

    def call(self, inputs):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)

"""## Transformer Decoder Layer"""

class TransformerDecoder(layers.Layer):
    def __init__(self, embed_dim, num_heads, feed_forward_dim, drop_out_dec):
        super().__init__()
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = layers.LayerNormalization(epsilon=1e-6)
        self.self_att = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.enc_att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.self_dropout = layers.Dropout(drop_out_dec)
        self.enc_dropout = layers.Dropout(drop_out_cross)
        self.ffn_dropout = layers.Dropout(drop_out_dec)
        self.ffn = keras.Sequential(
            [
                layers.Dense(feed_forward_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )

    def causal_attention_mask(self, batch_size, n_dest, n_src, dtype):
        """Masks the upper half of the dot product matrix in self attention.

        This prevents flow of information from future tokens to current token.
        1's in the lower triangle, counting from the lower right corner.
        """
        i = tf.range(n_dest)[:, None]
        j = tf.range(n_src)
        m = i >= j - n_src + n_dest
        mask = tf.cast(m, dtype)
        mask = tf.reshape(mask, [1, n_dest, n_src])
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)], 0
        )
        return tf.tile(mask, mult)

    def call(self, enc_out, target):
        input_shape = tf.shape(target)
        batch_size = input_shape[0]
        seq_len = input_shape[1]
        causal_mask = self.causal_attention_mask(batch_size, seq_len, seq_len, tf.bool)
        target_att = self.self_att(target, target, attention_mask=causal_mask)
        target_norm = self.layernorm1(target + self.self_dropout(target_att))
        enc_out = self.enc_att(target_norm, enc_out)
        enc_out_norm = self.layernorm2(self.enc_dropout(enc_out) + target_norm)
        ffn_out = self.ffn(enc_out_norm)
        ffn_out_norm = self.layernorm3(enc_out_norm + self.ffn_dropout(ffn_out))
        return ffn_out_norm

"""## Complete the Transformer model

Our model takes audio spectrograms as inputs and predicts a sequence of characters.
During training, we give the decoder the target character sequence shifted to the left
as input. During inference, the decoder uses its own past predictions to predict the
next token.
"""

class Transformer(keras.Model):
    def __init__(
        self,
        num_hid,
        num_head,
        num_feed_forward,
        source_maxlen, # apparently not used
        target_maxlen,
        num_layers_enc,
        num_layers_dec,
        num_classes,
    ):
        super().__init__()
        self.loss_metric = keras.metrics.Mean(name="loss")
        self.num_layers_enc = num_layers_enc
        self.num_layers_dec = num_layers_dec
        self.target_maxlen = target_maxlen
        self.num_classes = num_classes

        self.enc_input = SpeechFeatureEmbedding(num_hid=num_hid, maxlen=source_maxlen)
        self.dec_input = TokenEmbedding(
            num_vocab=num_classes+1, maxlen=target_maxlen, d_model=num_hid
        )

        self.encoder = keras.Sequential(
            [self.enc_input]
            + [
                TransformerEncoder(num_hid, num_head, num_feed_forward,drop_out_enc)
                for _ in range(num_layers_enc)
            ]
        )
        
        

        for i in range(num_layers_dec):
            setattr(
                self,
                f"dec_layer_{i}",
                TransformerDecoder(num_hid, num_head, num_feed_forward,drop_out_dec),
            )

        self.classifier = layers.Dense(num_classes)

    def decode(self, enc_out, target):
        y = self.dec_input(target)
        for i in range(self.num_layers_dec):
            y = getattr(self, f"dec_layer_{i}")(enc_out, y)
        return y

    def call(self, inputs):
        source = inputs[0]
        target = inputs[1]
        x = self.encoder(source)
        y = self.decode(x, target)
        return self.classifier(y)

    @property
    def metrics(self):
        return [self.loss_metric]

    def train_step(self, batch):
        """Processes one batch inside model.fit()."""
        source = batch["source"]
        target = batch["target"]
        dec_input = target[:, :-1]
        dec_target = target[:, 1:]
        with tf.GradientTape() as tape:
            preds = self([source, dec_input])
            one_hot = tf.one_hot(dec_target, depth=self.num_classes)
            mask = tf.math.logical_not(tf.math.equal(dec_target, 0))
            loss = self.compiled_loss(one_hot, preds, sample_weight=mask)
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.loss_metric.update_state(loss)
        return {"loss": self.loss_metric.result()}

    def test_step(self, batch):
        source = batch["source"]
        target = batch["target"]
        dec_input = target[:, :-1]
        dec_target = target[:, 1:]
        preds = self([source, dec_input])
        one_hot = tf.one_hot(dec_target, depth=self.num_classes)
        mask = tf.math.logical_not(tf.math.equal(dec_target, 0))
        loss = self.compiled_loss(one_hot, preds, sample_weight=mask)
        self.loss_metric.update_state(loss)
        return {"loss": self.loss_metric.result()}

    def generate(self, source, target_start_token_idx):
        """Performs inference over one batch of inputs using greedy decoding."""
        bs = tf.shape(source)[0]
        enc = self.encoder(source)
        dec_input = tf.ones((bs, 1), dtype=tf.int32) * target_start_token_idx
        dec_logits = []
        for i in range(self.target_maxlen - 1):
            dec_out = self.decode(enc, dec_input)
            logits = self.classifier(dec_out)
            logits = tf.argmax(logits, axis=-1, output_type=tf.int32)
            last_logit = tf.expand_dims(logits[:, -1], axis=-1)
            dec_logits.append(last_logit)
            dec_input = tf.concat([dec_input, last_logit], axis=-1)
        return dec_input
    
    # def __repr__(self):
    #  return repr('Hello ' + self.name + ' your age is ' + 
    #              str(self.age) + ' and you have enrolled for ' 
    #              + self.course)


# saveto = "./datasets/mini-dataset/"
#saveto = "../sn/mini"


# tokenizer = Tokenizer.from_file("tokenizer-sanskrit.json")

# output = tokenizer.encode("यथाह वाग्भटः मातृजम् ह्यस्य हृदयम् मातुश्च")

class GetData:
    
    def __init__(self, audio_files_path, script_file):
        
        self.saveto= audio_files_path
        self.wavs = glob("{}/**/*.wav".format(self.saveto), recursive=True)
        
        random.Random(1853).shuffle(self.wavs)
        # random.shuffle(self.wavs)

        self.id_to_text = {}
        with open(script_file, encoding="utf-8") as f:
            for line in f:
                id = line.strip().split("|")[0]
                text = line.strip().split("|")[1]
                self.id_to_text[id] = text


    def pairup_audio_and_script(self, maxlen=250):
        """ returns mapping of audio paths and transcription texts """
        data = []
        for w in self.wavs:
            id = w.split("/")[-1].split(".")[0]
            if len(self.id_to_text[id]) < maxlen:
                data.append({"audio": w, "text": self.id_to_text[id]})
            
        return data

def compute_mel_spectrogram(audio, frame_length=512, frame_step=256, fft_length=512, num_mel_bins=128, sample_rate=16000):
    """
    Compute Mel spectrogram for the given audio signal using TensorFlow.
    
    Parameters:
        audio (tf.Tensor): Input audio signal.
        frame_length (int): Length of each frame.
        frame_step (int): Step size between frames.
        fft_length (int): FFT window size.
        num_mel_bins (int): Number of Mel bands to generate.
        sample_rate (int): Sampling rate of the audio.
        
    Returns:
        mel_spectrogram (tf.Tensor): Mel spectrogram.
    """
    # Compute the STFT of the input audio
    stfts = tf.signal.stft(audio, frame_length, frame_step, fft_length)
    spectrograms = tf.abs(stfts)

    # Define the Mel-scale filter bank
    num_spectrogram_bins = fft_length // 2 + 1
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz=80.0, upper_edge_hertz=sample_rate / 2)

    # Apply the Mel-scale filter bank to the spectrogram
    mel_spectrogram = tf.tensordot(spectrograms, linear_to_mel_weight_matrix, 1)
    mel_spectrogram.set_shape(spectrograms.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))
    
    # Convert to a log scale (optional)
    log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)

    #return mel_spectrogram
    return log_mel_spectrogram

# Function to compute the WER
# def compute_WER(test_dataset):
#     predictions = []
#     targets = []
#     for batch in val_ds:
#          source = batch["source"]
#          target = batch["target"].numpy()
#          bs = tf.shape(source)[0]
#          preds = model.generate(source, target_start_token_idx)
#          preds = preds.numpy()
#          for i in range(bs):
#              target_text = "".join([idx_to_token(_)  for _ in target[i, :] if _ !=0] )
#              prediction = ""
#              for idx in preds[i, :]:
#                  prediction += idx_to_token(idx)
#                  if idx == target_end_token_idx:
#                      break
             
#              predictions.append(prediction)
#              targets.append(target_text)
             
#          wer_score = wer(targets, predictions)
      
     
#          # print("-" * 100)
#          # print(f"Word Error Rate: {wer_score:.4f}")
#          # print("-" * 100)
#          # for i in np.random.randint(0, len(predictions), 5):
#          #    print(f"Target    : {targets[i]}")
#          #    print(f"Prediction: {predictions[i]}")
#          #    print("-" * 100)

#          # print("-" * 100)
#          # print(f"Word Error Rate: {wer_score:.4f}")
#          # print("-" * 100)
#          # for i in np.random.randint(0, len(predictions), 5):
#          #    print(f"Target    : {targets[i]}")
#          #    print(f"Prediction: {predictions[i]}")
#          #    print("-" * 100)
         
#          return wer_score

def compute_WER(test_dataset):
    predictions = []
    targets = []

    for batch in test_dataset:  # Iterate through all batches in the dataset
        source = batch["source"]
        target = batch["target"].numpy()
        bs = tf.shape(source)[0]

        preds = model.generate(source, target_start_token_idx)
        preds = preds.numpy()

        for i in range(bs):
            target_text = "".join([idx_to_token(_) for _ in target[i, :] if _ != 0])
            prediction = ""
            for idx in preds[i, :]:
                prediction += idx_to_token(idx)
                if idx == target_end_token_idx:
                    break

            predictions.append(prediction)
            targets.append(target_text)

    # Calculate WER for all batches
    wer_score = wer(targets, predictions)
    return wer_score


## Preprocess the dataset

## Callbacks to display predictions


class DisplayOutputs(keras.callbacks.Callback):
    def __init__(
        self, batch, idx_to_token, target_start_token_idx=7, target_end_token_idx=8
    ):
        """Displays a batch of outputs after every epoch

        Args:
            batch: A test batch containing the keys "source" and "target"
            idx_to_token: A List containing the vocabulary tokens corresponding to their indices
            target_start_token_idx: A start token index in the target vocabulary
            target_end_token_idx: An end token index in the target vocabulary
        """
        self.batch = batch
        self.target_start_token_idx = target_start_token_idx
        self.target_end_token_idx = target_end_token_idx
        self.idx_to_char = idx_to_token

    def on_epoch_end(self, epoch, logs=None):
        with open('results/results.txt', 'a') as file:
          file.write('\ntraining loss for epoch -' + 
            str(epoch) +' is ' + str("{:.3f}".format(logs['loss'])))   
          
        if epoch % 5 != 0 or epoch == 0:
            return
        source = self.batch["source"]
        target = self.batch["target"].numpy()
        bs = tf.shape(source)[0]
        preds = self.model.generate(source, self.target_start_token_idx)
        preds = preds.numpy()
        for i in range(0,bs,4):
            target_text = "".join([self.idx_to_char(_) for _ in target[i, :] if _ !=0] )
            prediction = ""
            for idx in preds[i, :]:
                prediction += self.idx_to_char(idx)
                if idx == self.target_end_token_idx:
                    break
            print(f"target:     {target_text.replace('-','')}")
            print(f"prediction: {prediction}\n")

class MetricTracker(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.train_losses = []  # List to store training losses
        self.val_wer_scores = []  # List to store validation WER

    def on_epoch_end(self, epoch, logs=None):
        # Append training loss
        loss = logs.get('loss')
        if loss is not None:
            self.train_losses.append(loss)
        else:
            print(f"Epoch {epoch + 1}: No loss value found in logs!")

        # Compute and append validation WER
        curr_wer = compute_WER(val_ds)
        self.val_wer_scores.append(curr_wer)
        print(f"Epoch {epoch + 1}: Loss: {loss:.4f}, Validation WER: {curr_wer:.4f}")

        # Save metrics to file
        with open('results/results.txt', 'a') as file:
            file.write(f"Epoch {epoch + 1}: Loss: {loss:.4f}, WER: {curr_wer:.4f}\n")

  

def plot_metrics(metric_tracker):
    # Debugging: Print the metric data
    print(f"Train Losses: {metric_tracker.train_losses}")
    print(f"Validation WER: {metric_tracker.val_wer_scores}")

    # Plot training loss
    plt.figure(figsize=(8, 5))
    epochs = range(1, len(metric_tracker.train_losses) + 1)  # Dynamically set epochs
    plt.plot(epochs, metric_tracker.train_losses, label="Training Loss", color="blue")
    plt.title("Training Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.xticks(range(1, len(metric_tracker.train_losses) + 1, 5))  
    plt.legend()
    plt.grid()
    plt.show()

    # Plot validation WER
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, metric_tracker.val_wer_scores, label="Validation WER", color="orange")
    plt.title("Validation WER Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("WER")
    plt.xticks(range(1, len(metric_tracker.val_wer_scores) + 1, 5))  
    plt.legend()
    plt.grid()
    plt.show()





class VectorizeChar:
    def __init__(self, max_len):
        self.tokenizer = Tokenizer.from_file(tokenizer_file)
        self.max_len = max_len


    def __call__(self, text):
        text = text.lower()
        text = text[: self.max_len - 2]
        text = "<" + text + ">"
        tmp = self.tokenizer.encode(text).ids
        pad_len = self.max_len - len(tmp)
        # tmp = self.tokenizer.encode(text).tokens + [0] * pad_len
        if(len(tmp) > self.max_len):
            print(f" max target string length greater than set threshold ({self.max_len}): {tmp}")
            exit(-1)
        
        return tmp + [0] * pad_len
# output = tokenizer.encode("यथाह वाग्भटः मातृजम् ह्यस्य हृदयम् मातुश्च")

    def get_vocabulary_size(self):
        return self.tokenizer.get_vocab_size()


    def idx_to_token(self):
       return self.tokenizer.id_to_token

vectorizer = VectorizeChar(max_target_len)
with open('results/results.txt', 'a') as file:
    file.write('\n------------------------------------------\n')
    file.write('\nResults computed on  ' + str(datetime.now()) + ' \n\n\n')


def create_text_ds(data):
    texts = [_["text"] for _ in data]
    text_ds = [vectorizer(t) for t in texts]
    text_ds = tf.data.Dataset.from_tensor_slices(text_ds)
    return text_ds


def path_to_audio(path):
    

  
    # # srate = 22050 # sampling rate
    # srate = 16000 # sampling rate
    # dur = 10 # 10 sec audio sequences assumed
    # frame_length = 512
    # frame_step = 256
    # fft_length = 512
 
    
    audio = tf.io.read_file(path)
    audio, _ = tf.audio.decode_wav(audio, 1)
    audio = tf.squeeze(audio, axis=-1)
 
    # tf.dtypes.cast(audio, tf.int32)
    # tf.dtypes.cast(audio, tf.float16)
    

 
    seq_len = int((srate*dur - frame_length)/frame_step + 1)   
    
    # spectrogram using stft
   
    # stfts = tf.signal.stft(audio, frame_length, frame_step, fft_length)
    # x = tf.math.pow(tf.abs(stfts), 0.5)
    log_mel_spectrogram = compute_mel_spectrogram(audio, frame_length=512, frame_step=256, fft_length=512, num_mel_bins=128, sample_rate=16000)
    # normalisation
    means = tf.math.reduce_mean(log_mel_spectrogram, 1, keepdims=True)
    stddevs = tf.math.reduce_std(log_mel_spectrogram, 1, keepdims=True)
    log_mel_spectrogram = (log_mel_spectrogram - means) / (stddevs + 1e-9 )
    # audio_len = tf.shape(x)[0]
    # padding to 10 seconds sampled at 16KHz
    pad_len = seq_len
    paddings = tf.constant([[0, pad_len], [0, 0]])
    log_mel_spectrogram = tf.pad(log_mel_spectrogram, paddings, "CONSTANT")[:pad_len, :]

    # Iterate and print values
    # for item in x:
    #   print(item)
    
    # sys.exit(-1)

    return log_mel_spectrogram

# Data augmentation functions
def add_noise(audio, noise_factor=0.007):
    if tf.random.uniform(()) < 0.7:  # Apply noise randomly
        noise = tf.random.normal(tf.shape(audio), stddev=noise_factor)
        return audio + noise
    return audio  # Return unchanged audio if noise is not applied

def time_stretch(audio, rate_min=0.8, rate_max=1.2):
    if tf.random.uniform(()) < 0.3:  # Apply time stretching randomly
        audio_np = audio.numpy()  # Convert TensorFlow tensor to NumPy array
        rate = tf.random.uniform((), minval=rate_min, maxval=rate_max).numpy()
        stretched_audio = librosa.effects.time_stretch(audio_np, rate=rate)
        return tf.convert_to_tensor(stretched_audio, dtype=tf.float32)
    return audio  # Return unchanged audio if stretching is not applied

def pitch_shift(audio, sr, n_steps_min=-3, n_steps_max=3):
    if tf.random.uniform(()) < 0.5:  # Apply pitch shifting randomly
        audio_np = audio.numpy()  # Convert TensorFlow tensor to NumPy array
        n_steps = tf.random.uniform((), minval=n_steps_min, maxval=n_steps_max).numpy()
        shifted_audio = librosa.effects.pitch_shift(audio_np, sr=sr, n_steps=n_steps)
        return tf.convert_to_tensor(shifted_audio, dtype=tf.float32)
    return audio  # Return unchanged audio if pitch shifting is not applied


def create_audio_ds(data):
    flist = [_["audio"] for _ in data]
    audio_ds = tf.data.Dataset.from_tensor_slices(flist)
    audio_ds = audio_ds.map(
        path_to_audio, num_parallel_calls=tf.data.AUTOTUNE
    )
    return audio_ds

def create_audio_ds_with_augmentation(data, augment=True, num_augmentations=2):
    audio_transcripts = []
    for item in data:
        file_path = item["audio"]
        transcript = item["text"]

        # Extract the base name of the source file without the directory path
        base_name = os.path.basename(file_path).replace(".wav", "")

        # Include the original audio and its transcript
        audio_transcripts.append({"audio": file_path, "text": transcript})

        if augment and tf.random.uniform(()) < 0.7:  # Apply augmentation to 70% of files
            audio_binary = tf.io.read_file(file_path)
            audio, _ = tf.audio.decode_wav(audio_binary, desired_channels=1)
            audio = tf.squeeze(audio)

            for aug_idx in range(1, num_augmentations + 1):
                augmented_audio = audio

                # Apply augmentations randomly
                augmented_audio = add_noise(augmented_audio)
                augmented_audio = time_stretch(augmented_audio)
                augmented_audio = pitch_shift(augmented_audio, sr=16000)

                # Generate the augmented file name
                temp_audio_path = f"/home/sysadm/samapankar/term_proj/datasets/final_devanagari/augmented_files/aug_{base_name}_audio_{aug_idx}.wav"
                
                # Save the augmented audio to the generated file path
                sf.write(temp_audio_path, augmented_audio.numpy(), 16000)
                audio_transcripts.append({"audio": temp_audio_path, "text": transcript})

    return audio_transcripts



def create_tf_dataset(data, bs):
    audio_ds = create_audio_ds(data)
    text_ds = create_text_ds(data)


    ds = tf.data.Dataset.zip((audio_ds, text_ds))
    
    # ds_size = audio_ds.cardinality().numpy()
    
    
    ds = ds.map(lambda x, y: {"source": x, "target": y})

    # ds = ds.shuffle(5000,seed=1234,reshuffle_each_iteration=True)

    ds = ds.batch(bs)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

def create_tf_dataset_with_augmentation(data, bs, augment=True, num_augmentations=2):
    audio_transcripts = create_audio_ds_with_augmentation(data, augment, num_augmentations)

    audio_ds = tf.data.Dataset.from_tensor_slices([item["audio"] for item in audio_transcripts])
    text_ds = tf.data.Dataset.from_tensor_slices([vectorizer(item["text"]) for item in audio_transcripts])

    audio_ds = audio_ds.map(path_to_audio, num_parallel_calls=tf.data.AUTOTUNE)

    ds = tf.data.Dataset.zip((audio_ds, text_ds))
    ds = ds.map(lambda x, y: {"source": x, "target": y})
    ds = ds.shuffle(5000, seed=1234, reshuffle_each_iteration=True)
    ds = ds.batch(bs)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds



if train_enable:
    train_data = GetData("/home/sysadm/samapankar/term_proj/datasets/final_devanagari/train", 
                         "/home/sysadm/samapankar/term_proj/datasets/final_devanagari/train.txt")
    test_data = GetData("/home/sysadm/samapankar/term_proj/datasets/final_devanagari/test", 
                        "/home/sysadm/samapankar/term_proj/datasets/final_devanagari/test.txt")
    
# if train_enable:
#     train_data = GetData("/home/sysadm/samapankar/term_proj/minidata/train", 
#                          "/home/sysadm/samapankar/term_proj/minidata/train.txt")
#     test_data = GetData("/home/sysadm/samapankar/term_proj/minidata/test", 
#                         "/home/sysadm/samapankar/term_proj/minidata/test.txt")

    # Training dataset with augmentation
    ds = create_tf_dataset_with_augmentation(
        train_data.pairup_audio_and_script(),
        bs=batch_size, 
        augment=True, 
        num_augmentations=2
    )
    
    # Validation dataset without augmentation
    val_ds = create_tf_dataset(
        test_data.pairup_audio_and_script(), 
        bs=batch_size
    )

    # print('Number of input files is:', len(train_data.pairup_audio_and_script()))
    print(f"Original dataset size: {len(train_data.pairup_audio_and_script())}")
    total_files = sum(1 for _ in ds.unbatch())
    print(f"Total dataset size (original + augmented): {total_files}")



# Iterate and print values
#for item in ds:
#    print(item)
if native_scheduler:
 """## Learning rate schedule"""
 class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000):
    super().__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps

  def __call__(self, step):
    step = tf.cast(step, dtype=tf.float32)
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
else:
 class CustomSchedule(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
        self,
        init_lr,
        lr_after_warmup,
        final_lr,
        warmup_epochs,
        decay_epochs,
        steps_per_epoch
    ):
        super().__init__()
        self.init_lr = init_lr
        self.lr_after_warmup = lr_after_warmup
        self.final_lr = final_lr
        self.warmup_epochs = warmup_epochs
        self.decay_epochs = decay_epochs
        self.steps_per_epoch = steps_per_epoch

    def calculate_lr(self, epoch):
        """ linear warm up - linear decay """
        warmup_lr = (
            self.init_lr
            + ((self.lr_after_warmup - self.init_lr) / (self.warmup_epochs - 1)) * tf.cast(epoch,tf.float32)
        )
        decay_lr = tf.math.maximum(
            self.final_lr,
            self.lr_after_warmup
            - ( tf.cast(epoch,tf.float32) - self.warmup_epochs)
            * (self.lr_after_warmup - self.final_lr)
            / self.decay_epochs,
        )
        return tf.math.minimum(warmup_lr, decay_lr)

#     def __call__(self, step):
#         epoch = step // self.steps_per_epoch
#         return (self.calculate_lr(epoch))

# The vocabulary to convert predicted indices into characters
idx_to_token = vectorizer.idx_to_token()

#some arbitrary value for non-train mode   
no_steps_per_epoch = 5
"""## Create & train the end-to-end model"""
if (train_enable == True):
    
    batch = next(iter(val_ds))

    no_steps_per_epoch = int(len(ds)/batch_size)
    
    display_cb = DisplayOutputs(
        batch, idx_to_token, target_start_token_idx, target_end_token_idx
    )  # set the arguments as per vocabulary index for '<' and '>'
    
    

num_classes=vectorizer.get_vocabulary_size()

    
loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True, 
                                                  label_smoothing=0.1)



    
model = Transformer(num_hid, num_head, num_feed_forward, source_maxlen,
        target_maxlen,num_layers_enc, num_layers_dec, num_classes)
    
    # -------------------------------*************************--------------------------------
    # -------------------------------*************************--------------------------------
    # -------------------------------*************************--------------------------------


if native_scheduler:
    learning_rate = CustomSchedule(num_hid)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                      epsilon=1e-9)
else:
    
    learning_rate = CustomSchedule(
    init_lr=0.00001,
    lr_after_warmup=0.001,
    final_lr=0.00001,
    warmup_epochs=90,
    decay_epochs=5,
    steps_per_epoch=no_steps_per_epoch,
    )
    optimizer = keras.optimizers.Adam(learning_rate)
    
    

model.compile(loss=loss_fn, optimizer=optimizer)



with open('results/results.txt', 'a') as file:
  file.write('==========================================================')  
  file.write('\nModel parameters: \n num_hid  {}, num_head {}, num_feed_forward {},\n\
   source_maxlen {}, target_maxlen {}, num_layers_enc {},\n\
   num_layers_dec {}  num_classes {}, drop_out_enc {}, drop_out_dec {}\n"'.format(num_hid, num_head, num_feed_forward, source_maxlen,
  target_maxlen,num_layers_enc, num_layers_dec,num_classes,drop_out_enc,drop_out_dec))
  file.write('==========================================================')  
     
         
             

#history = model.fit(ds, validation_data=val_ds, callbacks=[display_cb], epochs=50)
if(train_enable):
    # history = model.fit(ds, callbacks=[display_cb], epochs=150)
    
    # Initialize the metric tracker
    metric_tracker = MetricTracker()
    
    
    history = model.fit(ds, callbacks=[display_cb, EarlyStoppingAtMinLoss(15), metric_tracker], epochs=50)


    #Save the model architecture        
    json_string = model.to_json()
    with open('model_architecture.json', 'w') as f:
        f.write(json_string)

    model.summary()
    
    # Plot metrics after training
    plot_metrics(metric_tracker)

#loss, acc = model.evaluate(val_ds,verbose=2)

# Calling `save('my_model.keras')` creates a zip archive `my_model.keras`.
    # model.save('mymodel.keras')
    # model.save_weights("MyModel_tf",save_format='tf')




    with open('results/results.txt', 'a') as file:
        file.write('\n\nTraining loss is ' + str(history.history))
        file.write('\n The minimum loss is ' +
                   "{:.3f}".format(min((history.history)['loss'])) + '\n')
        with redirect_stdout(file):
            # model.summary()
            print('\n------------------------------------------\n')
            print('\n------------------------------------------\n')

    loss = model.evaluate(val_ds, verbose=2)
    print('validation loss is: ', loss)


# *********************************************************************************
# **************************TESTING CODE IS BELOW**********************************
# *********************************************************************************





#loss, acc = model.evaluate(val_ds)

# model_test = Transformer(
#     num_hid=200,
#     num_head=2,
#     num_feed_forward=400,
#     source_maxlen=100,
#     target_maxlen=max_target_len,
#     num_layers_enc=4,
#     num_layers_dec=1,
#     num_classes=(vectorizer.get_vocabulary_size())
# )



# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

#  loss_fn = tf.keras.losses.CategoricalCrossentropy(
#     from_logits=True, label_smoothing=0.1
# )

# # loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
# #     from_logits=True,reduction='none')
    
# learning_rate = CustomSchedule(
#     init_lr=0.00001,
#     lr_after_warmup=0.001,
#     final_lr=0.00001,
#     warmup_epochs=15,
#     decay_epochs=85,
#     steps_per_epoch=len(ds),
# )

# optimizer = keras.optimizers.Adam(learning_rate)
# model_test.compile(optimizer=optimizer, loss=loss_fn)

# compute_WER(val_ds)


        

# Let's check:
# Evaluate the model
############history = model_test.fit(ds, callbacks=[display_cb], epochs=1)








def call_test(audio_file):
    model_test = Transformer(num_hid, num_head, num_feed_forward, source_maxlen,
        target_maxlen,num_layers_enc, num_layers_dec, num_classes)
    
    # # model_test = model
    # learning_rate = CustomSchedule(
    # init_lr=0.00001,
    # lr_after_warmup=0.001,
    # final_lr=0.00001,
    # warmup_epochs=90,
    # decay_epochs=5,
    # steps_per_epoch=10,
    # )
    
    model_test.compile(optimizer=optimizer, loss=loss_fn)
    
    # It can be used to reconstruct the model identically.
    #model_test.load_weights("MyModel_tf.weights.h5")

    # y, s = librosa.load(audio_file, sr=16000) # Downsample 44.1kHz to 8kHz
    


    
    

 
    
     # Load audio data
    audio_data, sample_rate = librosa.load(audio_file,  sr=16000) # Downsample 44.1kHz to 8kHz)    

    # Decode using tf.audio.decode_wav (assuming resampled_audio is a NumPy array)
    # decoded_audio, _ = tf.audio.decode_wav(, 1)
    
    # Convert to a supported data type (e.g., float32)
    audio_data = audio_data.astype(np.int32)
    audio_data = audio_data.astype(np.float32)
    # audio, _ = tf.audio.decode_wav(audio, 1)
    audio = tf.convert_to_tensor(audio_data)
 
 
    seq_len = int((srate*dur - frame_length)/frame_step + 1)   
    
    # spectrogram using stft
   
    stfts = tf.signal.stft(audio, frame_length, frame_step, fft_length)
    x = tf.math.pow(tf.abs(stfts), 0.5)
    # normalisation
    means = tf.math.reduce_mean(x, 1, keepdims=True)
    stddevs = tf.math.reduce_std(x, 1, keepdims=True)
    x = (x - means) / (stddevs + 1e-9 )
    audio_len = tf.shape(x)[0]
    # padding to 10 seconds sampled at 16KHz
    pad_len = seq_len
    paddings = tf.constant([[0, pad_len], [0, 0]])
    spect = tf.pad(x, paddings, "CONSTANT")[:pad_len, :]
    
    
    
    # spect = path_to_audio(audio_file)
    spect = tf.expand_dims(spect, axis=0)


    preds = model_test.generate(spect, target_start_token_idx)
    preds = preds.numpy()
    prediction = ""
    for idx in preds[0].tolist():
        prediction += idx_to_token(idx)
        if idx == target_end_token_idx:
            break
        
    print(f"the prediction is {prediction}")
    return prediction    


# # call_test('/tmp/gradio/91883f375e4041c8d8774e7e067301de25909588/audio-0-100.wav')
# call_test('/home/sevak/Documents/workspace/src/datasets/mini-dataset/data/train/sp003-000001_RV_03.wav')

# #elements = list(val_ds.as_numpy_iterator())
# #loss, acc = model_test.evaluate(elements[0]['source'][0], elements[0]['target'][0],verbose=2)
# #print("Restored model, accuracy: {:5.2f}%".format(100 * acc))


# import gradio as gr


# def transcribe(audio):
#     print(audio)
#     text = call_test(audio)
#     return text

# iface = gr.Interface(
#     fn=transcribe,
#     inputs=gr.Audio(source="microphone", type="filepath"),
#     outputs="text",
#     title="STT of Sanskrit Demo",
#     description="Realtime demo for Sanskrit speech recognition using a transformer model.",
# )

# iface.launch()


