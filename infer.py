import torch
import librosa
import soundfile as sf
from transformers import AutoConfig

 
from model import Autoencoder
from safetensors.torch import load_model
from xcodec2.modeling_xcodec2 import XCodec2Model
 
autoencoder = Autoencoder().cuda()
load_model(autoencoder, "model.safetensors")
model_path = "HKUSTAudio/xcodec2"  
 
model = XCodec2Model.from_pretrained(model_path)
model.eval().cuda()

wav, sr = librosa.load("test.mp3", sr=16000, mono=True)   # MUST be 16KHz
wav_tensor = torch.from_numpy(wav).float().unsqueeze(0)  # Shape: (1, T)

 
with torch.no_grad():
    # TODO: task conditional LAE on these VQ codes
    vq_code = model.encode_code(input_waveform=wav_tensor[:, : 16000 * 3])
    inv_seq = autoencoder(vq_code, torch.zeros_like(vq_code, device=vq_code.device)).argmax(-1)
    recon_wav = model.decode_code(inv_seq).cpu()       # Shape: (1, 1, T')
 
sf.write("converted.mp3", recon_wav[0, 0, :].numpy(), sr)
print("Done! Check converted.mp3")

