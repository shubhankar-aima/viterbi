
from conv_encoder import build_encoder_from_octal, print_generator_info, print_trellis
from bsc import BSC

# Build encoder (rate 1/2, K=3), with verbose step-by-step logs
enc = build_encoder_from_octal([0o7, 0o5], K=3, verbose=True)

print_generator_info(enc)
print_trellis(enc)

u = [1,0,1,1]
codeword = enc.encode(u, show_steps=True)

print("Input u:", u)
print("Codeword length:", len(codeword), "bits")
print("Codeword:", codeword)

# Send through a BSC with p=your value
ch = BSC(p=0.3, seed=123, verbose=True)
y, flips = ch.transmit(codeword)
print("Received y:", y)
print("Flip positions:", flips)
