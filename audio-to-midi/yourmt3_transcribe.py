import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'amt/src')))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from model_helper import load_model_checkpoint, transcribe

# YPTF.MoE+Multi (noPS) — best noPS checkpoint per the HF space, precision 32 for CPU
checkpoint = "mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops@last.ckpt"
args = [checkpoint, '-p', '2024', '-tk', 'mc13_full_plus_256', '-dec', 'multi-t5',
        '-nl', '26', '-enc', 'perceiver-tf', '-sqr', '1', '-ff', 'moe',
        '-wf', '4', '-nmoe', '8', '-kmoe', '2', '-act', 'silu', '-epe', 'rope',
        '-rp', '1', '-ac', 'spec', '-hop', '300', '-atc', '1', '-pr', '32']

model = load_model_checkpoint(args=args, device="cpu")
print("model loaded")

for audio_file in sys.argv[1:]:
    info = {'filepath': audio_file,
            'track_name': os.path.splitext(os.path.basename(audio_file))[0]}
    midifile = transcribe(model, info)
    print("MIDI written:", midifile)
