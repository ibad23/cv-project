import json
import os

def modify_ultraplate():
    notebook_path = 'code/ultraplate_pipeline.ipynb'
    if not os.path.exists(notebook_path):
        print(f"Notebook {notebook_path} not found.")
        return

    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    modified = False
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            
            # 1. Override Real-ESRGAN weights
            if 'class RealESRGANModel(SRModel):' in source:
                new_init = """    def __init__(self):
        # Path to fine-tuned weights from Workstream 3
        w = "models/realesrgan_x4_french_lcd/realesrgan_x4_french_lcd.pth"
        if not os.path.exists(w):
            print(f"Fine-tuned weights not found at {w}, falling back to Hub.")
            from huggingface_hub import hf_hub_download
            for repo, fn in [("ai-forever/Real-ESRGAN",  "RealESRGAN_x4.pth"),
                             ("ai-forever/Real-ESRGAN",  "RealESRGAN_x4plus.pth")]:
                try:
                    w = hf_hub_download(repo_id=repo, filename=fn); break
                except Exception: continue
        
        self.model = RRDBNet(scale=4, nb=23)
        sd = torch.load(w, map_location="cpu")
        if   "params_ema" in sd: sd = sd["params_ema"]
        elif "params"     in sd: sd = sd["params"]
        self.model.load_state_dict(sd, strict=True)
        self.model.to(DEVICE).eval()
"""
                # Replace the __init__ method
                import re
                cell['source'] = [re.sub(r'def __init__\(self\):.*?self\.model\.to\(DEVICE\)\.eval\(\)', new_init, source, flags=re.DOTALL)]
                modified = True

            # 2. Override fast-plate-ocr weights
            if 'class FastPlateOCREngine(OCREngine):' in source:
                new_init_ocr = """    def __init__(self, model_name="european-plates-mobile-vit-v2-model"):
        from fast_plate_ocr import LicensePlateRecognizer
        # Path to fine-tuned weights from Workstream 2
        custom_weights = "models/fastplate_french_siv/fastplate_french_siv.onnx"
        if os.path.exists(custom_weights):
            print(f"Loading custom fine-tuned fast-plate-ocr weights from {custom_weights}")
            self.m = LicensePlateRecognizer(custom_weights)
        else:
            self.m = LicensePlateRecognizer(model_name)
        self.cfg = self.m.config
        self.pad = self.cfg.pad_char
"""
                import re
                cell['source'] = [re.sub(r'def __init__\(self, model_name=.*?\):.*?self\.pad = self\.cfg\.pad_char', new_init_ocr, source, flags=re.DOTALL)]
                modified = True

    if modified:
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print(f"Successfully modified {notebook_path} with fine-tuned weight overrides.")
    else:
        print(f"Could not find target classes in {notebook_path}.")

if __name__ == "__main__":
    modify_ultraplate()
