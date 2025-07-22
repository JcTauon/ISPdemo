
"""
A software ISP written in pure Python/NumPy for
converting single-frame DNG RAW
into 8-bit JPGs.
"""
import argparse
import os
from pathlib import Path
from typing import Tuple

import rawpy                       # RAW 解码
import numpy as np                 # 数值运算
from PIL import Image              # 图像 I/O



class SoftwareISP:
    """Core classes that encapsulate the complete ISP pipeline."""
    _PATTERN2IDX = {
        'RGGB': np.array([[0, 1],
                          [1, 2]]),
        'GRBG': np.array([[1, 0],
                          [2, 1]]),
        'GBRG': np.array([[1, 2],
                          [0, 1]]),
        'BGGR': np.array([[2, 1],
                          [1, 0]]),
    }

    def __init__(self, raw_path: str):
        self.raw = rawpy.imread(raw_path)
        self.raw_image = self.raw.raw_image_visible.astype(np.float32)
        self.height, self.width = self.raw_image.shape

        self.bayer_pattern = self.raw.raw_pattern.astype(int)
        self.black_levels = np.asarray(self.raw.black_level_per_channel, np.float32)
        self.white_balance = np.asarray(self.raw.camera_whitebalance, np.float32)
        self.white_level = float(self.raw.white_level)
        rgb_xyz = getattr(self.raw, "rgb_xyz_matrix", None)

        def _is_valid(m: np.ndarray) -> bool:
            return m is not None and m.size == 9 and np.max(np.abs(m)) > 1e-6

        if _is_valid(rgb_xyz):
            self.color_matrix = rgb_xyz.reshape(3, 3)
        else:
            cm = np.asarray(self.raw.color_matrix, np.float32)
            cm_flat = cm.ravel()
            if cm_flat.size >= 9 and np.max(np.abs(cm_flat[:9])) > 1e-6:
                self.color_matrix = cm_flat[:9].reshape(3, 3)
            else:
                self.color_matrix = np.eye(3, dtype=np.float32)


        if not np.all(np.isfinite(self.white_balance)) or self.white_balance[1] == 0:
            self.white_balance = np.array([2.0, 1.0, 2.0, 1.0], np.float32)
        if not np.all(np.isfinite(self.color_matrix)):
            self.color_matrix = np.eye(3, dtype=np.float32)

        pattern_str = ''.join('RGBG'[i] for i in self.bayer_pattern.ravel())
        self._tile_channel = self._PATTERN2IDX.get(pattern_str, self._PATTERN2IDX['RGGB'])

    # ==================== Main entrance ===================
    def process(self, contrast: float = 1.0, sharpen_amount: float = 0.5, dump = False):
        img = self._black_level_correction(self.raw_image)
        if dump: self._dump(img, "01_blc", apply_gamma=False)

        img = self._white_balance(img)
        if dump: self._dump(img, "02_wb", apply_gamma=False)

        img = self._demosaic(img)
        if dump: self._dump(img, "03_demosaic", apply_gamma=False)

        img = self._color_correction(img)
        if dump: self._dump(img, "04_ccm", apply_gamma=False)

        img = self._gamma_correction(img)
        if dump: self._dump(img, "05_gamma")

        img = self._apply_tone_curve(img, strength=contrast)
        if dump: self._dump(img, "06_tone")

        img = self._sharpen(img, amount=sharpen_amount)
        if dump: self._dump(img, "07_sharpen")
        return np.clip(img, 0, 1).astype(np.float32)

    # ================= Realization of phases ====================
    def _black_level_correction(self, bayer: np.ndarray) -> np.ndarray:
        bl_r, bl_g1, bl_b, bl_g2 = self.black_levels
        tile_bl = np.array([[bl_r, bl_g1],
                            [bl_g2, bl_b]], dtype=np.float32)
        mask = np.tile(tile_bl, (self.height // 2, self.width // 2))
        corrected = np.maximum(bayer - mask, 0.0)
        normalised = corrected / (self.white_level - mask)
        return np.clip(normalised, 0.0, 1.0)

    def _white_balance(self, bayer: np.ndarray) -> np.ndarray:
        wb_r, wb_g1, wb_b, wb_g2 = self.white_balance
        wb_r, wb_b, wb_g2 = wb_r / wb_g1, wb_b / wb_g1, wb_g2 / wb_g1
        tile_wb = np.array([[wb_r, 1.0],
                            [wb_g2, wb_b]], dtype=np.float32)
        mask = np.tile(tile_wb, (self.height // 2, self.width // 2))
        return bayer * mask

    def _demosaic(self, bayer: np.ndarray) -> np.ndarray:
        H, W = bayer.shape
        rgb = np.zeros((H, W, 3), np.float32)

        for i in range(2):
            for j in range(2):
                ch = self._tile_channel[i, j]
                rgb[i::2, j::2, ch] = bayer[i::2, j::2]

        kh = np.array([[0, .5, 0],
                       [0, 0, 0],
                       [0, .5, 0]], np.float32)
        kv = kh.T
        kd = np.array([[.25, 0, .25],
                       [0, 0, 0],
                       [.25, 0, .25]], np.float32)

        def conv(src: np.ndarray, k: np.ndarray) -> np.ndarray:
            pad = np.pad(src, ((1, 1), (1, 1)), mode='edge')
            out = np.zeros_like(src)
            k = k[::-1, ::-1]
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    out += k[dy + 1, dx + 1] * pad[1 + dy:H + 1 + dy, 1 + dx:W + 1 + dx]
            return out

        for c in range(3):
            miss = rgb[..., c] == 0
            interp = 0.5 * (conv(rgb[..., c], kh) + conv(rgb[..., c], kv)) \
                if c == 1 else conv(rgb[..., c], kd)
            rgb[..., c][miss] = interp[miss]
        return rgb

    def _color_correction(self, rgb: np.ndarray) -> np.ndarray:
        flat = rgb.reshape(-1, 3) @ self.color_matrix.T
        return np.clip(flat.reshape(rgb.shape), 0.0, 1.0)

    @staticmethod
    def _gamma_correction(rgb: np.ndarray) -> np.ndarray:
        return np.power(np.clip(rgb, 0.0, 1.0), 1/2.2, dtype=np.float32)

    @staticmethod
    def _apply_tone_curve(rgb: np.ndarray, strength: float) -> np.ndarray:
        k = 5.0 * strength
        y = 1 / (1 + np.exp(-k * (rgb - 0.5)))
        y_min, y_max = 1/(1+np.exp(k/2)), 1/(1+np.exp(-k/2))
        return (y - y_min) / (y_max - y_min)

    def _sharpen(self, rgb: np.ndarray, amount: float) -> np.ndarray:
        if amount <= 0:
            return rgb
        kernel = np.ones((3, 3), np.float32) / 9.0
        pad = np.pad(rgb, ((1, 1), (1, 1), (0, 0)), mode='reflect')
        blurred = sum(kernel[dy+1, dx+1] *
                      pad[1+dy:self.height+1+dy, 1+dx:self.width+1+dx]
                      for dy in (-1, 0, 1) for dx in (-1, 0, 1))
        sharpened = rgb + (rgb - blurred) * amount
        return np.clip(sharpened, 0.0, 1.0)

    def _dump(self, img, tag, apply_gamma=True, out_dir="debug"):
        Path(out_dir).mkdir(exist_ok=True)
        disp = np.clip(img, 0, 1)
        if apply_gamma:
            disp = disp ** (1 / 2.2)
        Image.fromarray((disp * 255 + 0.5).astype(np.uint8)) \
            .save(os.path.join(out_dir, f"{tag}.png"))


#===================== CLI & save =======================
def _save_jpeg(img: np.ndarray, path: str):
    img8 = (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)
    Image.fromarray(img8).save(path, quality=95, subsampling=0)


def _parse() -> Tuple[str, str, float, float, bool]:
    p = argparse.ArgumentParser(description="Software ISP – DNG RAW → JPG")
    p.add_argument("input_path", help="Enter DNG/RAW Path")
    p.add_argument("output_path", help="Out JPG Path")
    p.add_argument("--contrast", type=float, default=1.0, help="Contrast intensity")
    p.add_argument("--sharpen", type=float, default=0.5, help="Sharpening Intensity")
    p.add_argument("--dump", action="store_true", help="Save phase PNGs for debugging")
    a = p.parse_args()
    return a.input_path, a.output_path, a.contrast, a.sharpen, a.dump


def main():
    inp, outp, c, s, dump = _parse()
    print(f"[ISP] Start to process {inp} → {outp}")
    rgb = SoftwareISP(inp).process(contrast=c, sharpen_amount=s, dump=dump)
    _save_jpeg(rgb, outp)
    print(f"[ISP] Saved! {outp}")


if __name__ == "__main__":
    main()
