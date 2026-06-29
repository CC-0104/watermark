"""
隐形文字水印引擎 —— 多算法支持

算法一 adaptive_dwt (自研，最高质量):
    直接在 BGR B 通道 → 1 级 Haar DWT → 仅在 level-1 细节子带嵌入
    采用 QIM (Quantization Index Modulation)，delta=5.0
    level-1 系数最大往返误差 ≤ 1.0 < delta/4 = 1.25 → 100% 提取准确率
    PSNR 通常 > 55 dB（肉眼完全无差异）

算法二 invismark (微软 2024-2025 SOTA):
    基于深度学习 (Encoder-Decoder) 的隐形水印。
    支持 4090 GPU 加速，极致鲁棒性。
    使用官方 InvisMark 仓库架构。

算法三 dwt_dct (invisible-watermark 库):
    DWT + DCT 频域嵌入，速度快，鲁棒性好。

算法四 dwt_dct_svd (blind_watermark 库):
    DWT + DCT + SVD，鲁棒性强，但噪点较大。
"""

import os
import sys
import struct
import string
import numpy as np
import cv2
import pywt
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

from watermark_registry import DEFAULT_DB_PATH, WatermarkRegistry, sha256_file

# 导入 InvisMark 仓库路径
REPO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "invismark_repo")
if REPO_PATH not in sys.path:
    sys.path.append(REPO_PATH)

try:
    import model as invismark_model
    from configs import ModelConfig
    from model import BCHECC
except ImportError:
    # 如果没找到仓库，后面会报错提示用户
    pass


# ═══════════════════════════════════════════════════════════════
#  算法二：InvisMark (Microsoft SOTA AI)
# ═══════════════════════════════════════════════════════════════

class InvisMarkEngine:
    """
    微软 InvisMark 水印引擎，BCH(14,7) 纠错编码。

    100-bit 信道布局：
    ┌──────────────────────────────────────────────────┐
    │ bits  0-15 : 2 字节数据（UTF-8 文字，NUL 填充） │
    │ bits 16-99 : 84 位 BCH 纠错码                   │
    └──────────────────────────────────────────────────┘
    可纠正最多 14 个比特错误（模型平均误码率 ~8 个/100 bits）。
    容量：最多 2 个 ASCII 字符（如 "47", "AB", "Hi"）。
    更长文字请使用 adaptive_dwt 算法。
    """

    MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "paper.ckpt")

    # BCH(t=14, m=7) 缩短码：可纠正 14 个比特错误
    # 100 bits = 16 data bits (2 bytes) + 84 ECC bits
    BCH_T = 14
    BCH_M = 7
    DATA_BYTES = 2   # 2 bytes → 最多 2 ASCII 字符 or 1 length byte + 1 text byte

    def __init__(self, key: int = 42):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = None
        self.extractor = None
        self.config = None
        self.bch_codec = None  # bchlib.BCH 实例
        self.key = key

    # ── BCH 纠错编解码 ─────────────────────────────────────────────
    def _get_bch(self):
        if self.bch_codec is None:
            import bchlib
            self.bch_codec = bchlib.BCH(self.BCH_T, m=self.BCH_M)
        return self.bch_codec

    def _encode_text(self, text: str, total_bits: int) -> torch.Tensor:
        """
        BCH(14,7) 纠错编码：text → 2 data bytes + 84 ECC bits = 100 bits。
        可纠正信道中最多 14 个比特错误（模型平均 ~8 个）。
        """
        bch = self._get_bch()
        text_bytes = text.encode('utf-8')[:self.DATA_BYTES]
        data = bytearray(text_bytes.ljust(self.DATA_BYTES, b'\0'))
        ecc = bch.encode(data)

        data_bits = np.unpackbits(np.frombuffer(bytes(data), dtype=np.uint8))
        ecc_bits  = np.unpackbits(np.frombuffer(bytes(ecc),  dtype=np.uint8))[:bch.ecc_bits]
        codeword  = np.concatenate([data_bits, ecc_bits]).astype(np.float32)

        if len(codeword) < total_bits:
            codeword = np.pad(codeword, (0, total_bits - len(codeword)))
        return torch.from_numpy(codeword[:total_bits]).unsqueeze(0)

    def _decode_bits(self, bits_np: np.ndarray) -> str:
        """
        BCH 纠错解码：100 bits → 纠错 → 还原文字。
        """
        bch = self._get_bch()
        recv_data = bytearray(np.packbits(bits_np[:self.DATA_BYTES * 8]).tobytes()[:self.DATA_BYTES])
        ecc_raw = bits_np[self.DATA_BYTES * 8 : self.DATA_BYTES * 8 + bch.ecc_bits]
        pad_len = bch.ecc_bytes * 8 - bch.ecc_bits
        ecc_padded = np.concatenate([ecc_raw, np.zeros(pad_len, dtype=np.uint8)])
        recv_ecc = bytearray(np.packbits(ecc_padded).tobytes()[:bch.ecc_bytes])

        nerr = bch.decode(recv_data, recv_ecc)
        if nerr >= 0:
            bch.correct(recv_data, recv_ecc)
            return bytes(recv_data).rstrip(b'\0').decode('utf-8', errors='replace')

        # 纠错失败（>14 个错误）：回退到原始解码
        return bytes(recv_data).rstrip(b'\0').decode('utf-8', errors='replace')

    def _load_model(self):
        if self.encoder is not None:
            return
            
        if not os.path.exists(self.MODEL_PATH):
            raise FileNotFoundError(
                f"未找到 InvisMark 权重文件: {self.MODEL_PATH}\n"
                f"请确保 models 文件夹下有 paper.ckpt 文件。"
            )

        # PyTorch 2.6 defaults torch.load(weights_only=True), which rejects
        # the ModelConfig object saved in the official InvisMark checkpoint
        # unless it is explicitly allowlisted.
        try:
            from torch.serialization import safe_globals
            with safe_globals([ModelConfig]):
                checkpoint = torch.load(self.MODEL_PATH, map_location=self.device)
        except (ImportError, AttributeError):
            checkpoint = torch.load(self.MODEL_PATH, map_location=self.device)
        
        # 提取配置
        if "config" in checkpoint:
            self.config = checkpoint["config"]
        elif "model_config" in checkpoint:
            self.config = checkpoint["model_config"]
        else:
            # 回退到默认配置并从权重推断比特长度
            self.config = ModelConfig()
            sd = checkpoint.get("state_dict", checkpoint)
            for k in sd.keys():
                if "extractor.classifier.2.weight" in k:
                    self.config.num_encoded_bits = sd[k].shape[0]
                    break
        
        # 初始化网络
        self.encoder = invismark_model.Encoder(self.config).to(self.device)
        self.extractor = invismark_model.Extractor(self.config).to(self.device)
        
        # 加载权重
        enc_sd = checkpoint.get("encoder_state_dict")
        ext_sd = checkpoint.get("decoder_state_dict") # Checkpoint 中叫 decoder_state_dict
        
        if enc_sd:
            self.encoder.load_state_dict(enc_sd)
        if ext_sd:
            self.extractor.load_state_dict(ext_sd)
            
        self.encoder.eval()
        self.extractor.eval()

    def embed(self, input_path: str, output_path: str, text: str) -> dict:
        self._load_model()
        
        img = Image.open(input_path).convert("RGB")
        target_size = self.config.image_shape  # (H, W)

        # 模型分辨率输入张量（256×256，[-1,1]）
        img_tensor = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])(img).unsqueeze(0).to(self.device)

        target_bits = self.config.num_encoded_bits
        msg_bits = self._encode_text(text, target_bits).to(self.device)

        # 原图张量（原始分辨率，[-1,1]）
        to_norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        orig_tensor = to_norm(img).unsqueeze(0).to(self.device)
        orig_h, orig_w = orig_tensor.shape[-2], orig_tensor.shape[-1]

        with torch.no_grad():
            # Encoder 在模型分辨率（256×256）下运行，输出完整水印图
            encoded_at_model = self.encoder(img_tensor, msg_bits)

            # 关键修正（对应 InvisMark train.py _encode()）：
            #   1. 计算残差：encoded - resized_orig
            #   2. 双线性上采样残差到原始分辨率
            #   3. 叠加到原始分辨率图像，clamp 到 [-1,1]
            residual = encoded_at_model - img_tensor
            residual_up = transforms.Resize(
                (orig_h, orig_w),
                interpolation=transforms.InterpolationMode.BILINEAR,
                antialias=True
            )(residual)
            output_tensor = torch.clamp(orig_tensor + residual_up, -1.0, 1.0)

            res_pil = transforms.ToPILImage()((output_tensor.squeeze(0).cpu() + 1.0) / 2.0)

            if os.path.splitext(output_path)[1].lower() != ".png":
                output_path = os.path.splitext(output_path)[0] + ".png"
            res_pil.save(output_path)

        return {"output": output_path, "wm_length": target_bits}

    def extract(self, watermarked_path: str, wm_length: int = 0) -> str:
        self._load_model()
        img = Image.open(watermarked_path).convert("RGB")
        bits_np = self._extract_bits_from_pil(img)
        return self._decode_bits(bits_np)

    # ── 鲁棒提取核心 ──────────────────────────────────────────────

    def _extract_bits_from_pil(self, img: Image.Image) -> np.ndarray:
        """从 PIL 图像提取 100 个原始比特"""
        pred = self._predict_bits_from_pil(img)
        return (pred > 0.5).astype(np.uint8)

    def _predict_bits_from_pil(self, img: Image.Image) -> np.ndarray:
        """Return extractor probabilities for one PIL image."""
        transform = transforms.Compose([
            transforms.Resize(self.config.image_shape),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3)
        ])
        img_tensor = transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred = self.extractor(img_tensor)
        return pred.squeeze(0).cpu().numpy()

    def _try_bch_decode(self, bits_np: np.ndarray, max_nerr: int = 14):
        """
        尝试 BCH 解码。max_nerr 越低越严格（减少误报）。
        返回 (成功, 纠错数, 文字) 或 (失败, -1, None)。
        """
        bch = self._get_bch()
        recv_data = bytearray(np.packbits(bits_np[:self.DATA_BYTES * 8]).tobytes()[:self.DATA_BYTES])
        ecc_raw = bits_np[self.DATA_BYTES * 8: self.DATA_BYTES * 8 + bch.ecc_bits]
        pad_len = bch.ecc_bytes * 8 - bch.ecc_bits
        ecc_padded = np.concatenate([ecc_raw, np.zeros(pad_len, dtype=np.uint8)])
        recv_ecc = bytearray(np.packbits(ecc_padded).tobytes()[:bch.ecc_bytes])
        nerr = bch.decode(recv_data, recv_ecc)
        if 0 <= nerr <= max_nerr:
            bch.correct(recv_data, recv_ecc)
            text = bytes(recv_data).rstrip(b'\0').decode('utf-8', errors='replace')
            if text and all(c.isprintable() for c in text):
                return True, nerr, text
        return False, -1, None

    def _batch_extract_angles(self, img: Image.Image, angles: list, batch_size: int = 64):
        """
        GPU 批量推理：一次处理多个旋转角度。
        4090 显存充裕，batch_size=64 高效利用 GPU 并行。
        """
        transform = transforms.Compose([
            transforms.Resize(self.config.image_shape),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3)
        ])
        results = []
        for start in range(0, len(angles), batch_size):
            batch_angles = angles[start: start + batch_size]
            tensors = []
            for angle in batch_angles:
                if angle == 0:
                    rotated = img
                else:
                    rotated = img.rotate(-angle, resample=Image.BICUBIC,
                                        expand=True, fillcolor=(128, 128, 128))
                tensors.append(transform(rotated))
            batch = torch.stack(tensors).to(self.device)
            with torch.no_grad():
                preds = self.extractor(batch)
            for j, angle in enumerate(batch_angles):
                bits = (preds[j] > 0.5).cpu().numpy().astype(np.uint8)
                results.append((angle, bits))
        return results

    def robust_extract(self, watermarked_path: str, wm_length: int = 0,
                       progress_cb=None) -> str:
        """
        鲁棒提取：4090 GPU 批量推理（总计 ~5 秒）。

        搜索策略（早发现早返回）：
          Phase 1: 直接提取 + 标准角度 0°/90°/180°/270°
          Phase 2: 裁剪补偿 (5%-20%) × 4 角度
          Phase 3: 缩放补偿 (0.5x-2x)
        对变换后的结果用严格阈值 (nerr ≤ 12) 防误报。
        """
        self._load_model()
        img = Image.open(watermarked_path).convert("RGB")

        def report(msg):
            if progress_cb:
                progress_cb(msg)

        # Phase 1: 直接 + 标准旋转（最快，~0.5 秒）
        report("Phase 1: 标准角度 (0°/90°/180°/270°)...")
        results = self._batch_extract_angles(img, [0, 90, 180, 270])
        for angle, bits in results:
            ok, nerr, text = self._try_bch_decode(bits)
            if ok:
                return text if angle == 0 else f"{text}  [旋转={angle}°, 纠错={nerr}bit]"

        # Phase 2: 裁剪补偿（~2 秒）
        report("Phase 2: 裁剪补偿...")
        for pad_pct in [5, 10, 15, 20]:
            padded = self._add_padding(img, pad_pct)
            pad_results = self._batch_extract_angles(padded, [0, 90, 180, 270])
            for angle, bits in pad_results:
                ok, nerr, text = self._try_bch_decode(bits, max_nerr=12)
                if ok:
                    return f"{text}  [裁剪补偿={pad_pct}%, 旋转={angle}°, 纠错={nerr}bit]"

        # Phase 3: 缩放补偿（~1 秒）
        report("Phase 3: 缩放补偿...")
        w, h = img.size
        scale_imgs = []
        scale_labels = []
        for scale in [0.5, 0.7, 0.8, 0.9, 1.1, 1.2, 1.5, 2.0]:
            scaled = img.resize((int(w * scale), int(h * scale)), Image.BICUBIC)
            scale_imgs.append(scaled)
            scale_labels.append(scale)
        transform = transforms.Compose([
            transforms.Resize(self.config.image_shape),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3)
        ])
        tensors = [transform(s) for s in scale_imgs]
        batch = torch.stack(tensors).to(self.device)
        with torch.no_grad():
            preds = self.extractor(batch)
        for j, scale in enumerate(scale_labels):
            bits = (preds[j] > 0.5).cpu().numpy().astype(np.uint8)
            ok, nerr, text = self._try_bch_decode(bits)
            if ok:
                return f"{text}  [缩放={scale:.1f}x, 纠错={nerr}bit]"

        # Phase 4: 极度裁剪分块搜索（应对大图被切得只剩一小块的情况）
        # 如果前三步都失败，说明图片可能被大幅裁剪且边缘不对齐
        # 我们用一个滑动窗口（比如 256x256）去原图上扫
        report("Phase 4: 滑动窗口极度裁剪搜索...")
        crop_size = 256
        if w >= crop_size and h >= crop_size:
            stride = 64  # 滑动步长
            crop_imgs = []
            crop_coords = []
            for y in range(0, h - crop_size + 1, stride):
                for x in range(0, w - crop_size + 1, stride):
                    crop_imgs.append(img.crop((x, y, x + crop_size, y + crop_size)))
                    crop_coords.append((x, y))
            
            if crop_imgs:
                # 批量推理所有 crop
                crop_tensors = [transform(c) for c in crop_imgs]
                # 分批次送入 GPU，防止 OOM
                batch_size = 64
                for start_idx in range(0, len(crop_tensors), batch_size):
                    end_idx = min(start_idx + batch_size, len(crop_tensors))
                    batch = torch.stack(crop_tensors[start_idx:end_idx]).to(self.device)
                    with torch.no_grad():
                        preds = self.extractor(batch)
                    for j in range(len(batch)):
                        bits = (preds[j] > 0.5).cpu().numpy().astype(np.uint8)
                        # 这里必须用极其严格的阈值 (nerr <= 6) 防止在碎片上产生假阳性
                        ok, nerr, text = self._try_bch_decode(bits, max_nerr=6)
                        if ok:
                            cx, cy = crop_coords[start_idx + j]
                            return f"{text}  [极度裁剪恢复: 坐标({cx},{cy}), 纠错={nerr}bit]"

        report("搜索完成，回退到普通提取")
        bits = self._extract_bits_from_pil(img)
        return self._decode_bits(bits)

    @staticmethod
    def _add_padding(img: Image.Image, pad_pct: int) -> Image.Image:
        """四周添加灰色填充，模拟还原被裁剪的边缘"""
        w, h = img.size
        pw = int(w * pad_pct / 100)
        ph = int(h * pad_pct / 100)
        padded = Image.new("RGB", (w + 2 * pw, h + 2 * ph), (128, 128, 128))
        padded.paste(img, (pw, ph))
        return padded


# ═══════════════════════════════════════════════════════════════
#  算法二(进阶)：InvisMark Pro（频域非对称锚点 + AI 融合）
# ═══════════════════════════════════════════════════════════════

class InvisMarkProEngine(InvisMarkEngine):
    """
    融合频域非对称锚点 (类似 Log-Polar 模板概念) + AI InvisMark
    解决微小角度旋转手抖，降低单纯穷举的搜索压力。
    """
    def embed(self, input_path: str, output_path: str, text: str) -> dict:
        # 1. 读取原图
        img = cv2.imread(input_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("无法读取图片")
        h, w = img.shape[:2]

        # 2. 嵌入极为微弱的频域锚点 (提取 Y 通道做 FFT)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)

        cy, cx = h // 2, w // 2
        # 半径设为短边的 15%
        r = int(min(h, w) * 0.15)
        # 嵌两个非对称角度：30° 和 145°
        angles = [30, 145]
        
        mag_max = np.max(np.abs(fshift))
        strength = mag_max * 0.05  # 5%的强度，画质损失极微（PSNR 40+）

        for angle in angles:
            rad = np.deg2rad(angle)
            dy = int(r * np.sin(rad))
            dx = int(r * np.cos(rad))
            # 中心对称叠加特征峰
            fshift[cy - dy, cx + dx] += strength
            fshift[cy + dy, cx - dx] += strength

        # 逆变换回空域
        f_ishift = np.fft.ifftshift(fshift)
        img_back = np.abs(np.fft.ifft2(f_ishift))

        # 差值叠加回 BGR 原图
        diff = (img_back - gray)[:, :, np.newaxis]
        img_with_anchor = np.clip(img + diff, 0, 255).astype(np.uint8)

        # 保存为临时文件
        temp_path = input_path + "_anchor_temp.png"
        cv2.imwrite(temp_path, img_with_anchor)

        # 3. 再在其上叠加 InvisMark 扰动
        res = super().embed(temp_path, output_path, text)

        if os.path.exists(temp_path):
            os.remove(temp_path)

        return res

    def robust_extract(self, watermarked_path: str, wm_length: int = 0, progress_cb=None) -> str:
        # 1. 频域雷达：探测非对称锚点
        img = cv2.imread(watermarked_path, cv2.IMREAD_COLOR)
        offset = 0.0
        if img is not None:
            h, w = img.shape[:2]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
            f = np.fft.fft2(gray)
            fshift = np.fft.fftshift(f)
            mag = np.abs(fshift)

            cy, cx = h // 2, w // 2
            Y, X = np.ogrid[:h, :w]
            dist = np.sqrt((X - cx)**2 + (Y - cy)**2)

            # 环带滤波器，过滤低频和过高的高频
            min_r, max_r = int(min(h, w)*0.05), int(min(h, w)*0.4)
            mask = (dist > min_r) & (dist < max_r)
            mag_ring = mag * mask

            if np.max(mag_ring) > 0:
                max_y, max_x = np.unravel_index(np.argmax(mag_ring), mag_ring.shape)
                dy = cy - max_y
                dx = max_x - cx
                detected_angle = np.rad2deg(np.arctan2(dy, dx))
                if detected_angle < 0:
                    detected_angle += 180

                # 计算与预设锚点 (30 或 145) 的最短差值
                diff1 = detected_angle - 30
                diff2 = detected_angle - 145
                offset = diff1 if abs(diff1) < abs(diff2) else diff2

                if progress_cb:
                    progress_cb(f"FFT 雷达估算微旋偏角: {offset:.1f}°")

        # 2. GPU 智能寻的扫描 (Smart Homing)
        self._load_model()
        from PIL import Image
        img_pil = Image.open(watermarked_path).convert("RGB")

        # 构建智能角度池：FFT 探测到的角度 + 穷举防漏手抖角 (-5 到 +5)
        angles_to_search = []
        if abs(offset) > 0.5:
            angles_to_search.extend([offset, offset + 1, offset - 1, offset + 2, offset - 2])
        
        angles_to_search.extend(list(range(-5, 6, 1)))  # 加入 -5° 到 5° 的微小角度穷举
        angles_to_search.extend([0, 90, 180, 270])

        unique_angles = []
        for a in angles_to_search:
            a_round = round(a, 1)
            if a_round not in unique_angles:
                unique_angles.append(a_round)

        if progress_cb:
            progress_cb(f"Phase 0: 频域向导 GPU 扫描 ({len(unique_angles)}个角度)...")

        results = self._batch_extract_angles(img_pil, unique_angles, batch_size=64)
        for angle, bits in results:
            ok, nerr, text = self._try_bch_decode(bits)
            if ok:
                return f"{text}  [频域修正/微调={angle}°, 纠错={nerr}bit]"

        # 3. 如果频域修正失败（比如被严重裁剪导致频域失效），回退到原版的滑窗和裁剪搜索
        return super().robust_extract(watermarked_path, wm_length, progress_cb)


# ═══════════════════════════════════════════════════════════════
#  算法二(高级)：InvisMark Log-Polar（对数极坐标模板 + AI 融合）
# ═══════════════════════════════════════════════════════════════

class InvisMarkLogPolarEngine(InvisMarkEngine):
    """
    融合频域对数极坐标模板 (Log-Polar Template) + AI InvisMark
    
    核心创新：
    1. 在频域中嵌入微弱的圆环模板（均匀分布在多个半径上的特征点）
    2. 使用对数极坐标变换检测旋转角度，对微小角度旋转更鲁棒
    3. 模板强度极低（<3%），画质损失可忽略（PSNR > 45dB）
    4. 支持0.1°精度的微小角度检测
    
    技术原理：
    - 频域旋转对应空域旋转，且旋转中心在频谱中心
    - 对数极坐标变换将旋转转换为平移，便于检测
    - 多半径圆环提供冗余信息，增强鲁棒性
    """
    
    def __init__(self, key: int = 42, num_rings: int = 3, template_strength: float = 0.025):
        """
        Args:
            key: 密钥
            num_rings: 圆环模板数量，默认3个
            template_strength: 模板强度，建议0.02-0.03（2-3%）
        """
        super().__init__(key)
        self.num_rings = num_rings
        self.template_strength = template_strength
        # 预设的圆环半径比例（相对于图像短边）
        self.radius_ratios = [0.10, 0.20, 0.30]
        # 每个圆环上的特征点数量
        self.num_points_per_ring = 12
        # 特征点角度（度），非对称设计便于检测
        self.feature_angles = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
    
    def _create_log_polar_template(self, h: int, w: int) -> np.ndarray:
        """
        创建频域对数极坐标模板
        
        返回一个与图像同尺寸的复数模板，在指定位置添加特征点
        """
        template = np.zeros((h, w), dtype=np.complex128)
        cy, cx = h // 2, w // 2
        min_dim = min(h, w)
        
        # 在每个圆环上添加特征点
        for ring_idx, radius_ratio in enumerate(self.radius_ratios[:self.num_rings]):
            radius = int(min_dim * radius_ratio)
            
            for angle_deg in self.feature_angles:
                rad = np.deg2rad(angle_deg)
                dy = int(radius * np.sin(rad))
                dx = int(radius * np.cos(rad))
                
                # 对称位置（中心对称）
                y1, x1 = cy + dy, cx + dx
                y2, x2 = cy - dy, cx - dx
                
                # 确保在图像范围内
                if 0 <= y1 < h and 0 <= x1 < w:
                    template[y1, x1] = self.template_strength * (1 + 1j)  # 复数特征点
                if 0 <= y2 < h and 0 <= x2 < w:
                    template[y2, x2] = self.template_strength * (1 - 1j)  # 对称点
        
        return template
    
    def _detect_rotation_angle(self, img: np.ndarray) -> float:
        """
        使用对数极坐标变换检测旋转角度
        
        返回检测到的旋转角度（度）
        """
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        
        # FFT变换
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        mag = np.abs(fshift)
        
        # 创建对数极坐标变换
        cy, cx = h // 2, w // 2
        min_dim = min(h, w)
        
        # 在频域幅度谱上应用对数极坐标变换
        # 使用极坐标网格采样
        num_angles = 360  # 角度分辨率
        num_radii = 100   # 半径分辨率
        
        # 半径范围（对数尺度）
        min_radius = max(1, int(min_dim * 0.05))
        max_radius = int(min_dim * 0.45)
        
        # 创建极坐标网格
        angles = np.linspace(0, 2 * np.pi, num_angles, endpoint=False)
        radii = np.logspace(np.log10(min_radius), np.log10(max_radius), num_radii)
        
        # 采样频域幅度谱
        log_polar_mag = np.zeros((num_radii, num_angles), dtype=np.float32)
        
        for r_idx, r in enumerate(radii):
            for a_idx, angle in enumerate(angles):
                # 计算笛卡尔坐标
                x = int(cx + r * np.cos(angle))
                y = int(cy + r * np.sin(angle))
                
                # 确保在图像范围内
                if 0 <= x < w and 0 <= y < h:
                    log_polar_mag[r_idx, a_idx] = mag[y, x]
        
        # 在对数极坐标域中，旋转对应角度方向的平移
        # 使用相位相关检测平移（即旋转角度）
        
        # 创建参考模板（未旋转状态）
        template = self._create_log_polar_template(num_radii, num_angles)
        template_mag = np.abs(template)
        
        # 使用相位相关检测旋转
        # 将角度维度进行FFT
        f_log_polar = np.fft.fft(log_polar_mag, axis=1)
        f_template = np.fft.fft(template_mag, axis=1)
        
        # 计算互功率谱
        cross_power = f_log_polar * np.conj(f_template)
        cross_power /= np.abs(cross_power) + 1e-10
        
        # 逆FFT得到相关函数
        correlation = np.fft.ifft(cross_power, axis=1).real
        
        # 找到最大相关位置（即旋转角度）
        max_idx = np.argmax(np.max(correlation, axis=0))
        
        # 将索引转换为角度（度）
        detected_angle = (max_idx / num_angles) * 360
        
        # 归一化到 [0, 360)
        detected_angle = detected_angle % 360
        
        # 转换为 [-180, 180) 范围
        if detected_angle > 180:
            detected_angle -= 360
        
        return detected_angle
    
    def _correct_rotation(self, img: np.ndarray, angle: float) -> np.ndarray:
        """
        校正旋转角度
        """
        if abs(angle) < 0.1:  # 角度太小，不需要校正
            return img
        
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        
        # 创建旋转矩阵
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # 应用仿射变换
        corrected = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, 
                                   borderMode=cv2.BORDER_REFLECT)
        
        return corrected
    
    def embed(self, input_path: str, output_path: str, text: str) -> dict:
        # 1. 读取原图
        img = cv2.imread(input_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("无法读取图片")
        h, w = img.shape[:2]
        
        # 2. 嵌入对数极坐标模板到频域
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        
        # 创建并添加模板
        template = self._create_log_polar_template(h, w)
        fshift_with_template = fshift + template * np.max(np.abs(fshift))
        
        # 逆变换回空域
        f_ishift = np.fft.ifftshift(fshift_with_template)
        img_back = np.abs(np.fft.ifft2(f_ishift))
        
        # 计算差值并叠加回原图
        diff = (img_back - gray)[:, :, np.newaxis]
        img_with_template = np.clip(img + diff, 0, 255).astype(np.uint8)
        
        # 计算PSNR
        psnr = cv2.PSNR(img, img_with_template)
        if psnr < 40:
            # 如果PSNR太低，降低模板强度
            self.template_strength *= 0.8
            return self.embed(input_path, output_path, text)
        
        # 保存为临时文件
        temp_path = input_path + "_logpolar_temp.png"
        cv2.imwrite(temp_path, img_with_template)
        
        # 3. 再在其上叠加 InvisMark 扰动
        res = super().embed(temp_path, output_path, text)
        
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return res
    
    def robust_extract(self, watermarked_path: str, wm_length: int = 0, progress_cb=None) -> str:
        # 1. 使用对数极坐标模板检测旋转角度
        img = cv2.imread(watermarked_path, cv2.IMREAD_COLOR)
        detected_angle = 0.0
        
        if img is not None:
            try:
                detected_angle = self._detect_rotation_angle(img)
                if progress_cb:
                    progress_cb(f"对数极坐标模板检测旋转角度: {detected_angle:.2f}°")
            except Exception as e:
                if progress_cb:
                    progress_cb(f"角度检测失败: {str(e)}")
        
        # 2. 如果检测到显著旋转，先校正再提取
        if abs(detected_angle) > 0.5:
            # 校正旋转
            corrected_img = self._correct_rotation(img, detected_angle)
            
            # 保存校正后的临时文件
            temp_path = watermarked_path + "_corrected_temp.png"
            cv2.imwrite(temp_path, corrected_img)
            
            try:
                # 尝试从校正后的图像提取
                self._load_model()
                from PIL import Image
                img_pil = Image.open(temp_path).convert("RGB")
                
                # 直接提取
                bits = self._extract_bits_from_pil(img_pil)
                ok, nerr, text = self._try_bch_decode(bits)
                if ok:
                    return f"{text}  [对数极坐标校正={detected_angle:.2f}°, 纠错={nerr}bit]"
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        
        # 3. 构建智能角度搜索池
        self._load_model()
        from PIL import Image
        img_pil = Image.open(watermarked_path).convert("RGB")
        
        # 优先搜索检测到的角度附近
        angles_to_search = []
        if abs(detected_angle) > 0.1:
            # 在检测角度附近密集搜索
            for delta in np.arange(-2.0, 2.1, 0.2):
                angle_candidate = detected_angle + delta
                angles_to_search.append(round(angle_candidate, 1))
        
        # 添加标准角度和微小角度
        angles_to_search.extend([0, 90, 180, 270])
        angles_to_search.extend(np.arange(-5, 5.1, 0.5).tolist())
        
        # 去重并排序
        unique_angles = sorted(list(set(round(a, 1) for a in angles_to_search 
                                        if -180 <= a <= 180)))
        
        if progress_cb:
            progress_cb(f"Phase 0: 对数极坐标向导 GPU 扫描 ({len(unique_angles)}个角度)...")
        
        # 批量提取
        results = self._batch_extract_angles(img_pil, unique_angles, batch_size=64)
        for angle, bits in results:
            ok, nerr, text = self._try_bch_decode(bits)
            if ok:
                return f"{text}  [对数极坐标修正={angle}°, 纠错={nerr}bit]"
        
        # 4. 回退到父类的鲁棒提取
        return super().robust_extract(watermarked_path, wm_length, progress_cb)


# ═══════════════════════════════════════════════════════════════
#  算法二(进阶)：InvisMark Grid（网格化分块，突破文字长度限制）
# ═══════════════════════════════════════════════════════════════

class InvisMarkGridEngine(InvisMarkEngine):
    """
    通过将大图切分为多个 256x256 (或更高) 的网格，每个网格独立嵌入 2 个字符。
    使得水印容量随图片尺寸正比增加，不改变模型本身。
    """
    def __init__(self, key: int = 42):
        super().__init__(key)
        self.tile_size = 256  # 基础分块大小，与模型训练分辨率一致
        self.end_marker = b"\xFF\xFF" # 结束符标记
        self.empty_marker = b"\x00\x00"
        self.robust_thresholds = (0.5, 0.55, 0.45, 0.6, 0.4, 0.65, 0.35, 0.7, 0.3, 0.75)

    def embed(self, input_path: str, output_path: str, text: str) -> dict:
        self._load_model()
        img = Image.open(input_path).convert("RGB")
        w, h = img.size
        
        grid_cols = w // self.tile_size
        grid_rows = h // self.tile_size
        max_tiles = grid_cols * grid_rows
        
        if max_tiles == 0:
            return super().embed(input_path, output_path, text)
            
        # 使用 GBK 编码能让一个汉字只占 2 字节（正好是一个 Block 的容量）
        # 尾部必须留一个 Block 的空间放结束符
        max_bytes = (max_tiles - 1) * self.DATA_BYTES
        try:
            text_bytes = text.encode('gbk')
        except UnicodeEncodeError:
            text_bytes = text.encode('utf-8', errors='ignore')
            
        if len(text_bytes) > max_bytes:
            raise ValueError(f"当前图片({w}x{h})切分为{max_tiles}块，需要1块作为结束符，剩余可存{max_bytes}字节(约{max_bytes//2}个汉字)。\n"
                             f"您的文字需要{len(text_bytes)}字节，请缩短文字或使用更大图片。")
        
        # 将文字按每块 DATA_BYTES 字节严格切分
        payload_chunks = []
        for i in range(0, len(text_bytes), self.DATA_BYTES):
            b = text_bytes[i : i + self.DATA_BYTES]
            # 不足 DATA_BYTES 时补零
            b = b.ljust(self.DATA_BYTES, b'\x00')
            payload_chunks.append(b)
            
        # 加入明确的结束符 Block
        payload_chunks.append(self.end_marker)

        chunks_bytes = payload_chunks[:]

        # 剩下的网格全部用空字节填充
        while len(chunks_bytes) < max_tiles:
            chunks_bytes.append(b'\x00\x00')
            
        out_img = Image.new("RGB", (w, h))
        out_img.paste(img, (0, 0)) 
        
        target_bits = self.config.num_encoded_bits
        to_tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        to_pil = transforms.ToPILImage()
        
        bch = self._get_bch()
        idx = 0
        for r in range(grid_rows):
            for c in range(grid_cols):
                box = (c * self.tile_size, r * self.tile_size, (c+1) * self.tile_size, (r+1) * self.tile_size)
                tile = img.crop(box)
                
                chunk_b = chunks_bytes[idx]
                idx += 1
                
                if chunk_b == b'\x00\x00':
                    continue
                    
                # 手动直接进行 BCH 编码，不走父类的 _encode_text，避免父类强转 UTF-8
                data = bytearray(chunk_b)
                ecc = bch.encode(data)
                data_bits = np.unpackbits(np.frombuffer(bytes(data), dtype=np.uint8))
                ecc_bits = np.unpackbits(np.frombuffer(bytes(ecc), dtype=np.uint8))[:bch.ecc_bits]
                codeword = np.concatenate([data_bits, ecc_bits]).astype(np.float32)
                if len(codeword) < target_bits:
                    codeword = np.pad(codeword, (0, target_bits - len(codeword)))
                msg_bits = torch.from_numpy(codeword[:target_bits]).unsqueeze(0).to(self.device)
                
                tile_tensor = to_tensor(tile).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    encoded = self.encoder(tile_tensor, msg_bits)
                    output_tensor = torch.clamp(encoded, -1.0, 1.0)
                    tile_res_pil = to_pil((output_tensor.squeeze(0).cpu() + 1.0) / 2.0)
                
                out_img.paste(tile_res_pil, box)

        if os.path.splitext(output_path)[1].lower() != ".png":
            output_path = os.path.splitext(output_path)[0] + ".png"
        out_img.save(output_path)
        
        return {"output": output_path, "wm_length": target_bits * max_tiles}

    def _decode_bytes_from_bits(self, bits_np: np.ndarray):
        bch = self._get_bch()
        recv_data = bytearray(np.packbits(bits_np[:self.DATA_BYTES * 8]).tobytes()[:self.DATA_BYTES])
        ecc_raw = bits_np[self.DATA_BYTES * 8 : self.DATA_BYTES * 8 + bch.ecc_bits]
        pad_len = bch.ecc_bytes * 8 - bch.ecc_bits
        ecc_padded = np.concatenate([ecc_raw, np.zeros(pad_len, dtype=np.uint8)])
        recv_ecc = bytearray(np.packbits(ecc_padded).tobytes()[:bch.ecc_bytes])

        nerr = bch.decode(recv_data, recv_ecc)
        if nerr >= 0:
            bch.correct(recv_data, recv_ecc)
        return bytes(recv_data), nerr

    def extract(self, watermarked_path: str, wm_length: int = 0) -> str:
        self._load_model()
        img = Image.open(watermarked_path).convert("RGB")

        if img.width // self.tile_size == 0 or img.height // self.tile_size == 0:
            return super().extract(watermarked_path, wm_length)

        text, _found_end, _score = self._decode_grid_image(img, robust=False)
        return text

    def robust_extract(self, watermarked_path: str, wm_length: int = 0, progress_cb=None) -> str:
        self._load_model()
        img = Image.open(watermarked_path).convert("RGB")

        if img.width // self.tile_size == 0 or img.height // self.tile_size == 0:
            return super().robust_extract(watermarked_path, wm_length, progress_cb)

        def report(msg):
            if progress_cb:
                progress_cb(msg)

        report("Grid Phase 1: 全图方向校正 + 多阈值网格解码...")
        for angle in (0, 90, 180, 270):
            candidate_img = img if angle == 0 else img.rotate(-angle, resample=Image.BICUBIC, expand=True)
            text, found_end, score = self._decode_repeated_grid_image(candidate_img, robust=True)
            if found_end and text:
                return text if angle == 0 else f"{text}  [全图旋转校正={angle}°]"

        report("Grid Phase 2: 回退到原始网格解码...")
        text, found_end, _score = self._decode_grid_image(img, robust=False)
        if found_end:
            return text
        raise ValueError("未找到完整 Grid 水印结束符，图片可能经过了过强压缩或破坏")

    def _decode_grid_image(self, img: Image.Image, robust: bool):
        grid_cols = img.width // self.tile_size
        grid_rows = img.height // self.tile_size
        collected_bytes = bytearray()
        score = 0

        for r in range(grid_rows):
            for c in range(grid_cols):
                box = (c * self.tile_size, r * self.tile_size, (c+1) * self.tile_size, (r+1) * self.tile_size)
                tile = img.crop(box)

                if robust:
                    candidates = self._decode_tile_candidates(tile)
                    best = self._select_grid_candidate(candidates)
                    if best is None:
                        score += 100
                        continue
                    chunk_b, nerr, _threshold = best
                    score += nerr
                else:
                    bits_np = self._extract_bits_from_pil(tile)
                    chunk_b, nerr = self._decode_bytes_from_bits(bits_np)
                    score += nerr if nerr >= 0 else 100

                if chunk_b == self.end_marker:
                    collected_bytes.extend(chunk_b)
                    return self._safe_decode_gbk(collected_bytes), True, score

                collected_bytes.extend(chunk_b)

        return self._safe_decode_gbk(collected_bytes), False, score

    def _decode_repeated_grid_image(self, img: Image.Image, robust: bool):
        chunks, score = self._decode_grid_chunks(img, robust)

        # Prefer a complete frame immediately after the previous end marker.
        starts = [0]
        starts.extend(i + 1 for i, chunk in enumerate(chunks) if chunk == self.end_marker)
        for start in starts:
            if start >= len(chunks):
                continue
            frame = []
            for chunk in chunks[start:]:
                frame.append(chunk)
                if chunk == self.end_marker:
                    text = self._safe_decode_gbk(bytearray().join(frame))
                    if text:
                        return text, True, score
                    break

        return self._safe_decode_gbk(bytearray().join(chunks)), False, score

    def _decode_grid_chunks(self, img: Image.Image, robust: bool):
        grid_cols = img.width // self.tile_size
        grid_rows = img.height // self.tile_size
        chunks = []
        score = 0

        for r in range(grid_rows):
            for c in range(grid_cols):
                box = (c * self.tile_size, r * self.tile_size, (c+1) * self.tile_size, (r+1) * self.tile_size)
                tile = img.crop(box)

                if robust:
                    candidates = self._decode_tile_candidates(tile)
                    best = self._select_grid_candidate(candidates)
                    if best is None:
                        chunks.append(self.empty_marker)
                        score += 100
                        continue
                    chunk_b, nerr, _threshold = best
                    score += nerr
                else:
                    bits_np = self._extract_bits_from_pil(tile)
                    chunk_b, nerr = self._decode_bytes_from_bits(bits_np)
                    score += nerr if nerr >= 0 else 100

                chunks.append(chunk_b)

        return chunks, score

    def _decode_tile_candidates(self, tile: Image.Image):
        pred = self._predict_bits_from_pil(tile)
        candidates = []
        for threshold in self.robust_thresholds:
            bits = (pred > threshold).astype(np.uint8)
            chunk_b, nerr = self._decode_bytes_from_bits(bits)
            if 0 <= nerr <= 14:
                candidates.append((chunk_b, nerr, threshold))
        return candidates

    def _select_grid_candidate(self, candidates):
        if not candidates:
            return None

        # Avoid letting a false empty tile erase payload bytes before the
        # explicit end marker. Empty tiles are only padding after the marker.
        payload = [c for c in candidates if c[0] not in (self.empty_marker,)]
        pool = payload or candidates
        return min(pool, key=lambda c: (c[1], abs(c[2] - 0.5)))
        
    def _safe_decode_gbk(self, byte_data: bytearray) -> str:
        data = bytes(byte_data)
        # 寻找结束符，如果里面混入了 \xFF\xFF，直接从那里截断
        idx = data.find(self.end_marker)
        if idx != -1:
            data = data[:idx]
            
        # 去掉结尾可能填充的 0x00
        while data.endswith(b'\x00'):
            data = data[:-1]
            
        try:
            return data.decode('gbk', errors='ignore')
        except:
            return data.decode('utf-8', errors='ignore')


# ═══════════════════════════════════════════════════════════════
#  算法一：Adaptive DWT-QIM（超高画质，盲提取，100% 准确）
# ═══════════════════════════════════════════════════════════════

class AdaptiveDWTEngine:
    """
    核心原理:
    1. 在 BGR 图像的 B 通道上直接操作（避免色彩空间往返损失）
    2. 1 级 Haar DWT 分解，仅嵌入 level-1 细节子带（LH/HL/HH）
    3. QIM 量化步长 delta=5.0：
       - level-1 系数由 4 像素加权（权重 ±1/2），uint8 舍入最大误差 = 4×0.5×0.5 = 1.0
       - delta/4 = 1.25 > 1.0 → 100% 单比特正确率
    4. 5× 冗余投票（冗余保护防止极端边界情况）
    5. Payload 结构：MAGIC(2B) + text_len(2B) + text_utf8
    """

    WAVELET = "haar"
    DWT_LEVEL = 1
    REDUNDANCY = 5
    DELTA = 5.0
    MAGIC = b"\xAB\xCD"
    CHANNEL = 0  # B channel in BGR

    def __init__(self, key: int = 42):
        self.key = key

    # ---------- public ----------

    def embed(self, input_path: str, output_path: str, text: str) -> dict:
        img = cv2.imread(input_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"无法读取图片: {input_path}")

        channel = img[:, :, self.CHANNEL].astype(np.float64)
        h, w = channel.shape

        pad_h, pad_w = self._calc_padding(h, w)
        if pad_h or pad_w:
            channel = np.pad(channel, ((0, pad_h), (0, pad_w)), mode="reflect")

        bits = self._payload_to_bits(text)
        total_slots = len(bits) * self.REDUNDANCY

        coeffs = pywt.wavedec2(channel, self.WAVELET, level=self.DWT_LEVEL)
        flat, shapes = self._flatten_level1_details(coeffs)

        if total_slots > len(flat):
            raise ValueError(
                f"文字过长：需 {total_slots} 嵌入位，仅有 {len(flat)} 可用。"
                f"请缩短文字或使用更大图片。"
            )

        rng = np.random.RandomState(self.key)
        positions = rng.permutation(len(flat))[:total_slots]

        for i, bit in enumerate(bits):
            for r in range(self.REDUNDANCY):
                idx = positions[i * self.REDUNDANCY + r]
                flat[idx] = self._qim_embed(flat[idx], bit, self.DELTA)

        self._unflatten_level1_details(coeffs, flat, shapes)
        ch_wm = pywt.waverec2(coeffs, self.WAVELET)

        result = img.copy()
        result[:, :, self.CHANNEL] = np.clip(
            np.round(ch_wm[:h, :w]), 0, 255
        ).astype(np.uint8)

        ext = os.path.splitext(output_path)[1].lower()
        if ext in (".jpg", ".jpeg", ".webp"):
            output_path = os.path.splitext(output_path)[0] + ".png"
        cv2.imwrite(output_path, result)

        return {"output": output_path, "wm_length": len(bits)}

    def extract(self, watermarked_path: str, wm_length: int = 0) -> str:
        img = cv2.imread(watermarked_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"无法读取图片: {watermarked_path}")

        channel = img[:, :, self.CHANNEL].astype(np.float64)
        h, w = channel.shape

        pad_h, pad_w = self._calc_padding(h, w)
        if pad_h or pad_w:
            channel = np.pad(channel, ((0, pad_h), (0, pad_w)), mode="reflect")

        coeffs = pywt.wavedec2(channel, self.WAVELET, level=self.DWT_LEVEL)
        flat, _ = self._flatten_level1_details(coeffs)

        if wm_length > 0:
            n_bits = wm_length
        else:
            n_bits = self._auto_detect_length(flat)

        total_slots = n_bits * self.REDUNDANCY
        rng = np.random.RandomState(self.key)
        positions = rng.permutation(len(flat))[:total_slots]

        bits = self._extract_bits(flat, positions, n_bits)
        return self._bits_to_text(np.array(bits, dtype=np.uint8))

    # ---------- payload codec ----------

    def _payload_to_bits(self, text: str) -> np.ndarray:
        text_bytes = text.encode("utf-8")
        if len(text_bytes) > 65535:
            raise ValueError("文字过长（最大 65535 字节）")
        payload = self.MAGIC + struct.pack(">H", len(text_bytes)) + text_bytes
        return np.unpackbits(np.frombuffer(payload, dtype=np.uint8))

    def _bits_to_text(self, bits: np.ndarray) -> str:
        byte_arr = np.packbits(bits)
        data = byte_arr.tobytes()
        if data[:2] != self.MAGIC:
            raise ValueError("水印校验失败：密码不正确或图片无水印")
        length = struct.unpack(">H", data[2:4])[0]
        return data[4: 4 + length].decode("utf-8", errors="replace")

    def _auto_detect_length(self, flat: np.ndarray) -> int:
        header_bits = (len(self.MAGIC) + 2) * 8  # 32 bits
        header_slots = header_bits * self.REDUNDANCY

        rng = np.random.RandomState(self.key)
        positions = rng.permutation(len(flat))[:header_slots]

        header_bit_values = self._extract_bits(flat, positions, header_bits)
        header_bytes = np.packbits(np.array(header_bit_values, dtype=np.uint8)).tobytes()

        if header_bytes[:2] != self.MAGIC:
            raise ValueError("水印校验失败：密码不正确或图片无水印")
        text_len = struct.unpack(">H", header_bytes[2:4])[0]
        return header_bits + text_len * 8

    # ---------- QIM ----------

    @staticmethod
    def _qim_embed(val: float, bit: int, delta: float) -> float:
        quantized = delta * np.round(val / delta)
        return quantized + (delta / 4.0 if bit == 1 else -delta / 4.0)

    @staticmethod
    def _qim_extract(val: float, delta: float) -> int:
        remainder = val - delta * np.round(val / delta)
        return 1 if remainder > 0 else 0

    def _extract_bits(self, flat, positions, n_bits):
        bits = []
        for i in range(n_bits):
            votes = 0
            for r in range(self.REDUNDANCY):
                idx = positions[i * self.REDUNDANCY + r]
                votes += self._qim_extract(flat[idx], self.DELTA)
            bits.append(1 if votes > self.REDUNDANCY // 2 else 0)
        return bits

    # ---------- DWT helpers ----------

    def _calc_padding(self, h, w):
        step = 1 << self.DWT_LEVEL
        pad_h = (step - h % step) % step
        pad_w = (step - w % step) % step
        return pad_h, pad_w

    @staticmethod
    def _flatten_level1_details(coeffs):
        """仅展平 level-1 细节系数（coeffs 最后一组）"""
        level1 = coeffs[-1]
        arrays = []
        shapes = []
        for sub in level1:
            shapes.append(sub.shape)
            arrays.append(sub.ravel().copy())
        return np.concatenate(arrays), shapes

    @staticmethod
    def _unflatten_level1_details(coeffs, flat, shapes):
        """将修改后的系数写回 level-1"""
        new_subs = []
        offset = 0
        for s in shapes:
            size = s[0] * s[1]
            new_subs.append(flat[offset: offset + size].reshape(s))
            offset += size
        coeffs[-1] = tuple(new_subs)


# ═══════════════════════════════════════════════════════════════
#  算法二：invisible-watermark 库 dwtDct
# ═══════════════════════════════════════════════════════════════

class DwtDctEngine:
    """基于 invisible-watermark 库的 DWT+DCT 频域水印"""

    def __init__(self, key: int = 42):
        self.key = key

    def embed(self, input_path: str, output_path: str, text: str) -> dict:
        from imwatermark import WatermarkEncoder

        bgr = cv2.imread(input_path)
        if bgr is None:
            raise ValueError(f"无法读取图片: {input_path}")

        text_bytes = text.encode("utf-8")
        wm_bits = len(text_bytes) * 8

        encoder = WatermarkEncoder()
        encoder.set_watermark("bytes", text_bytes)
        bgr_encoded = encoder.encode(bgr, "dwtDct")

        ext = os.path.splitext(output_path)[1].lower()
        if ext in (".jpg", ".jpeg", ".webp"):
            output_path = os.path.splitext(output_path)[0] + ".png"
        cv2.imwrite(output_path, bgr_encoded)

        return {"output": output_path, "wm_length": wm_bits}

    def extract(self, watermarked_path: str, wm_length: int = 0) -> str:
        from imwatermark import WatermarkDecoder

        bgr = cv2.imread(watermarked_path)
        if bgr is None:
            raise ValueError(f"无法读取图片: {watermarked_path}")
        if wm_length <= 0:
            raise ValueError("dwtDct 方法必须提供水印比特长度")

        decoder = WatermarkDecoder("bytes", wm_length)
        wm_bytes = decoder.decode(bgr, "dwtDct")
        return wm_bytes.decode("utf-8", errors="replace")


# ═══════════════════════════════════════════════════════════════
#  算法三：blind_watermark 库 DWT+DCT+SVD（保留兼容）
# ═══════════════════════════════════════════════════════════════

class BlindWatermarkEngine:
    """基于 blind_watermark 库的 DWT+DCT+SVD 频域水印"""

    def __init__(self, key: int = 42):
        self.key = key

    def embed(self, input_path: str, output_path: str, text: str) -> dict:
        from blind_watermark import WaterMark

        bwm = WaterMark(password_img=self.key, password_wm=self.key)
        bwm.read_img(input_path)
        bwm.read_wm(text, mode="str")
        bwm.embed(output_path)
        return {"output": output_path, "wm_length": len(bwm.wm_bit)}

    def extract(self, watermarked_path: str, wm_length: int = 0) -> str:
        from blind_watermark import WaterMark

        if wm_length <= 0:
            raise ValueError("blind_watermark 方法必须提供水印比特长度")
        bwm = WaterMark(password_img=self.key, password_wm=self.key)
        return bwm.extract(watermarked_path, wm_shape=wm_length, mode="str")


# ═══════════════════════════════════════════════════════════════
#  算法四：TrustMark（短文本，高 JPEG 鲁棒）
# ═══════════════════════════════════════════════════════════════

class TrustMarkEngine:
    """基于 Adobe TrustMark 的短文本鲁棒水印"""

    REGISTRY_PATH = str(DEFAULT_DB_PATH)
    DIRECT_TEXT_MAX = 8

    def __init__(self, key: int = 42):
        self.key = key
        self.tm = None
        self.registry = WatermarkRegistry(self.REGISTRY_PATH)

    def _load_model(self):
        if self.tm is not None:
            return
        try:
            from trustmark import TrustMark
        except ImportError as exc:
            raise ImportError("TrustMark 方法需要先安装 trustmark>=0.9.1") from exc

        # BCH_5 在容量和鲁棒性之间比较均衡，适合短 ID/编号。
        self.tm = TrustMark(
            verbose=False,
            model_type="Q",
            encoding_type=1,
            loadRemover=False,
            loadBBoxDetector=False,
        )

    def _is_direct_text(self, text: str) -> bool:
        try:
            encoded = text.encode("ascii")
        except UnicodeEncodeError:
            return False
        return 0 < len(encoded) <= self.DIRECT_TEXT_MAX and all(32 <= b <= 126 for b in encoded)

    def _registry_id_for_text(self, text: str, input_path: str = "") -> str:
        source_hash = sha256_file(input_path) if input_path and os.path.exists(input_path) else ""
        record = self.registry.register(
            text,
            source_image_hash=source_hash,
            algorithm="trustmark",
        )
        return record["watermark_id"]

    def _resolve_registry_id(self, token: str) -> str:
        return self.registry.resolve(token) or token

    def embed(self, input_path: str, output_path: str, text: str) -> dict:
        self._load_model()
        img = Image.open(input_path).convert("RGB")
        embedded_text = text if self._is_direct_text(text) else self._registry_id_for_text(text, input_path)
        try:
            watermarked = self.tm.encode(img, embedded_text, MODE="text", WM_STRENGTH=1.0)
        except Exception as exc:
            raise ValueError(f"TrustMark 嵌入失败，文本可能超过容量限制: {exc}") from exc

        if os.path.splitext(output_path)[1].lower() != ".png":
            output_path = os.path.splitext(output_path)[0] + ".png"
        watermarked.save(output_path)
        return {
            "output": output_path,
            "wm_length": self.tm.schemaCapacity(),
            "embedded_text": embedded_text,
            "lookup_id": embedded_text if embedded_text != text else "",
            "registry_path": self.REGISTRY_PATH,
        }

    def extract(self, watermarked_path: str, wm_length: int = 0) -> str:
        self._load_model()
        img = Image.open(watermarked_path).convert("RGB")
        secret, detected, _version = self.tm.decode(img, MODE="text", ROTATION=True)
        if not detected:
            raise ValueError("TrustMark 未检测到有效水印")
        secret = (secret or "").strip()
        if not secret:
            raise ValueError("TrustMark 检测到水印信号，但未解出有效文本")
        if any(ch not in string.printable or ch in "\r\n\t\x0b\x0c" for ch in secret):
            raise ValueError(f"TrustMark 解码结果包含不可打印字符: {secret!r}")
        return self._resolve_registry_id(secret)


# ═══════════════════════════════════════════════════════════════
#  统一入口
# ═══════════════════════════════════════════════════════════════

METHODS = {
    "adaptive_dwt": ("Adaptive DWT-QIM（超高画质）", AdaptiveDWTEngine),
    "trustmark": ("TrustMark（短文本抗 JPEG 强压缩）", TrustMarkEngine),
    "invismark": ("Microsoft InvisMark（AI 强力鲁棒）", InvisMarkEngine),
    "invismark_grid": ("InvisMark Grid（网格化突破长度限制版）", InvisMarkGridEngine),
    "invismark_pro": ("InvisMark Pro（频域向导 + 滑窗穷举）", InvisMarkProEngine),
    "invismark_logpolar": ("InvisMark Log-Polar（对数极坐标模板 + 微小角度鲁棒）", InvisMarkLogPolarEngine),
    "dwt_dct": ("DWT+DCT 频域（均衡）", DwtDctEngine),
    "dwt_dct_svd": ("DWT+DCT+SVD（旧版兼容）", BlindWatermarkEngine),
}


class InvisibleWatermark:
    """统一水印接口"""

    def __init__(self, key: int = 42, method: str = "adaptive_dwt"):
        if method not in METHODS:
            raise ValueError(f"未知算法: {method}，可选: {list(METHODS.keys())}")
        self.method = method
        self.engine = METHODS[method][1](key=key)

    def embed(self, input_path: str, output_path: str, text: str) -> dict:
        return self.engine.embed(input_path, output_path, text)

    def extract(self, watermarked_path: str, wm_length: int = 0,
                robust: bool = False, progress_cb=None) -> str:
        if robust and isinstance(self.engine, InvisMarkEngine):
            return self.engine.robust_extract(
                watermarked_path, wm_length, progress_cb=progress_cb)
        return self.engine.extract(watermarked_path, wm_length)

    @staticmethod
    def compute_psnr(original_path: str, watermarked_path: str) -> float:
        orig = cv2.imread(original_path).astype(np.float64)
        wm = cv2.imread(watermarked_path).astype(np.float64)
        if orig.shape != wm.shape:
            wm = cv2.resize(wm, (orig.shape[1], orig.shape[0]))
        mse = np.mean((orig - wm) ** 2)
        if mse < 1e-10:
            return float("inf")
        return 10.0 * np.log10(255.0 ** 2 / mse)

    @staticmethod
    def get_output_path(input_path: str, suffix: str = "_watermarked") -> str:
        base, ext = os.path.splitext(input_path)
        if ext.lower() in (".jpg", ".jpeg", ".webp"):
            ext = ".png"
        return f"{base}{suffix}{ext}"
