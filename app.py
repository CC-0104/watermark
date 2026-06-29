import os
import json
import hashlib
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import torch

from watermark_engine import InvisibleWatermark, METHODS

CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".wm_config.json")


def load_config() -> dict:
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_config(data: dict):
    existing = load_config()
    existing.update(data)
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)


class WatermarkApp:
    PREVIEW_MAX = 420

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("图像隐形文字水印工具")
        self.root.geometry("980x740")
        self.root.minsize(880, 680)
        self.root.configure(bg="#f0f2f5")

        self.embed_src_path = None
        self.embed_out_path = None
        self.extract_src_path = None

        self._setup_styles()

        self.has_gpu = torch.cuda.is_available()
        self.gpu_name = torch.cuda.get_device_name(0) if self.has_gpu else "无"

        notebook = ttk.Notebook(root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=12, pady=(12, 0))

        self.embed_tab = ttk.Frame(notebook)
        self.extract_tab = ttk.Frame(notebook)
        notebook.add(self.embed_tab, text="  嵌入水印  ")
        notebook.add(self.extract_tab, text="  提取水印  ")

        self._build_embed_tab()
        self._build_extract_tab()

        self.status_var = tk.StringVar(value=f"就绪 | GPU: {self.gpu_name}" if self.has_gpu else "就绪 | CPU 模式")
        status_bar = ttk.Label(root, textvariable=self.status_var, style="Status.TLabel",
                               anchor=tk.W, relief=tk.SUNKEN, padding=(8, 4))
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)

    def _setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TNotebook", background="#f0f2f5")
        style.configure("TNotebook.Tab", padding=[22, 8], font=("Microsoft YaHei UI", 11))
        style.configure("TFrame", background="#f0f2f5")
        style.configure("TLabel", background="#f0f2f5", font=("Microsoft YaHei UI", 10))
        style.configure("TButton", font=("Microsoft YaHei UI", 10), padding=6)
        style.configure("Header.TLabel", font=("Microsoft YaHei UI", 13, "bold"), foreground="#1a73e8")
        style.configure("PSNR.TLabel", font=("Microsoft YaHei UI", 11, "bold"), foreground="#0d7c3d")
        style.configure("Status.TLabel", font=("Microsoft YaHei UI", 9), foreground="#555")
        style.configure("Accent.TButton", font=("Microsoft YaHei UI", 11, "bold"))
        style.configure("TLabelframe.Label", font=("Microsoft YaHei UI", 10))

    # ═══════════════════════════════════════════════
    #  嵌入水印 Tab
    # ═══════════════════════════════════════════════

    def _build_embed_tab(self):
        top = ttk.Frame(self.embed_tab)
        top.pack(fill=tk.X, padx=16, pady=(16, 4))
        ttk.Label(top, text="将文字水印隐形嵌入图片（肉眼不可见，可无损提取）",
                  style="Header.TLabel").pack(anchor=tk.W)

        ctrl = ttk.Frame(self.embed_tab)
        ctrl.pack(fill=tk.X, padx=16, pady=4)

        row = 0
        ttk.Button(ctrl, text="选择图片", command=self._embed_select_image).grid(
            row=row, column=0, sticky=tk.W, pady=2)
        self.embed_file_var = tk.StringVar(value="未选择")
        ttk.Label(ctrl, textvariable=self.embed_file_var, wraplength=600).grid(
            row=row, column=1, padx=8, sticky=tk.W)

        row += 1
        ttk.Label(ctrl, text="水印文字：").grid(row=row, column=0, sticky=tk.W, pady=(10, 0))
        self.embed_text_var = tk.StringVar()
        ttk.Entry(ctrl, textvariable=self.embed_text_var, width=48,
                  font=("Microsoft YaHei UI", 11)).grid(
            row=row, column=1, padx=8, pady=(10, 0), sticky=tk.W)

        row += 1
        ttk.Label(ctrl, text="算法：").grid(row=row, column=0, sticky=tk.W, pady=(8, 0))
        algo_frame = ttk.Frame(ctrl)
        algo_frame.grid(row=row, column=1, padx=8, pady=(8, 0), sticky=tk.W)

        self.embed_method_var = tk.StringVar(value="invismark_grid")
        method_names = [f"{k}  —  {v[0]}" for k, v in METHODS.items()]
        method_keys = list(METHODS.keys())
        self.embed_method_combo = ttk.Combobox(
            algo_frame, values=method_names, state="readonly", width=42,
            font=("Microsoft YaHei UI", 10))
        default_idx = method_keys.index("invismark_grid") if "invismark_grid" in method_keys else 0
        self.embed_method_combo.current(default_idx)
        self.embed_method_combo.pack(side=tk.LEFT)

        def on_method_change(_):
            idx = self.embed_method_combo.current()
            self.embed_method_var.set(method_keys[idx])
            self._sync_embed_password_row(method_keys[idx])
        self.embed_method_combo.bind("<<ComboboxSelected>>", on_method_change)

        row += 1
        self._embed_pwd_row = row
        self.embed_pwd_label = ttk.Label(ctrl, text="密码 (数字/字母/中文)：")
        self.embed_pwd_label.grid(row=row, column=0, sticky=tk.W, pady=(8, 0))
        self.embed_pwd_frame = ttk.Frame(ctrl)
        self.embed_pwd_frame.grid(row=row, column=1, padx=8, pady=(8, 0), sticky=tk.W)
        self.embed_pwd_var = tk.StringVar(value="42")
        ttk.Entry(self.embed_pwd_frame, textvariable=self.embed_pwd_var, width=16).pack(side=tk.LEFT)
        ttk.Label(self.embed_pwd_frame, text="  （提取时需相同密码）").pack(side=tk.LEFT)

        ctrl.columnconfigure(1, weight=1)
        self._sync_embed_password_row(self.embed_method_var.get())

        btn_frame = ttk.Frame(self.embed_tab)
        btn_frame.pack(fill=tk.X, padx=16, pady=8)
        self.embed_btn = ttk.Button(btn_frame, text="嵌入水印", style="Accent.TButton",
                                    command=self._do_embed, state=tk.DISABLED)
        self.embed_btn.pack(side=tk.LEFT)
        self.embed_save_btn = ttk.Button(btn_frame, text="另存为...",
                                         command=self._embed_save_as, state=tk.DISABLED)
        self.embed_save_btn.pack(side=tk.LEFT, padx=12)
        self.psnr_var = tk.StringVar(value="")
        ttk.Label(btn_frame, textvariable=self.psnr_var, style="PSNR.TLabel").pack(side=tk.LEFT, padx=16)

        preview_frame = ttk.Frame(self.embed_tab)
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=16, pady=(0, 12))

        left_pv = ttk.LabelFrame(preview_frame, text="原图预览")
        left_pv.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))
        self.embed_orig_canvas = tk.Canvas(left_pv, bg="#e8e8e8", highlightthickness=0)
        self.embed_orig_canvas.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        right_pv = ttk.LabelFrame(preview_frame, text="水印图预览")
        right_pv.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(6, 0))
        self.embed_result_canvas = tk.Canvas(right_pv, bg="#e8e8e8", highlightthickness=0)
        self.embed_result_canvas.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        self._embed_orig_photo = None
        self._embed_result_photo = None

    def _sync_embed_password_row(self, method: str):
        if method.startswith("invismark") or method == "trustmark":
            self.embed_pwd_label.grid_remove()
            self.embed_pwd_frame.grid_remove()
        else:
            self.embed_pwd_label.grid(row=self._embed_pwd_row, column=0, sticky=tk.W, pady=(8, 0))
            self.embed_pwd_frame.grid(row=self._embed_pwd_row, column=1, padx=8, pady=(8, 0), sticky=tk.W)

    def _sync_extract_password_row(self, method: str):
        if method.startswith("invismark") or method == "trustmark":
            self.extract_pwd_label.grid_remove()
            self.extract_pwd_entry.grid_remove()
        else:
            self.extract_pwd_label.grid(row=self._extract_pwd_row, column=0, sticky=tk.W, pady=(8, 0))
            self.extract_pwd_entry.grid(row=self._extract_pwd_row, column=1, padx=8, pady=(8, 0), sticky=tk.W)

    def _embed_select_image(self):
        path = filedialog.askopenfilename(
            title="选择要添加水印的图片",
            filetypes=[("图片文件", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif *.webp"), ("所有文件", "*.*")]
        )
        if not path:
            return
        self.embed_src_path = path
        self.embed_file_var.set(path)
        self.embed_btn.config(state=tk.NORMAL)
        self.embed_out_path = None
        self.embed_save_btn.config(state=tk.DISABLED)
        self.psnr_var.set("")
        self._show_preview(path, self.embed_orig_canvas, "_embed_orig_photo")
        self.embed_result_canvas.delete("all")
        self.status_var.set(f"已选择：{os.path.basename(path)}")

    def _do_embed(self):
        text = self.embed_text_var.get().strip()
        if not text:
            messagebox.showwarning("提示", "请输入水印文字")
            return
        if not self.embed_src_path:
            messagebox.showwarning("提示", "请先选择图片")
            return
        method = self.embed_method_var.get()
        if method.startswith("invismark") or method == "trustmark":
            pwd = 42
            pwd_str = None
        else:
            try:
                pwd_str = self.embed_pwd_var.get().strip() or "42"
                pwd = int(hashlib.md5(pwd_str.encode("utf-8")).hexdigest()[:8], 16)
            except Exception:
                messagebox.showwarning("提示", "密码处理错误")
                return
        if method in ("invismark", "invismark_pro", "invismark_logpolar") and len(text.encode('utf-8')) > 2:
            messagebox.showwarning(
                "InvisMark 容量限制",
                f"该模式 (BCH纠错) 最多支持 2 个 ASCII 字符。\n"
                f"当前文字 \"{text}\" 为 {len(text.encode('utf-8'))} 字节，超出限制。\n\n"
                f"更长文字请改用 InvisMark Grid 或 adaptive_dwt 算法。"
            )
            return
        self.embed_btn.config(state=tk.DISABLED)
        self.psnr_var.set("")
        self.status_var.set(f"正在用 {METHODS[method][0]} 嵌入水印，请稍候...")
        self.root.update_idletasks()

        def task():
            try:
                wm = InvisibleWatermark(key=pwd, method=method)
                out = InvisibleWatermark.get_output_path(self.embed_src_path)
                result = wm.embed(self.embed_src_path, out, text)
                psnr = InvisibleWatermark.compute_psnr(self.embed_src_path, result["output"])
                result["psnr"] = psnr
                result["method"] = method
                self.embed_out_path = result["output"]
                cfg_upd = {
                    "last_wm_length": result["wm_length"],
                    "last_method": method,
                }
                if pwd_str is not None:
                    cfg_upd["last_password"] = pwd_str
                save_config(cfg_upd)
                self.root.after(0, lambda: self._embed_done(result))
            except Exception as e:
                self.root.after(0, lambda: self._embed_error(str(e)))

        threading.Thread(target=task, daemon=True).start()

    def _embed_done(self, result):
        self.embed_btn.config(state=tk.NORMAL)
        self.embed_save_btn.config(state=tk.NORMAL)
        self._show_preview(self.embed_out_path, self.embed_result_canvas, "_embed_result_photo")

        psnr = result["psnr"]
        psnr_str = f"PSNR = {psnr:.2f} dB" if psnr != float("inf") else "PSNR = ∞ (无损)"
        quality = "完美" if psnr > 50 else ("优秀" if psnr > 45 else ("良好" if psnr > 40 else "一般"))
        self.psnr_var.set(f"{psnr_str}  [{quality}]")

        self.status_var.set(
            f"水印嵌入成功！{psnr_str}  |  比特长度={result['wm_length']}  |  {os.path.basename(result['output'])}"
        )
        req = ["相同算法"]
        if not result["method"].startswith("invismark") and result["method"] != "trustmark":
            req.insert(0, "相同密码")
        if result["method"] not in ("adaptive_dwt", "trustmark"):
            req.append("此比特长度")
        messagebox.showinfo("成功", (
            f"隐形水印已嵌入！\n\n"
            f"算法：{METHODS[result['method']][0]}\n"
            f"画质：{psnr_str}  [{quality}]\n"
            f"输出文件：{result['output']}\n"
            f"水印比特长度：{result['wm_length']}\n\n"
            f"提取时需要：" + " + ".join(req)
        ))

    def _embed_error(self, msg):
        self.embed_btn.config(state=tk.NORMAL)
        self.status_var.set("嵌入失败")
        messagebox.showerror("错误", f"嵌入水印失败：\n{msg}")

    def _embed_save_as(self):
        if not self.embed_out_path or not os.path.exists(self.embed_out_path):
            return
        ext = os.path.splitext(self.embed_out_path)[1]
        path = filedialog.asksaveasfilename(
            title="另存水印图片",
            defaultextension=ext,
            filetypes=[("PNG", "*.png"), ("BMP", "*.bmp"), ("TIFF", "*.tiff"), ("所有文件", "*.*")]
        )
        if path:
            import shutil
            shutil.copy2(self.embed_out_path, path)
            self.status_var.set(f"已保存到：{path}")

    # ═══════════════════════════════════════════════
    #  提取水印 Tab
    # ═══════════════════════════════════════════════

    def _build_extract_tab(self):
        top = ttk.Frame(self.extract_tab)
        top.pack(fill=tk.X, padx=16, pady=(16, 4))
        ttk.Label(top, text="从水印图片中提取隐藏的文字（无需原图）",
                  style="Header.TLabel").pack(anchor=tk.W)

        ctrl = ttk.Frame(self.extract_tab)
        ctrl.pack(fill=tk.X, padx=16, pady=4)
        cfg = load_config()

        row = 0
        ttk.Button(ctrl, text="选择水印图片", command=self._extract_select_image).grid(
            row=row, column=0, sticky=tk.W, pady=2)
        self.extract_file_var = tk.StringVar(value="未选择")
        ttk.Label(ctrl, textvariable=self.extract_file_var, wraplength=600).grid(
            row=row, column=1, padx=8, sticky=tk.W)

        row += 1
        ttk.Label(ctrl, text="算法：").grid(row=row, column=0, sticky=tk.W, pady=(10, 0))
        method_names = [f"{k}  —  {v[0]}" for k, v in METHODS.items()]
        method_keys = list(METHODS.keys())
        self.extract_method_var = tk.StringVar(value=cfg.get("last_method", "invismark_grid"))
        self.extract_method_combo = ttk.Combobox(
            ctrl, values=method_names, state="readonly", width=42,
            font=("Microsoft YaHei UI", 10))
        last_idx = method_keys.index(self.extract_method_var.get()) if self.extract_method_var.get() in method_keys else 0
        self.extract_method_combo.current(last_idx)
        self.extract_method_combo.grid(row=row, column=1, padx=8, pady=(10, 0), sticky=tk.W)

        def on_ext_method_change(_):
            idx = self.extract_method_combo.current()
            m = method_keys[idx]
            self.extract_method_var.set(m)
            needs_len = m not in ("adaptive_dwt", "trustmark") and not m.startswith("invismark")
            self.extract_len_entry.config(state=tk.NORMAL if needs_len else tk.DISABLED)
            self._sync_extract_password_row(m)
        self.extract_method_combo.bind("<<ComboboxSelected>>", on_ext_method_change)

        row += 1
        ttk.Label(ctrl, text="水印比特长度：").grid(row=row, column=0, sticky=tk.W, pady=(8, 0))
        len_frame = ttk.Frame(ctrl)
        len_frame.grid(row=row, column=1, padx=8, pady=(8, 0), sticky=tk.W)
        self.extract_len_var = tk.StringVar(value=str(cfg.get("last_wm_length", "")))
        self.extract_len_entry = ttk.Entry(len_frame, textvariable=self.extract_len_var, width=12)
        self.extract_len_entry.pack(side=tk.LEFT)
        ttk.Label(len_frame, text="  （仅传统频域算法需填写）").pack(side=tk.LEFT)
        m = self.extract_method_var.get()
        if m in ("adaptive_dwt", "trustmark") or m.startswith("invismark"):
            self.extract_len_entry.config(state=tk.DISABLED)

        row += 1
        self._extract_pwd_row = row
        self.extract_pwd_label = ttk.Label(ctrl, text="密码 (数字/字母/中文)：")
        self.extract_pwd_label.grid(row=row, column=0, sticky=tk.W, pady=(8, 0))
        self.extract_pwd_var = tk.StringVar(value=str(cfg.get("last_password", "42")))
        self.extract_pwd_entry = ttk.Entry(ctrl, textvariable=self.extract_pwd_var, width=16)
        self.extract_pwd_entry.grid(row=row, column=1, padx=8, pady=(8, 0), sticky=tk.W)
        self._sync_extract_password_row(self.extract_method_var.get())

        row += 1
        self.robust_var = tk.BooleanVar(value=True)
        robust_frame = ttk.Frame(ctrl)
        robust_frame.grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=(8, 0))
        ttk.Checkbutton(robust_frame, text="鲁棒提取（抗旋转/裁剪/缩放，适用于 InvisMark / TrustMark）",
                        variable=self.robust_var).pack(side=tk.LEFT)

        ctrl.columnconfigure(1, weight=1)

        btn_frame = ttk.Frame(self.extract_tab)
        btn_frame.pack(fill=tk.X, padx=16, pady=8)
        self.extract_btn = ttk.Button(btn_frame, text="提取水印", style="Accent.TButton",
                                      command=self._do_extract, state=tk.DISABLED)
        self.extract_btn.pack(side=tk.LEFT)

        mid = ttk.Frame(self.extract_tab)
        mid.pack(fill=tk.BOTH, expand=True, padx=16, pady=(0, 12))

        pv_frame = ttk.LabelFrame(mid, text="水印图预览")
        pv_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))
        self.extract_canvas = tk.Canvas(pv_frame, bg="#e8e8e8", highlightthickness=0)
        self.extract_canvas.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self._extract_photo = None

        result_frame = ttk.LabelFrame(mid, text="提取结果")
        result_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(6, 0))
        self.extract_result_text = tk.Text(
            result_frame, wrap=tk.WORD, font=("Microsoft YaHei UI", 16),
            bg="#fff", relief=tk.FLAT, padx=16, pady=16)
        self.extract_result_text.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

    def _extract_select_image(self):
        path = filedialog.askopenfilename(
            title="选择要提取水印的图片",
            filetypes=[("图片文件", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif *.webp"), ("所有文件", "*.*")]
        )
        if not path:
            return
        self.extract_src_path = path
        self.extract_file_var.set(path)
        self.extract_btn.config(state=tk.NORMAL)
        self._show_preview(path, self.extract_canvas, "_extract_photo")
        self.extract_result_text.delete("1.0", tk.END)
        self.status_var.set(f"已选择：{os.path.basename(path)}")

    def _do_extract(self):
        if not self.extract_src_path:
            messagebox.showwarning("提示", "请先选择水印图片")
            return
        method = self.extract_method_var.get()
        if method.startswith("invismark") or method == "trustmark":
            pwd = 42
            wm_len = 0
        else:
            try:
                pwd_str = self.extract_pwd_var.get().strip() or "42"
                pwd = int(hashlib.md5(pwd_str.encode("utf-8")).hexdigest()[:8], 16)
            except Exception:
                messagebox.showwarning("提示", "密码处理错误")
                return

        wm_len = 0
        if method not in ("adaptive_dwt", "trustmark") and not method.startswith("invismark"):
            try:
                wm_len = int(self.extract_len_var.get().strip())
            except ValueError:
                messagebox.showwarning("提示", f"{method} 算法需要填写水印比特长度（嵌入时返回的数字）")
                return

        self.extract_btn.config(state=tk.DISABLED)
        self.status_var.set("正在提取水印，请稍候...")
        self.root.update_idletasks()

        use_robust = self.robust_var.get() and (method.startswith("invismark") or method == "trustmark")

        def progress_cb(msg):
            self.root.after(0, lambda: self.status_var.set(f"鲁棒搜索: {msg}"))

        def task():
            try:
                wm = InvisibleWatermark(key=pwd, method=method)
                text = wm.extract(self.extract_src_path, wm_len,
                                  robust=use_robust, progress_cb=progress_cb)
                self.root.after(0, lambda: self._extract_done(text))
            except Exception as e:
                self.root.after(0, lambda: self._extract_error(str(e)))

        threading.Thread(target=task, daemon=True).start()

    def _extract_done(self, text):
        self.extract_btn.config(state=tk.NORMAL)
        self.extract_result_text.delete("1.0", tk.END)
        self.extract_result_text.insert("1.0", text)
        self.status_var.set("水印提取成功！")

    def _extract_error(self, msg):
        self.extract_btn.config(state=tk.NORMAL)
        self.status_var.set("提取失败")
        messagebox.showerror("错误", f"提取水印失败：\n{msg}")

    # ═══════════════════════════════════════════════
    #  预览工具
    # ═══════════════════════════════════════════════

    def _show_preview(self, path: str, canvas: tk.Canvas, photo_attr: str):
        try:
            img = Image.open(path)
            canvas.update_idletasks()
            cw = max(canvas.winfo_width(), 200)
            ch = max(canvas.winfo_height(), 200)
            max_w = min(cw - 8, self.PREVIEW_MAX)
            max_h = min(ch - 8, self.PREVIEW_MAX)
            img.thumbnail((max_w, max_h), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            setattr(self, photo_attr, photo)
            canvas.delete("all")
            canvas.create_image(cw // 2, ch // 2, anchor=tk.CENTER, image=photo)
        except Exception as e:
            canvas.delete("all")
            canvas.create_text(100, 80, text=f"无法预览: {e}", fill="red")


def main():
    root = tk.Tk()
    WatermarkApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
