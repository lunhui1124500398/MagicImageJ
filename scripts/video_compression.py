#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
视频压缩工具 (Video Compression Tool)
带 GUI 界面，使用 FFmpeg 对视频进行压缩。

功能:
- 选择输入视频文件
- 配置压缩参数 (CRF, Preset)
- 实时显示压缩进度和结果

依赖: FFmpeg (需要在系统 PATH 中)
"""

import sys
import os
import subprocess
import threading
from pathlib import Path

# === GUI with tkinter ===
try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    HAS_TK = True
except ImportError:
    HAS_TK = False


def check_ffmpeg():
    """检查 FFmpeg 是否可用"""
    try:
        startupinfo = None
        if os.name == 'nt':
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        result = subprocess.run(
            ['ffmpeg', '-version'],
            capture_output=True,
            startupinfo=startupinfo
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def get_file_size_mb(path):
    """获取文件大小 (MB)"""
    return os.path.getsize(path) / (1024 * 1024)


class VideoCompressorGUI:
    """视频压缩器 GUI"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("视频压缩工具 - FFmpeg")
        self.root.geometry("550x520")  # 增大窗口
        self.root.resizable(True, True)
        
        # 变量
        self.input_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.crf_value = tk.IntVar(value=28)
        self.preset_value = tk.StringVar(value="medium")
        self.is_running = False
        self.video_duration = 0  # 视频总时长(秒)
        
        self._setup_ui()
        self._check_ffmpeg()
    
    def _setup_ui(self):
        """构建 UI"""
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # === 1. 输入文件 ===
        input_frame = ttk.LabelFrame(main_frame, text="输入文件", padding="5")
        input_frame.pack(fill=tk.X, pady=5)
        
        ttk.Entry(input_frame, textvariable=self.input_path, width=50).pack(side=tk.LEFT, padx=5)
        ttk.Button(input_frame, text="浏览...", command=self._browse_input).pack(side=tk.LEFT)
        
        # === 2. 输出文件 ===
        output_frame = ttk.LabelFrame(main_frame, text="输出文件 (留空自动生成)", padding="5")
        output_frame.pack(fill=tk.X, pady=5)
        
        ttk.Entry(output_frame, textvariable=self.output_path, width=50).pack(side=tk.LEFT, padx=5)
        ttk.Button(output_frame, text="浏览...", command=self._browse_output).pack(side=tk.LEFT)
        
        # === 3. 压缩参数 ===
        params_frame = ttk.LabelFrame(main_frame, text="压缩参数", padding="10")
        params_frame.pack(fill=tk.X, pady=5)
        
        # CRF 滑块
        crf_frame = ttk.Frame(params_frame)
        crf_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(crf_frame, text="质量 (CRF):").pack(side=tk.LEFT)
        ttk.Label(crf_frame, text="高质量").pack(side=tk.LEFT, padx=(10, 0))
        
        self.crf_scale = ttk.Scale(crf_frame, from_=18, to=35, variable=self.crf_value, 
                                    orient=tk.HORIZONTAL, length=200, command=self._update_crf_label)
        self.crf_scale.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(crf_frame, text="小文件").pack(side=tk.LEFT)
        
        self.crf_label = ttk.Label(crf_frame, text="28", width=3)
        self.crf_label.pack(side=tk.LEFT, padx=10)
        
        # Preset 下拉框
        preset_frame = ttk.Frame(params_frame)
        preset_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(preset_frame, text="编码速度:").pack(side=tk.LEFT)
        preset_combo = ttk.Combobox(preset_frame, textvariable=self.preset_value, width=15,
                                     values=["ultrafast", "fast", "medium", "slow", "veryslow"],
                                     state="readonly")
        preset_combo.pack(side=tk.LEFT, padx=10)
        ttk.Label(preset_frame, text="(慢 = 更好压缩)").pack(side=tk.LEFT)
        
        # === 4. 进度条 ===
        progress_frame = ttk.Frame(main_frame)
        progress_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(progress_frame, text="进度:").pack(side=tk.LEFT)
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, 
                                             maximum=100, length=300, mode='determinate')
        self.progress_bar.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.progress_label = ttk.Label(progress_frame, text="0%", width=6)
        self.progress_label.pack(side=tk.LEFT)
        
        # === 5. 开始按钮 ===
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=8)
        
        # 使用 ttk.Button 配合样式
        style = ttk.Style()
        style.configure('Start.TButton', font=('Microsoft YaHei', 10))
        style.configure('Exit.TButton', font=('Microsoft YaHei', 9))
        
        self.start_btn = ttk.Button(btn_frame, text="▶ 开始压缩", command=self._start_compress,
                                     style='Start.TButton', width=15)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(btn_frame, text="退出", command=self.root.quit,
                   style='Exit.TButton', width=8).pack(side=tk.RIGHT, padx=5)
        
        # === 6. 进度和日志 ===
        log_frame = ttk.LabelFrame(main_frame, text="输出日志", padding="5")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # 创建带滚动条的文本框
        log_container = ttk.Frame(log_frame)
        log_container.pack(fill=tk.BOTH, expand=True)
        
        self.log_text = tk.Text(log_container, height=8, state=tk.DISABLED, wrap=tk.WORD,
                                 bg="#1e1e1e", fg="#d4d4d4", font=("Consolas", 9))
        
        # 添加滚动条
        scrollbar = ttk.Scrollbar(log_container, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # 状态栏
        self.status_var = tk.StringVar(value="就绪 - 请选择视频文件后点击「开始压缩」")
        ttk.Label(main_frame, textvariable=self.status_var).pack(side=tk.BOTTOM, anchor=tk.W)
    
    def _update_crf_label(self, value):
        """更新 CRF 显示值"""
        self.crf_label.config(text=str(int(float(value))))
    
    def _get_video_duration(self, video_path):
        """获取视频时长 (秒)"""
        try:
            startupinfo = None
            if os.name == 'nt':
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            
            result = subprocess.run(
                ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                 '-of', 'default=noprint_wrappers=1:nokey=1', video_path],
                capture_output=True, startupinfo=startupinfo
            )
            duration = float(result.stdout.decode('utf-8', errors='replace').strip())
            return duration
        except:
            return 0
    
    def _update_progress(self, value):
        """更新进度条 (线程安全)"""
        def update():
            self.progress_var.set(value)
            self.progress_label.config(text=f"{int(value)}%")
        self.root.after(0, update)
    
    def _check_ffmpeg(self):
        """检查 FFmpeg"""
        if check_ffmpeg():
            self._log("[OK] FFmpeg 已就绪")
        else:
            self._log("[Error] 未找到 FFmpeg!")
            self._log("请下载并安装: https://ffmpeg.org/download.html")
            self.start_btn.config(state=tk.DISABLED)
    
    def _browse_input(self):
        """浏览输入文件"""
        path = filedialog.askopenfilename(
            title="选择要压缩的视频",
            filetypes=[
                ("视频文件", "*.mp4 *.avi *.mkv *.mov *.wmv *.webm"),
                ("所有文件", "*.*")
            ]
        )
        if path:
            self.input_path.set(path)
            # 自动生成输出路径
            p = Path(path)
            self.output_path.set(str(p.parent / f"{p.stem}_compressed.mp4"))
    
    def _browse_output(self):
        """浏览输出文件"""
        path = filedialog.asksaveasfilename(
            title="保存压缩后的视频",
            defaultextension=".mp4",
            filetypes=[("MP4 视频", "*.mp4"), ("所有文件", "*.*")]
        )
        if path:
            self.output_path.set(path)
    
    def _log(self, message):
        """添加日志"""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)
    
    def _start_compress(self):
        """开始压缩"""
        if self.is_running:
            return
        
        input_path = self.input_path.get().strip()
        if not input_path or not Path(input_path).exists():
            messagebox.showerror("错误", "请选择有效的输入文件!")
            return
        
        output_path = self.output_path.get().strip()
        if not output_path:
            p = Path(input_path)
            output_path = str(p.parent / f"{p.stem}_compressed.mp4")
            self.output_path.set(output_path)
        
        # 在后台线程执行
        self.is_running = True
        self.start_btn.config(state=tk.DISABLED)
        self.status_var.set("正在压缩...")
        
        thread = threading.Thread(target=self._compress_thread, 
                                   args=(input_path, output_path), daemon=True)
        thread.start()
    
    def _compress_thread(self, input_path, output_path):
        """压缩线程"""
        stderr_output = []
        
        def stderr_reader(proc):
            """在单独线程中读取 stderr 避免缓冲区死锁"""
            try:
                for line in proc.stderr:
                    stderr_output.append(line)
            except:
                pass
        
        try:
            crf = self.crf_value.get()
            preset = self.preset_value.get()
            
            self._log("-" * 40)
            self._log(f"[Input] {input_path}")
            self._log(f"[Size] 原始: {get_file_size_mb(input_path):.2f} MB")
            self._log(f"[Config] CRF={crf}, Preset={preset}")
            self._log("[Processing] 正在压缩...")
            
            # 获取视频时长
            self.video_duration = self._get_video_duration(input_path)
            self._log(f"[Duration] 视频时长: {self.video_duration:.1f} 秒")
            
            # FFmpeg 命令 (使用 -progress 获取进度)
            cmd = [
                'ffmpeg',
                '-y',
                '-i', input_path,
                '-vcodec', 'libx264',
                '-crf', str(crf),
                '-preset', preset,
                '-acodec', 'aac',
                '-b:a', '128k',
                '-progress', 'pipe:1',
                '-nostats',
                output_path
            ]
            
            # 执行
            startupinfo = None
            if os.name == 'nt':
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                startupinfo=startupinfo
            )
            
            # 启动 stderr 读取线程 (防止缓冲区填满导致死锁)
            stderr_thread = threading.Thread(target=stderr_reader, args=(process,), daemon=True)
            stderr_thread.start()
            
            # 主线程读取 stdout 获取进度
            last_progress = 0
            for line in process.stdout:
                line_str = line.decode('utf-8', errors='replace').strip()
                
                # 解析 out_time (时间格式 HH:MM:SS.ms)
                if line_str.startswith('out_time='):
                    try:
                        time_str = line_str.split('=')[1]
                        if ':' in time_str and time_str != 'N/A':
                            parts = time_str.split(':')
                            time_sec = float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
                            if self.video_duration > 0:
                                progress = min(99, (time_sec / self.video_duration) * 100)
                                if progress > last_progress + 0.5:  # 只有变化超过0.5%才更新
                                    last_progress = progress
                                    self._update_progress(progress)
                    except:
                        pass
            
            # 等待进程和线程完成
            process.wait()
            stderr_thread.join(timeout=2)
            
            if process.returncode == 0 and Path(output_path).exists():
                self._update_progress(100)
                new_size = get_file_size_mb(output_path)
                old_size = get_file_size_mb(input_path)
                ratio = (1 - new_size / old_size) * 100
                
                self._log("-" * 40)
                self._log(f"[Done] 压缩完成!")
                self._log(f"[Size] 新大小: {new_size:.2f} MB")
                self._log(f"[Ratio] 压缩率: {ratio:.1f}%")
                self._log(f"[Saved] {output_path}")
                
                self.root.after(0, lambda: self.status_var.set("压缩完成!"))
                self.root.after(0, lambda: messagebox.showinfo("完成", 
                    f"压缩完成!\n\n原始: {old_size:.2f} MB\n压缩后: {new_size:.2f} MB\n压缩率: {ratio:.1f}%"))
            else:
                # 合并 stderr 输出
                error_bytes = b''.join(stderr_output)
                error_msg = error_bytes.decode('utf-8', errors='replace') if error_bytes else "未知错误"
                self._log(f"[Error] FFmpeg 错误 (返回码: {process.returncode})")
                # 显示最后500个字符的错误信息
                self._log(error_msg[-500:] if len(error_msg) > 500 else error_msg)
                self.root.after(0, lambda: self.status_var.set("压缩失败"))
                
        except Exception as e:
            self._log(f"[Error] {str(e)}")
            import traceback
            self._log(traceback.format_exc()[:300])
            self.root.after(0, lambda: self.status_var.set("发生错误"))
        finally:
            self.is_running = False
            self.root.after(0, lambda: self.start_btn.config(state='normal'))
            self.root.after(0, lambda: self.progress_bar.stop())
            self.root.after(0, lambda: self.progress_bar.config(mode='determinate'))
    
    def run(self):
        """运行 GUI"""
        self.root.mainloop()


def main():
    """主函数"""
    if not HAS_TK:
        print("[Error] 需要 tkinter 支持")
        return 1
    
    app = VideoCompressorGUI()
    app.run()
    return 0


if __name__ == '__main__':
    sys.exit(main())
