# -*- coding: utf-8 -*-
import os, subprocess, threading, time, sys, json, mimetypes
import cv2
import numpy as np
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from datetime import datetime
from email.parser import BytesParser
from email.policy import default
from urllib.parse import urlparse, parse_qs
from collections import deque

# --- CONFIGURAZIONE PERCORSI ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR = os.path.join(BASE_DIR, "files")
HTML_DIR = os.path.join(BASE_DIR, "HTML")

API_PORT = 57344
HTML_PORT = 8080
API_KEY = "password" 

os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(HTML_DIR, exist_ok=True)

# --- GENERAZIONE INDEX ---
def generate_index():
    index_path = os.path.join(HTML_DIR, "index.html")
    files = [f for f in os.listdir(HTML_DIR) if os.path.isfile(os.path.join(HTML_DIR, f)) and f != "index.html"]
    links_html = "".join([f'        <li><a href="/{f}">HTML/{f}</a></li>\n' for f in files])
    html_content = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Server Index</title>
    <style>body{{font-family:sans-serif;padding:20px;background:#f4f4f9;}}li{{margin:8px 0;background:white;padding:12px;border-radius:6px;box-shadow:0 2px 4px rgba(0,0,0,0.05);list-style:none;}}a{{text-decoration:none;color:#007bff;font-weight:bold;}}</style>
    </head><body><h2>File in HTML/</h2><ul>{links_html}</ul></body></html>"""
    with open(index_path, "w", encoding="utf-8") as f: f.write(html_content)
    print("[*] Index generato.")

# Stato globale con buffer circolare per frame
FRAME_BUFFER = deque(maxlen=3)  # Mantiene ultimi 3 frame
FRAME_BUFFER_LOCK = threading.Lock()
FRAME_COUNTER = 0
LAST_FRAME_TIME = time.time()
FPS_COUNTER = 0
CURRENT_FPS = 0

ffmpeg_mjpeg = None
ffmpeg_mp4 = None
current_camera_name = None 
avg_frame = None
webm_buffer = b""  # Buffer per accumulare chunk WebM
first_chunk_received = False  # Flag per primo chunk

camera_commands = {
    "flash": "off", "alarm": "off", "rec": "off",     
    "switch": "off", "motion": "off", "motion_threshold": "2"
}
commands_lock = threading.Lock()

# Telemetria batteria, temperatura e GPS
telemetry_data = {
    "battery": "N/A",
    "charging": "0",
    "temperature": "N/A",
    "gps_lat": "N/A",
    "gps_lon": "N/A",
    "gps_accuracy": "N/A",
    "last_update": None
}
telemetry_lock = threading.Lock()

# Statistiche stream
stream_stats = {
    "fps": 0,
    "active_clients": 0,
    "total_frames": 0,
    "last_chunk_time": None
}
stats_lock = threading.Lock()

# --- MOTION DETECTION ---
def process_motion(jpg_data):
    global avg_frame
    try:
        nparr = np.frombuffer(jpg_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if frame is None: return
        frame = cv2.resize(frame, (320, 240))
        if avg_frame is None: avg_frame = frame.copy().astype("float"); return
        cv2.accumulateWeighted(frame, avg_frame, 0.05)
        delta = cv2.absdiff(frame, cv2.convertScaleAbs(avg_frame))
        px = np.count_nonzero(cv2.threshold(delta, 15, 255, cv2.THRESH_BINARY)[1])
        with commands_lock:
            thr = float(camera_commands.get("motion_threshold", 1.0))
            camera_commands["motion"] = "on" if (px / (320*240)) * 100 >= thr else "off"
    except: pass

# --- FPS MONITOR ---
def update_fps():
    global CURRENT_FPS, FPS_COUNTER, LAST_FRAME_TIME
    while True:
        time.sleep(1)
        with stats_lock:
            stream_stats["fps"] = FPS_COUNTER
            CURRENT_FPS = FPS_COUNTER
            FPS_COUNTER = 0

threading.Thread(target=update_fps, daemon=True).start()

# --- HANDLER API ---
class ApiHandler(BaseHTTPRequestHandler):
    def _send_json(self, code, payload):
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.end_headers()
        self.wfile.write(payload.encode("utf-8"))

    def do_GET(self):
        if not self.check_auth(): return
        path = urlparse(self.path).path
        query = parse_qs(urlparse(self.path).query)
        
        if path == "/control":
            cmd = query.get("cmd", [""])[0]
            if "/" in cmd:
                k, v = cmd.split("/")
                with commands_lock: camera_commands[k] = v
                self._send_json(200, '{"status": "ok"}')
        
        elif path == "/get_commands":
            # Ricevi telemetria dai client
            batt = query.get("batt", [None])[0]
            charging = query.get("charging", [None])[0]
            temp = query.get("temp", [None])[0]
            gps_lat = query.get("gps_lat", [None])[0]
            gps_lon = query.get("gps_lon", [None])[0]
            gps_accuracy = query.get("gps_accuracy", [None])[0]
            
            with telemetry_lock:
                if batt is not None: telemetry_data["battery"] = batt
                if charging is not None: telemetry_data["charging"] = charging
                if temp is not None: telemetry_data["temperature"] = temp
                if gps_lat is not None: telemetry_data["gps_lat"] = gps_lat
                if gps_lon is not None: telemetry_data["gps_lon"] = gps_lon
                if gps_accuracy is not None: telemetry_data["gps_accuracy"] = gps_accuracy
                
                if any(x is not None for x in [batt, temp, gps_lat]):
                    telemetry_data["last_update"] = datetime.now().strftime("%H:%M:%S")
            
            with commands_lock: 
                response = camera_commands.copy()
            self._send_json(200, json.dumps(response))
        
        elif path == "/get_telemetry":
            with telemetry_lock:
                data_copy = telemetry_data.copy()
            self._send_json(200, json.dumps(data_copy))
        
        elif path == "/get_stats":
            # Nuovo endpoint per statistiche stream
            with stats_lock:
                stats_copy = stream_stats.copy()
            self._send_json(200, json.dumps(stats_copy))
        
        elif path == "/snapshot":
            # Nuovo endpoint per snapshot
            with FRAME_BUFFER_LOCK:
                if len(FRAME_BUFFER) > 0:
                    latest_frame = FRAME_BUFFER[-1]
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(latest_frame)
                else:
                    self.send_error(404, "No frame available")
        
        elif path.startswith("/live"): 
            self.handle_live_stream()
        
        elif path.startswith("/new_session"):
            # Genera un session ID semplice
            import uuid
            session_id = str(uuid.uuid4())[:8]
            self._send_json(200, json.dumps({"status": "reset", "session_id": session_id}))

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Cache-Control")
        self.end_headers()
    
    def do_POST(self):
        if not self.check_auth(): return
        if self.path.startswith("/upload"): self.handle_upload()
        elif self.path.startswith("/finalize"): self.reset_session(); self._send_json(200, '{"status": "done"}')

    def check_auth(self):
        query = parse_qs(urlparse(self.path).query)
        if query.get("key", [""])[0] != API_KEY:
            self.send_error(403); return False
        return True

    def start_ffmpeg_mjpeg(self):
        # OTTIMIZZAZIONE: Ridotto probesize, aumentato buffer, qualità migliorata
        cmd = ["ffmpeg", "-y", "-loglevel", "warning", 
               "-f", "webm", "-i", "pipe:0",
               "-probesize", "32", "-analyzeduration", "1000000",
               "-vf", "scale=480:-1:flags=fast_bilinear,drawtext=fontsize=12:fontcolor=white:box=1:boxcolor=black@0.5:boxborderw=4:x=8:y=8:text='%{localtime\\:%H\\\\\\:%M\\\\\\:%S}'",
               "-r", "25", "-q:v", "3",
               "-f", "mjpeg", "pipe:1"]
        
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, 
                           stderr=subprocess.PIPE, bufsize=65536)
        
        def reader(proc):
            global FRAME_COUNTER, FPS_COUNTER
            buf = b""
            while proc.poll() is None:
                try:
                    chunk = proc.stdout.read(16384)
                    if not chunk: break
                    buf += chunk
                    
                    while True:
                        s, e = buf.find(b"\xff\xd8"), buf.find(b"\xff\xd9")
                        if s != -1 and e != -1 and e > s:
                            jpg = buf[s:e+2]
                            
                            with FRAME_BUFFER_LOCK: 
                                FRAME_BUFFER.append(jpg)
                                FRAME_COUNTER += 1
                                FPS_COUNTER += 1
                            
                            with stats_lock:
                                stream_stats["total_frames"] = FRAME_COUNTER
                            
                            # Motion detection asincrona
                            threading.Thread(target=process_motion, args=(jpg,), daemon=True).start()
                            buf = buf[e+2:]
                        else: 
                            break
                except Exception as e:
                    print(f"[ERROR] Reader exception: {e}")
                    break
        
        threading.Thread(target=reader, args=(p,), daemon=True).start()
        return p

    def start_ffmpeg_mp4(self, cam_name):
        safe = "".join([c if c.isalnum() else "_" for c in cam_name])
        p = os.path.join(VIDEO_DIR, safe)
        os.makedirs(p, exist_ok=True)
        out = os.path.join(p, f"rec_{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}.mp4")
        
        # OTTIMIZZAZIONE: Preset più veloce, ridotta risoluzione, keyframe più frequenti
        cmd = ["ffmpeg", "-y", "-loglevel", "warning", 
               "-f", "webm", "-i", "pipe:0",
               "-c:v", "libx264", "-preset", "veryfast",
               "-crf", "25",
               "-g", "25",
               "-vf", "scale=480:-1:flags=fast_bilinear,drawtext=fontsize=12:fontcolor=white:box=1:boxcolor=black@0.5:boxborderw=4:x=8:y=8:text='%{localtime\\:%d-%m-%Y %H\\\\\\:%M\\\\\\:%S}'",
               "-fflags", "nobuffer", "-flush_packets", "1", 
               "-movflags", "+faststart",
               out]
        
        return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=131072)

    def handle_upload(self):
        global ffmpeg_mjpeg, ffmpeg_mp4, current_camera_name, webm_buffer, first_chunk_received
        try:
            cl = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(cl)
            cam = parse_qs(urlparse(self.path).query).get("camera", ["default"])[0]
            
            # Reset quando cambia camera (per switch)
            if current_camera_name != cam:
                print(f"[SWITCH] Camera cambiata: {current_camera_name} -> {cam}")
                self.reset_session()
                current_camera_name = cam
                webm_buffer = b""
                first_chunk_received = False
                time.sleep(0.1)

            msg = BytesParser(policy=default).parsebytes(
                b"Content-Type: " + self.headers.get("Content-Type").encode() + b"\r\n\r\n" + body)
            chunk = next((p.get_payload(decode=True) for p in msg.iter_parts() if p.get_filename()), None)
            if not chunk: 
                self._send_json(200, '{"status": "ok"}')
                return

            # Accumula primi chunk per header completo
            if not first_chunk_received:
                webm_buffer += chunk
                if len(webm_buffer) > 50000:  # ~50KB dovrebbero contenere header
                    first_chunk_received = True
                    chunk = webm_buffer
                    webm_buffer = b""
                    print(f"[INFO] First chunk complete: {len(chunk)} bytes")
                else:
                    self._send_json(200, '{"status": "buffering"}')
                    return

            # Restart FFmpeg se necessario
            if not ffmpeg_mjpeg or ffmpeg_mjpeg.poll() is not None: 
                print("[START] Avvio FFmpeg MJPEG")
                ffmpeg_mjpeg = self.start_ffmpeg_mjpeg()
                time.sleep(0.3)
            
            if not ffmpeg_mp4 or ffmpeg_mp4.poll() is not None: 
                print("[START] Avvio FFmpeg MP4")
                ffmpeg_mp4 = self.start_ffmpeg_mp4(cam)
                time.sleep(0.3)
            
            # Scrittura con gestione errori e retry
            if ffmpeg_mjpeg and ffmpeg_mjpeg.poll() is None:
                try:
                    ffmpeg_mjpeg.stdin.write(chunk)
                    ffmpeg_mjpeg.stdin.flush()
                except BrokenPipeError:
                    print(f"[ERROR] MJPEG BrokenPipe - Restarting...")
                    ffmpeg_mjpeg = self.start_ffmpeg_mjpeg()
                    time.sleep(0.3)
                    try:
                        ffmpeg_mjpeg.stdin.write(chunk)
                        ffmpeg_mjpeg.stdin.flush()
                    except Exception as e:
                        print(f"[ERROR] MJPEG retry failed: {e}")
                except Exception as e:
                    print(f"[ERROR] MJPEG write: {e}")
                    ffmpeg_mjpeg = None
            
            if ffmpeg_mp4 and ffmpeg_mp4.poll() is None:
                try:
                    ffmpeg_mp4.stdin.write(chunk)
                    ffmpeg_mp4.stdin.flush()
                except BrokenPipeError:
                    print(f"[ERROR] MP4 BrokenPipe - Restarting...")
                    ffmpeg_mp4 = self.start_ffmpeg_mp4(cam)
                    time.sleep(0.3)
                    try:
                        ffmpeg_mp4.stdin.write(chunk)
                        ffmpeg_mp4.stdin.flush()
                    except Exception as e:
                        print(f"[ERROR] MP4 retry failed: {e}")
                except Exception as e:
                    print(f"[ERROR] MP4 write: {e}")
                    ffmpeg_mp4 = None
            
            with stats_lock:
                stream_stats["last_chunk_time"] = datetime.now().strftime("%H:%M:%S")
            
            self._send_json(200, '{"status": "ok"}')
        except Exception as e:
            print(f"[ERROR] Upload: {e}")
            import traceback
            traceback.print_exc()
            self._send_json(500, '{"status": "error"}')

    def reset_session(self):
        global ffmpeg_mjpeg, ffmpeg_mp4, avg_frame, webm_buffer, first_chunk_received
        print("[RESET] Resetting session...")
        
        # Reset motion detection
        avg_frame = None
        webm_buffer = b""
        first_chunk_received = False
        
        # Chiusura MJPEG
        if ffmpeg_mjpeg:
            try: 
                ffmpeg_mjpeg.stdin.close()
                ffmpeg_mjpeg.wait(timeout=2)
            except: 
                try: ffmpeg_mjpeg.kill()
                except: pass
        
        # Chiusura MP4 con finalizzazione corretta
        if ffmpeg_mp4:
            try:
                ffmpeg_mp4.stdin.close()
                ffmpeg_mp4.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try: 
                    ffmpeg_mp4.terminate()
                    ffmpeg_mp4.wait(timeout=2)
                except:
                    try: ffmpeg_mp4.kill()
                    except: pass
        
        ffmpeg_mjpeg = ffmpeg_mp4 = None
        
        # Clear buffer per evitare frame vecchi
        with FRAME_BUFFER_LOCK:
            FRAME_BUFFER.clear()
        
        print("[RESET] Sessione resettata")

    def handle_live_stream(self):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Connection", "close")
        self.end_headers()
        
        with stats_lock:
            stream_stats["active_clients"] += 1
        
        last_id = -1
        try:
            while True:
                with FRAME_BUFFER_LOCK:
                    if len(FRAME_BUFFER) > 0:
                        # Usa sempre l'ultimo frame disponibile
                        frame = FRAME_BUFFER[-1]
                        current_id = FRAME_COUNTER
                    else:
                        frame = None
                        current_id = last_id
                
                if frame and current_id != last_id:
                    try:
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
                        self.wfile.flush()
                        last_id = current_id
                    except:
                        break
                
                time.sleep(0.04)  # ~25 FPS per client
        except:
            pass
        finally:
            with stats_lock:
                stream_stats["active_clients"] = max(0, stream_stats["active_clients"] - 1)

# --- HANDLER HTML ---
class HtmlHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        req = urlparse(self.path).path.lstrip('/')
        if req == "" or req.endswith('/'): req = "index.html"
        full_path = os.path.abspath(os.path.join(HTML_DIR, req))
        if not full_path.startswith(os.path.abspath(HTML_DIR)): self.send_error(403); return
        if os.path.exists(full_path) and os.path.isfile(full_path):
            self.send_response(200)
            self.send_header("Content-Type", mimetypes.guess_type(full_path)[0] or "text/plain")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            with open(full_path, 'rb') as f: self.wfile.write(f.read())
        else: self.send_error(404)

if __name__ == "__main__":
    print("[*] Starting Optimized Surveillance Server")
    print(f"[*] API Port: {API_PORT}")
    print(f"[*] HTML Port: {HTML_PORT}")
    generate_index()
    threading.Thread(target=lambda: ThreadingHTTPServer(("0.0.0.0", API_PORT), ApiHandler).serve_forever(), daemon=True).start()
    threading.Thread(target=lambda: ThreadingHTTPServer(("0.0.0.0", HTML_PORT), HtmlHandler).serve_forever(), daemon=True).start()
    print("[*] Server Running - Press Ctrl+C to stop")
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt: 
        print("\n[*] Shutting down...")
