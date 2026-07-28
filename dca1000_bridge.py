"""Ponte Python verso mmWave Studio e la scheda di acquisizione DCA1000.

Incapsula i comandi Lua/RSTD usati per connettere l'hardware, armare la
registrazione e avviare o fermare uno stream, esponendo uno stato adatto alla
GUI principale.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path


DEFAULT_STUDIO_ROOT = Path(r"C:\ti\mmwave_studio_02_01_01_00\mmWaveStudio")
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 2777
SUCCESS_CODES = {0, 30000}


class MmwaveStudioError(RuntimeError):
    """Raised when the bridge cannot talk to mmWave Studio."""


def list_available_com_ports() -> list[str]:
    """Elenca porte seriali Windows, prima dal registro poi con pyserial."""
    """Elenca porte seriali Windows, prima dal registro poi con pyserial."""
    ports: list[str] = []
    try:
        import winreg  # type: ignore

        reg_path = r"HARDWARE\DEVICEMAP\SERIALCOMM"
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, reg_path) as key:
            index = 0
            while True:
                try:
                    _, value, _ = winreg.EnumValue(key, index)
                except OSError:
                    break
                if value:
                    ports.append(str(value).upper())
                index += 1
    except Exception:
        pass

    if not ports:
        try:
            from serial.tools import list_ports  # type: ignore

            for item in list_ports.comports():
                device = str(getattr(item, "device", "")).strip()
                if device:
                    ports.append(device.upper())
        except Exception:
            pass

    return sorted(set(ports))


def _status_code(raw_status) -> int:
    """
    Normalize pythonnet return values.

    Some RSTD/.NET calls return an int, others come back as tuples such as
    `(30000, '...')` or `(0, ...)`. We only need the leading numeric status.
    """
    if isinstance(raw_status, int):
        return raw_status
    if isinstance(raw_status, (tuple, list)):
        if not raw_status:
            raise MmwaveStudioError("Empty status returned by RSTD API")
        return _status_code(raw_status[0])
    try:
        return int(raw_status)
    except Exception as exc:
        raise MmwaveStudioError(
            f"Unsupported status type returned by RSTD API: {type(raw_status).__name__} -> {raw_status!r}"
        ) from exc


def _lua_path(value: str | Path) -> str:
    return str(Path(value)).replace("\\", "\\\\")


@dataclass(slots=True)
class SequenceResult:
    """Esiti della sequenza atomica StartRecord → StartFrame → StopFrame."""

    """Esiti della sequenza atomica StartRecord → StartFrame → StopFrame."""

    adc_data_path: Path
    start_record_status: int
    start_frame_status: int
    stop_frame_status: int
    frame_run_s: float


@dataclass(slots=True)
class RadarConnectionConfig:
    """Parametri della porta UART usata da mmWave Studio per il radar."""

    """Parametri della porta UART usata da mmWave Studio per il radar."""

    uart_com_port: int = 24
    baudrate: int = 921600
    timeout_ms: int = 1000


@dataclass(slots=True)
class DCA1000Config:
    """Parametri rete e modalità di cattura della DCA1000."""

    """Parametri rete e modalità di cattura della DCA1000."""

    capture_device: str = "DCA1000"
    pc_ip: str = "192.168.33.30"
    capture_card_ip: str = "192.168.33.180"
    capture_card_mac: str = "12:34:56:78:90:12"
    config_port: int = 4096
    record_port: int = 4098
    packet_delay: int = 25
    mode_recording: int = 1
    mode_device_type: int = 1
    mode_data_format: int = 1
    mode_lvds_mode: int = 2
    mode_data_transfer: int = 3
    mode_data_capture: int = 30
    adc_data_path: Path = DEFAULT_STUDIO_ROOT / "PostProc" / "adc_data.bin"


@dataclass(slots=True)
class GuiBridgeState:
    """Stato serializzabile che la GUI può leggere senza interrogare l'hardware."""

    """Stato serializzabile che la GUI può leggere senza interrogare l'hardware."""

    connected: bool
    streaming: bool
    rstd_connected: bool = False
    radar_connected: bool = False
    dca_ready: bool = False
    last_error: str = ""
    last_message: str = ""
    last_rearm_s: float = 0.0


class MmwaveStudioBridge:
    """
    Minimal Python bridge for mmWave Studio 2.1.1.0 via RSTD.NetStart().

    Workflow:
    1. Open mmWave Studio.
    2. In the Lua shell run: RSTD.NetStart()
    3. Use this bridge from Python or from CLI.
    """

    def __init__(
        self,
        studio_root: str | Path = DEFAULT_STUDIO_ROOT,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
    ) -> None:
        self.studio_root = Path(studio_root)
        self.host = str(host)
        self.port = int(port)
        self._api = None
        self._connected = False
        self._hw_connected = False
        self._streaming = False
        self._radar_connected = False
        self._dca_ready = False
        self._last_rearm_s = 0.0
        self._lock = threading.RLock()
        self._last_error = ""
        self._last_message = "Idle"

    @property
    def netclient_dll(self) -> Path:
        return self.studio_root / "Clients" / "RtttNetClientController" / "RtttNetClientAPI.dll"

    def connect(self) -> None:
        with self._lock:
            if self._connected:
                return
            self._load_api()

            init_status = _status_code(self._api.Init())
            if init_status not in SUCCESS_CODES:
                raise MmwaveStudioError(f"RtttNetClient.Init failed with status {init_status}")

            connect_status = _status_code(self._api.Connect(self.host, self.port))
            if connect_status not in SUCCESS_CODES:
                raise MmwaveStudioError(
                    "RtttNetClient.Connect failed with status "
                    f"{connect_status}. In mmWave Studio run RSTD.NetStart() in the Lua shell."
                )
            self._connected = True
            self._last_error = ""
            self._last_message = f"Connected to RSTD {self.host}:{self.port}"

    def disconnect(self) -> None:
        with self._lock:
            if self._api is None or not self._connected:
                return
            try:
                self._api.Disconnect()
            finally:
                self._connected = False
                self._hw_connected = False
                self._streaming = False
                self._radar_connected = False
                self._dca_ready = False
                self._last_message = "Disconnected from RSTD"

    def __enter__(self) -> "MmwaveStudioBridge":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.disconnect()

    def send_lua(self, lua_command: str, *, label: str | None = None) -> int:
        """Invia un comando RSTD, uniformando codici di esito ed errori contestuali."""
        """Invia un comando RSTD, uniformando codici di esito ed errori contestuali."""
        with self._lock:
            self.connect()
            print(f"[mmwave] TX {label or 'lua'}: {lua_command}", flush=True)
            status = _status_code(self._api.SendCommand(str(lua_command)))
            if status not in SUCCESS_CODES:
                msg = f"SendCommand failed with status {status}"
                if label:
                    msg = f"{msg} while sending {label}"
                self._last_error = msg
                print(f"[mmwave] ERR: {msg}", flush=True)
                raise MmwaveStudioError(f"{msg}\nLua:\n{lua_command}")
            self._last_error = ""
            self._last_message = label or "Lua command sent"
            print(f"[mmwave] OK {label or 'lua'} -> status {status}", flush=True)
            return status

    @property
    def is_rstd_connected(self) -> bool:
        return bool(self._connected)

    @property
    def is_hw_connected(self) -> bool:
        return bool(self._hw_connected)

    @property
    def is_streaming(self) -> bool:
        return bool(self._streaming)

    def get_gui_state(self) -> GuiBridgeState:
        return GuiBridgeState(
            connected=bool(self._hw_connected),
            streaming=bool(self._streaming),
            rstd_connected=bool(self._connected),
            radar_connected=bool(self._radar_connected),
            dca_ready=bool(self._dca_ready),
            last_error=str(self._last_error),
            last_message=str(self._last_message),
            last_rearm_s=float(self._last_rearm_s),
        )

    def set_status(self, *, message: str | None = None, error: str | None = None) -> GuiBridgeState:
        with self._lock:
            if message is not None:
                self._last_message = str(message)
            if error is not None:
                self._last_error = str(error)
            return self.get_gui_state()

    def select_capture_device(self, capture_device: str = "DCA1000") -> int:
        return self.send_lua(
            f'ar1.SelectCaptureDevice("{str(capture_device)}")',
            label="SelectCaptureDevice",
        )

    def radar_connect(self, uart_com_port: int, baudrate: int = 921600, timeout_ms: int = 1000) -> int:
        port_name = f"COM{int(uart_com_port)}".upper()
        available_ports = list_available_com_ports()
        if available_ports and port_name not in available_ports:
            raise MmwaveStudioError(
                f"Serial port {port_name} not found. Available ports: {', '.join(available_ports)}"
            )
        lua = f"ar1.Connect({int(uart_com_port)},{int(baudrate)},{int(timeout_ms)})"
        return self.send_lua(lua, label="Radar Connect")

    def radar_disconnect(self) -> int:
        return self.send_lua("ar1.Disconnect()", label="Radar Disconnect")

    def setup_dca1000(self, config: DCA1000Config) -> list[int]:
        # Il firmware DCA1000 richiede quest'ordine: scheda selezionata, rete,
        # modalità LVDS e infine ritardo fra pacchetti prima di poter armare la cattura.
        commands = [
            f'ar1.SelectCaptureDevice("{config.capture_device}")',
            (
                'ar1.CaptureCardConfig_EthInit('
                f'"{config.pc_ip}", "{config.capture_card_ip}", "{config.capture_card_mac}", '
                f"{int(config.config_port)}, {int(config.record_port)})"
            ),
            (
                "ar1.CaptureCardConfig_Mode("
                f"{int(config.mode_recording)}, {int(config.mode_device_type)}, {int(config.mode_data_format)}, "
                f"{int(config.mode_lvds_mode)}, {int(config.mode_data_transfer)}, {int(config.mode_data_capture)})"
            ),
            f"ar1.CaptureCardConfig_PacketDelay({int(config.packet_delay)})",
        ]
        statuses = self.send_commands(commands)
        self._dca_ready = True
        return statuses

    def connect_hardware(
        self,
        radar: RadarConnectionConfig | None = None,
        dca: DCA1000Config | None = None,
    ) -> GuiBridgeState:
        """Connette nell'ordine necessario: RSTD, radar UART, poi DCA1000."""
        """Connette nell'ordine necessario: RSTD, radar UART, poi DCA1000."""
        radar_cfg = radar or RadarConnectionConfig()
        dca_cfg = dca or DCA1000Config()
        with self._lock:
            if self._hw_connected:
                self._last_message = "Hardware already connected"
                return self.get_gui_state()
            # RSTD deve essere collegato prima; Studio associa poi radar e DCA1000
            # alla sessione nell'ordine selezione scheda -> UART radar -> rete DCA1000.
            self.connect()
            self.select_capture_device(dca_cfg.capture_device)
            self.radar_connect(
                uart_com_port=radar_cfg.uart_com_port,
                baudrate=radar_cfg.baudrate,
                timeout_ms=radar_cfg.timeout_ms,
            )
            self._radar_connected = True
            self.setup_dca1000(dca_cfg)
            self._hw_connected = True
            self._streaming = False
            self._last_error = ""
            self._last_message = (
                f"Connected DCA1000 + radar on COM{int(radar_cfg.uart_com_port)} | "
                f"PC {dca_cfg.pc_ip} -> FPGA {dca_cfg.capture_card_ip}"
            )
            print(f"[mmwave] {self._last_message}", flush=True)
            return self.get_gui_state()

    def disconnect_hardware(self, *, stop_delay_s: float = 2.0) -> GuiBridgeState:
        with self._lock:
            if self._streaming:
                try:
                    # Fermare i frame prima di scollegare la UART evita che il radar
                    # continui a inviare campioni verso una DCA1000 non più configurata.
                    self.stop_streaming(stop_delay_s=stop_delay_s)
                except Exception:
                    # Lo shutdown prosegue anche se lo stato locale e quello hardware
                    # sono già divergenti: le risorse rimanenti vanno comunque rilasciate.
                    pass
            if self._hw_connected:
                try:
                    if self._radar_connected:
                        self.radar_disconnect()
                finally:
                    # Aggiorniamo sempre lo stato locale, anche quando la disconnessione
                    # UART fallisce, per non lasciare la GUI in uno stato fittiziamente attivo.
                    self._hw_connected = False
                    self._radar_connected = False
                    self._dca_ready = False
            self.disconnect()
            self._last_error = ""
            self._last_message = "Hardware disconnected"
            print(f"[mmwave] {self._last_message}", flush=True)
            return self.get_gui_state()

    def toggle_connection(
        self,
        radar: RadarConnectionConfig | None = None,
        dca: DCA1000Config | None = None,
    ) -> GuiBridgeState:
        if self._hw_connected:
            return self.disconnect_hardware()
        return self.connect_hardware(radar=radar, dca=dca)

    def start_record(self, adc_data_path: str | Path, capture_mode: int = 1) -> int:
        adc_path = _lua_path(adc_data_path)
        lua = f'ar1.CaptureCardConfig_StartRecord("{adc_path}", {int(capture_mode)})'
        return self.send_lua(lua, label="CaptureCardConfig_StartRecord")

    def start_frame(self) -> int:
        return self.send_lua("ar1.StartFrame()", label="StartFrame")

    def stop_frame(self) -> int:
        return self.send_lua("ar1.StopFrame()", label="StopFrame")

    def send_commands(self, commands: list[str]) -> list[int]:
        statuses: list[int] = []
        for command in commands:
            statuses.append(self.send_lua(command))
        return statuses

    def start_streaming(
        self,
        adc_data_path: str | Path,
        *,
        capture_mode: int = 1,
        arm_delay_s: float = 1.0,
    ) -> GuiBridgeState:
        """Arma la DCA1000 prima di avviare i frame radar, nell'ordine RSTD corretto."""
        """Arma la DCA1000 prima di avviare i frame radar, nell'ordine RSTD corretto."""
        with self._lock:
            if not self._hw_connected:
                raise MmwaveStudioError("Hardware is not connected. Connect DCA1000 and radar first.")
            if self._streaming:
                self._last_message = "Streaming already active"
                return self.get_gui_state()
            # La DCA1000 va armata prima del radar: StartFrame fa partire subito i chirp
            # e un record non ancora armato perderebbe l'inizio della cattura.
            self.start_record(adc_data_path, capture_mode=capture_mode)
            if arm_delay_s > 0:
                # Lasciamo al firmware il tempo di applicare il comando di armamento.
                time.sleep(float(arm_delay_s))
            self.start_frame()
            self._streaming = True
            self._last_error = ""
            self._last_message = f"Streaming started -> {Path(adc_data_path)}"
            print(f"[mmwave] {self._last_message}", flush=True)
            return self.get_gui_state()

    def stop_streaming(self, *, stop_delay_s: float = 2.0) -> GuiBridgeState:
        with self._lock:
            if not self._streaming:
                self._last_message = "Streaming already stopped"
                return self.get_gui_state()
            # StopFrame precede l'attesa: il ritardo consente alla catena DCA1000 di
            # completare i pacchetti gia' in transito prima di dichiarare fermo lo stream.
            self.stop_frame()
            if stop_delay_s > 0:
                time.sleep(float(stop_delay_s))
            self._streaming = False
            self._last_error = ""
            self._last_message = "Streaming stopped"
            print(f"[mmwave] {self._last_message}", flush=True)
            return self.get_gui_state()

    def rearm_streaming(
        self,
        adc_data_path: str | Path,
        *,
        capture_mode: int = 1,
        arm_delay_s: float = 0.25,
        stop_delay_s: float = 0.25,
    ) -> GuiBridgeState:
        """Ri-arma una registrazione senza ripetere la connessione hardware completa."""
        """Ri-arma una registrazione senza ripetere la connessione hardware completa."""
        with self._lock:
            if not self._hw_connected:
                raise MmwaveStudioError("Cannot rearm streaming: hardware is not connected.")
            try:
                # Lo stop è intenzionalmente best-effort: dopo un'interruzione della GUI
                # il radar potrebbe essere già fermo, ma il nuovo record deve essere riarmato.
                self.stop_frame()
                if stop_delay_s > 0:
                    time.sleep(float(stop_delay_s))
            except Exception:
                pass
            self.start_record(adc_data_path, capture_mode=capture_mode)
            if arm_delay_s > 0:
                time.sleep(float(arm_delay_s))
            self.start_frame()
            self._streaming = True
            self._last_rearm_s = float(time.perf_counter())
            self._last_error = ""
            self._last_message = f"Streaming re-armed -> {Path(adc_data_path)}"
            print(f"[mmwave] {self._last_message}", flush=True)
            return self.get_gui_state()

    def rearm_dca_only(
        self,
        adc_data_path: str | Path,
        *,
        capture_mode: int = 1,
        arm_delay_s: float = 0.1,
    ) -> GuiBridgeState:
        with self._lock:
            if not self._hw_connected:
                raise MmwaveStudioError("Cannot re-arm DCA1000: hardware is not connected.")
            self.start_record(adc_data_path, capture_mode=capture_mode)
            if arm_delay_s > 0:
                time.sleep(float(arm_delay_s))
            self._streaming = True
            self._last_rearm_s = float(time.perf_counter())
            self._last_error = ""
            self._last_message = f"DCA1000 re-armed -> {Path(adc_data_path)}"
            print(f"[mmwave] {self._last_message}", flush=True)
            return self.get_gui_state()

    def toggle_streaming(
        self,
        adc_data_path: str | Path,
        *,
        capture_mode: int = 1,
        arm_delay_s: float = 1.0,
        stop_delay_s: float = 2.0,
    ) -> GuiBridgeState:
        if self._streaming:
            return self.stop_streaming(stop_delay_s=stop_delay_s)
        return self.start_streaming(
            adc_data_path=adc_data_path,
            capture_mode=capture_mode,
            arm_delay_s=arm_delay_s,
        )

    def capture_once(
        self,
        adc_data_path: str | Path,
        *,
        capture_mode: int = 1,
        arm_delay_s: float = 1.0,
        frame_run_s: float = 2.0,
        stop_delay_s: float = 2.0,
    ) -> SequenceResult:
        """
        Replicates the sequence:
        StartRecord -> wait -> StartFrame -> wait -> StopFrame -> wait
        """

        # L'ordine rispecchia la sequenza hardware: armamento DCA1000, produzione dei
        # frame radar, quindi arresto e tempo residuo per svuotare la cattura.
        adc_path = Path(adc_data_path)
        start_record_status = self.start_record(adc_path, capture_mode=capture_mode)
        if arm_delay_s > 0:
            time.sleep(float(arm_delay_s))

        start_frame_status = self.start_frame()
        if frame_run_s > 0:
            time.sleep(float(frame_run_s))

        stop_frame_status = self.stop_frame()
        if stop_delay_s > 0:
            time.sleep(float(stop_delay_s))

        return SequenceResult(
            adc_data_path=adc_path,
            start_record_status=start_record_status,
            start_frame_status=start_frame_status,
            stop_frame_status=stop_frame_status,
            frame_run_s=float(frame_run_s),
        )

    def build_capture_once_lua(
        self,
        adc_data_path: str | Path,
        *,
        capture_mode: int = 1,
        arm_delay_ms: int = 1000,
        frame_run_ms: int = 2000,
        stop_delay_ms: int = 2000,
    ) -> str:
        adc_path = _lua_path(adc_data_path)
        lines = [f'ar1.CaptureCardConfig_StartRecord("{adc_path}", {int(capture_mode)})']
        if int(arm_delay_ms) > 0:
            lines.append(f"RSTD.Sleep({int(arm_delay_ms)})")
        lines.append("ar1.StartFrame()")
        if int(frame_run_ms) > 0:
            lines.append(f"RSTD.Sleep({int(frame_run_ms)})")
        lines.append("ar1.StopFrame()")
        if int(stop_delay_ms) > 0:
            lines.append(f"RSTD.Sleep({int(stop_delay_ms)})")
        return "\n".join(lines)

    def _load_api(self) -> None:
        if self._api is not None:
            return

        dll_path = self.netclient_dll
        if not dll_path.exists():
            raise MmwaveStudioError(f"RtttNetClientAPI.dll not found: {dll_path}")

        try:
            import clr  # type: ignore
        except ImportError as exc:
            raise MmwaveStudioError(
                "pythonnet is required. Install it with "
                r"`.\.venv\Scripts\python.exe -m pip install pythonnet`"
            ) from exc

        dll_dir = str(dll_path.parent)
        runtime_dir = str(self.studio_root / "RunTime")
        if dll_dir not in sys.path:
            sys.path.insert(0, dll_dir)
        if runtime_dir not in sys.path:
            sys.path.insert(0, runtime_dir)

        clr.AddReference(str(dll_path.parent / "RtttNet.dll"))
        controller_dll = dll_path.parent / "RtttNetClientController.dll"
        if controller_dll.exists():
            clr.AddReference(str(controller_dll))
        rstd_net_dll = dll_path.parent / "RstdNet.dll"
        if rstd_net_dll.exists():
            clr.AddReference(str(rstd_net_dll))
        clr.AddReference(str(dll_path))

        from RtttNetClientAPI import RtttNetClient  # type: ignore

        self._api = RtttNetClient


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Python bridge for mmWave Studio 2.1.1.0. "
            "Before using it, open mmWave Studio and run RSTD.NetStart() in the Lua shell."
        )
    )
    parser.add_argument("--studio-root", default=str(DEFAULT_STUDIO_ROOT))
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--print-only", action="store_true", help="Print the Lua preview and exit.")

    sub = parser.add_subparsers(dest="command", required=True)

    lua_cmd = sub.add_parser("lua", help="Send raw Lua code.")
    lua_cmd.add_argument("script")

    start_record = sub.add_parser("start-record", help="Send ar1.CaptureCardConfig_StartRecord(...).")
    start_record.add_argument("adc_path")
    start_record.add_argument("--capture-mode", type=int, default=1)

    sub.add_parser("start-frame", help="Send ar1.StartFrame().")
    sub.add_parser("stop-frame", help="Send ar1.StopFrame().")

    capture_once = sub.add_parser(
        "capture-once",
        help="Run StartRecord -> StartFrame -> StopFrame with Python-side delays.",
    )
    capture_once.add_argument("adc_path")
    capture_once.add_argument("--capture-mode", type=int, default=1)
    capture_once.add_argument("--arm-delay-s", type=float, default=1.0)
    capture_once.add_argument("--frame-run-s", type=float, default=2.0)
    capture_once.add_argument("--stop-delay-s", type=float, default=2.0)

    capture_lua = sub.add_parser(
        "capture-lua",
        help="Build one Lua script with StartRecord -> StartFrame -> StopFrame.",
    )
    capture_lua.add_argument("adc_path")
    capture_lua.add_argument("--capture-mode", type=int, default=1)
    capture_lua.add_argument("--arm-delay-ms", type=int, default=1000)
    capture_lua.add_argument("--frame-run-ms", type=int, default=2000)
    capture_lua.add_argument("--stop-delay-ms", type=int, default=2000)

    return parser


def _build_preview(args: argparse.Namespace, bridge: MmwaveStudioBridge) -> str:
    if args.command == "lua":
        return str(args.script)
    if args.command == "start-record":
        return f'ar1.CaptureCardConfig_StartRecord("{_lua_path(args.adc_path)}", {int(args.capture_mode)})'
    if args.command == "start-frame":
        return "ar1.StartFrame()"
    if args.command == "stop-frame":
        return "ar1.StopFrame()"
    if args.command == "capture-once":
        return bridge.build_capture_once_lua(
            args.adc_path,
            capture_mode=int(args.capture_mode),
            arm_delay_ms=int(round(float(args.arm_delay_s) * 1000.0)),
            frame_run_ms=int(round(float(args.frame_run_s) * 1000.0)),
            stop_delay_ms=int(round(float(args.stop_delay_s) * 1000.0)),
        )
    if args.command == "capture-lua":
        return bridge.build_capture_once_lua(
            args.adc_path,
            capture_mode=int(args.capture_mode),
            arm_delay_ms=int(args.arm_delay_ms),
            frame_run_ms=int(args.frame_run_ms),
            stop_delay_ms=int(args.stop_delay_ms),
        )
    raise MmwaveStudioError(f"Unsupported command: {args.command}")


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    bridge = MmwaveStudioBridge(
        studio_root=args.studio_root,
        host=args.host,
        port=args.port,
    )

    preview = _build_preview(args, bridge)
    if args.print_only or args.command == "capture-lua":
        print(preview)
        return 0

    try:
        with bridge:
            if args.command == "lua":
                bridge.send_lua(args.script, label="raw lua")
                print("Lua command sent successfully.")
                return 0

            if args.command == "start-record":
                bridge.start_record(args.adc_path, capture_mode=args.capture_mode)
                print(f"StartRecord sent successfully for {Path(args.adc_path)}")
                return 0

            if args.command == "start-frame":
                bridge.start_frame()
                print("StartFrame sent successfully.")
                return 0

            if args.command == "stop-frame":
                bridge.stop_frame()
                print("StopFrame sent successfully.")
                return 0

            if args.command == "capture-once":
                result = bridge.capture_once(
                    args.adc_path,
                    capture_mode=args.capture_mode,
                    arm_delay_s=args.arm_delay_s,
                    frame_run_s=args.frame_run_s,
                    stop_delay_s=args.stop_delay_s,
                )
                print(f"Capture completed: {result.adc_data_path}")
                print(
                    "Statuses: "
                    f"start_record={result.start_record_status}, "
                    f"start_frame={result.start_frame_status}, "
                    f"stop_frame={result.stop_frame_status}"
                )
                return 0

    except MmwaveStudioError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(f"Unsupported command: {args.command}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
