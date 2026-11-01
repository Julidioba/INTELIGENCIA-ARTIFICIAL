#!/usr/bin/env python3
"""
Cascarón: esquivar con movimiento en 8 direcciones + registro de datos.


"""

from __future__ import annotations

import csv
import math
import os
import random
from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple

import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.optim as optim

BASE_W, BASE_H = 1280, 800
WINDOW_FRACTION = 0.90
N_PELOTAS = 2
CSV_NAME = "datos_esquivar.csv"

ACCION_QUIETO = 0
ACCION_ARR = 1
ACCION_ABA = 2
ACCION_IZQ = 3
ACCION_DER = 4
ACCION_ARR_IZQ = 5
ACCION_ARR_DER = 6
ACCION_ABA_IZQ = 7
ACCION_ABA_DER = 8

ACCION_NOMBRES = {
    ACCION_QUIETO: "quieto",
    ACCION_ARR: "arriba",
    ACCION_ABA: "abajo",
    ACCION_IZQ: "izq",
    ACCION_DER: "der",
    ACCION_ARR_IZQ: "arr-izq",
    ACCION_ARR_DER: "arr-der",
    ACCION_ABA_IZQ: "aba-izq",
    ACCION_ABA_DER: "aba-der",
}


def dirs_a_accion(dx: int, dy: int) -> int:
    return {
        (0, 0): ACCION_QUIETO,
        (0, -1): ACCION_ARR,
        (0, 1): ACCION_ABA,
        (-1, 0): ACCION_IZQ,
        (1, 0): ACCION_DER,
        (-1, -1): ACCION_ARR_IZQ,
        (1, -1): ACCION_ARR_DER,
        (-1, 1): ACCION_ABA_IZQ,
        (1, 1): ACCION_ABA_DER,
    }.get((dx, dy), ACCION_QUIETO)


def accion_a_dirs(accion: int) -> Tuple[int, int]:
    return {
        ACCION_QUIETO: (0, 0),
        ACCION_ARR: (0, -1),
        ACCION_ABA: (0, 1),
        ACCION_IZQ: (-1, 0),
        ACCION_DER: (1, 0),
        ACCION_ARR_IZQ: (-1, -1),
        ACCION_ARR_DER: (1, -1),
        ACCION_ABA_IZQ: (-1, 1),
        ACCION_ABA_DER: (1, 1),
    }.get(accion, (0, 0))


def tamano_ventana() -> Tuple[int, int]:
    pygame.display.init()
    info = pygame.display.Info()
    sw = info.current_w or BASE_W
    sh = info.current_h or BASE_H
    w = min(max(BASE_W, int(sw * WINDOW_FRACTION)), max(640, sw - 40))
    h = min(max(BASE_H, int(sh * WINDOW_FRACTION)), max(480, sh - 80))
    return w, h


@dataclass
class Sample:
    ball1_dx: float
    ball1_dy: float
    ball1_vx: float
    ball1_vy: float
    ball2_dx: float
    ball2_dy: float
    ball2_vx: float
    ball2_vy: float
    chase_dx: float
    chase_dy: float
    player_x: float
    player_y: float
    move_dx: int
    move_dy: int
    accion: int


FEATURE_NAMES = [
    "ball1_dx",
    "ball1_dy",
    "ball1_vx",
    "ball1_vy",
    "ball2_dx",
    "ball2_dy",
    "ball2_vx",
    "ball2_vy",
    "chase_dx",
    "chase_dy",
    "player_x",
    "player_y",
    "move_dx",
    "move_dy",
    "accion",
]


@dataclass
class Pelota:
    rect: pygame.Rect
    vx: float
    vy: float


@dataclass
class Perseguidor:
    x: float
    y: float
    radio: float
    velocidad: float

    def hitbox(self) -> pygame.Rect:
        r = int(self.radio)
        return pygame.Rect(int(self.x) - r, int(self.y) - r, r * 2, r * 2)
class RedMLP(nn.Module):
    """
    Perceptrón Multicapa (MLP) para clasificación de acciones (0-8)
    a partir de 12 variables de entrada (posiciones y velocidades relativas).
    """
    def __init__(self, in_features: int = 12, hidden_dim: int = 128, out_classes: int = 9) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, out_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Juego:
    def __init__(self) -> None:
        pygame.init()
        self.w, self.h = tamano_ventana()
        self._ventana = (self.w, self.h)
        self.pantalla = pygame.display.set_mode(self._ventana)
        pygame.display.set_caption("Esquivar pelotas")
        self._fullscreen = False

        self.BLANCO = (255, 255, 255)
        self.NEGRO = (0, 0, 0)
        self.GRIS = (200, 200, 200)
        self.AMARILLO = (255, 220, 120)

        self.scale = max(1.0, min(self.w / BASE_W, self.h / BASE_H) * 1.15)
        self.speed = max(3, int(6 * self.scale))
        self.player_size = (int(36 * self.scale), int(52 * self.scale))
        self.ball_size = (int(22 * self.scale), int(22 * self.scale))

        self.fuente = pygame.font.SysFont("Arial", int(24 * self.scale))
        self.fuente_chica = pygame.font.SysFont("Arial", int(18 * self.scale))
        self._cargar_assets()

        self.corriendo = True
        self.modo_auto = False
        self.datos: List[Sample] = []
        self.colisiones = 0
        self.frames_vivos = 0
        self.frame_i = 0
        self.frame_tick = 0

        # Modelo MLP (Perceptrón Multicapa)
        self.modelo: Optional[RedMLP] = None
        self.modelo_entrenado = False
        self.scaler_mean: Optional[np.ndarray] = None
        self.scaler_std: Optional[np.ndarray] = None
        self._cargar_modelo_guardado()

        # Soporte para control (Joystick / Gamepad)
        pygame.joystick.init()
        self.joystick: Optional[pygame.joystick.Joystick] = None
        self._detectar_control()

        self.jugador = pygame.Rect(0, 0, *self.player_size)
        self.pelotas: List[Pelota] = []
        self.perseguidor: Optional[Perseguidor] = None
        self._reset()

    def _detectar_control(self) -> None:
        if pygame.joystick.get_count() > 0:
            try:
                self.joystick = pygame.joystick.Joystick(0)
                self.joystick.init()
            except Exception:
                self.joystick = None
        else:
            self.joystick = None

    def _cargar_assets(self) -> None:
        def load(path: str, size: Tuple[int, int], color=(180, 180, 180)) -> pygame.Surface:
            try:
                img = pygame.image.load(path).convert_alpha()
                return pygame.transform.smoothscale(img, size)
            except Exception:
                s = pygame.Surface(size, pygame.SRCALPHA)
                s.fill((*color, 255))
                return s

        base = os.path.dirname(os.path.abspath(__file__))
        sp = os.path.join(base, "assets", "sprites")
        gm = os.path.join(base, "assets", "game")
        self.frames = [
            load(os.path.join(sp, f"mono_frame_{i}.png"), self.player_size) for i in (1, 2, 3, 4)
        ]
        self.ball_img = load(os.path.join(sp, "purple_ball.png"), self.ball_size, (160, 100, 255))
        self.ball_img2 = self.ball_img.copy()
        self.ball_img2.fill((255, 120, 40, 80), special_flags=pygame.BLEND_RGBA_ADD)
        self.fondo = load(os.path.join(gm, "fondo2.png"), (self.w, self.h), (30, 35, 50))

    def _reset(self) -> None:
        self.jugador.center = (self.w // 2, self.h // 2)
        self.frames_vivos = 0
        self.pelotas.clear()
        for i in range(N_PELOTAS):
            side = -1 if i % 2 == 0 else 1
            x = int(self.w * 0.1) if side < 0 else int(self.w * 0.9)
            y = random.randint(40, self.h - 40)
            vx = random.uniform(3.5, 7.5) * self.scale * side
            vy = random.uniform(-5.0, 5.0) * self.scale
            if abs(vy) < 1.5 * self.scale:
                vy = 2.5 * self.scale * (1 if i % 2 == 0 else -1)
            self.pelotas.append(Pelota(pygame.Rect(x, y, *self.ball_size), vx, vy))

        ang = random.uniform(0, math.tau)
        dist = min(self.w, self.h) * 0.35
        self.perseguidor = Perseguidor(
            x=self.w / 2 + math.cos(ang) * dist,
            y=self.h / 2 + math.sin(ang) * dist,
            radio=28 * self.scale,
            velocidad=2.8 * self.scale,
        )

    # ---- movimiento ----
    def _obtener_direccion(self, keys) -> Tuple[int, int]:
        dx = dy = 0

        # 1. Leer stick del control o D-Pad si está conectado
        if self.joystick is not None:
            DEADZONE = 0.28
            try:
                # Eje 0 = horizontal (izq/der), Eje 1 = vertical (arr/aba)
                ax_x = self.joystick.get_axis(0)
                ax_y = self.joystick.get_axis(1)
                if ax_x < -DEADZONE:
                    dx = -1
                elif ax_x > DEADZONE:
                    dx = 1

                if ax_y < -DEADZONE:
                    dy = -1
                elif ax_y > DEADZONE:
                    dy = 1

                # Si el stick está en reposo, probar con la cruceta (D-Pad)
                if dx == 0 and dy == 0 and self.joystick.get_numhats() > 0:
                    hat_x, hat_y = self.joystick.get_hat(0)
                    if hat_x != 0:
                        dx = hat_x
                    if hat_y != 0:
                        dy = -hat_y  # En Pygame el hat +1 es arriba
            except Exception:
                pass

        # 2. Si no se usó el control, leer teclado (flechas / WASD)
        if dx == 0 and dy == 0:
            if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                dx -= 1
            if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                dx += 1
            if keys[pygame.K_UP] or keys[pygame.K_w]:
                dy -= 1
            if keys[pygame.K_DOWN] or keys[pygame.K_s]:
                dy += 1

        return max(-1, min(1, dx)), max(-1, min(1, dy))

    def _mover_jugador(self, dx: int, dy: int) -> None:
        self.jugador.x += dx * self.speed
        self.jugador.y += dy * self.speed
        self.jugador.clamp_ip(self.pantalla.get_rect())

    def _mover_pelotas(self) -> None:
        for p in self.pelotas:
            p.rect.x += int(p.vx)
            p.rect.y += int(p.vy)
            if p.rect.left <= 0 or p.rect.right >= self.w:
                p.vx *= -1
                p.rect.x = max(0, min(p.rect.x, self.w - p.rect.w))
            if p.rect.top <= 0 or p.rect.bottom >= self.h:
                p.vy *= -1
                p.rect.y = max(0, min(p.rect.y, self.h - p.rect.h))

    def _mover_perseguidor(self) -> None:
        assert self.perseguidor is not None
        dx = self.jugador.centerx - self.perseguidor.x
        dy = self.jugador.centery - self.perseguidor.y
        dist = math.hypot(dx, dy) or 1.0
        self.perseguidor.x += self.perseguidor.velocidad * dx / dist
        self.perseguidor.y += self.perseguidor.velocidad * dy / dist
        r = self.perseguidor.radio
        self.perseguidor.x = max(r, min(self.w - r, self.perseguidor.x))
        self.perseguidor.y = max(r, min(self.h - r, self.perseguidor.y))

    def _colision(self) -> bool:
        assert self.perseguidor is not None
        if self.jugador.colliderect(self.perseguidor.hitbox()):
            return True
        return any(self.jugador.colliderect(p.rect) for p in self.pelotas)

    # ---- datos ----
    def _feat_pelota(self, p: Pelota) -> Tuple[float, float, float, float]:
        return (
            float(p.rect.centerx - self.jugador.centerx),
            float(p.rect.centery - self.jugador.centery),
            float(p.vx),
            float(p.vy),
        )

    def construir_sample(self, move_dx: int, move_dy: int) -> Sample:
        assert self.perseguidor is not None
        f: List[float] = []
        for i in range(2):
            if i < len(self.pelotas):
                f.extend(self._feat_pelota(self.pelotas[i]))
            else:
                f.extend([0.0, 0.0, 0.0, 0.0])
        return Sample(
            ball1_dx=f[0],
            ball1_dy=f[1],
            ball1_vx=f[2],
            ball1_vy=f[3],
            ball2_dx=f[4],
            ball2_dy=f[5],
            ball2_vx=f[6],
            ball2_vy=f[7],
            chase_dx=float(self.perseguidor.x - self.jugador.centerx),
            chase_dy=float(self.perseguidor.y - self.jugador.centery),
            player_x=float(self.jugador.centerx),
            player_y=float(self.jugador.centery),
            move_dx=int(move_dx),
            move_dy=int(move_dy),
            accion=dirs_a_accion(move_dx, move_dy),
        )

    def registrar(self, move_dx: int, move_dy: int) -> None:
        self.datos.append(self.construir_sample(move_dx, move_dy))

    def exportar_csv(self) -> str:
        if not self.datos:
            return "No hay datos."
        ruta = os.path.join(os.path.dirname(os.path.abspath(__file__)), CSV_NAME)
        with open(ruta, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FEATURE_NAMES)
            w.writeheader()
            for s in self.datos:
                w.writerow(asdict(s))
        return f"Guardado {CSV_NAME} ({len(self.datos)} filas)."

    def _cargar_modelo_guardado(self) -> None:
        ruta = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modelo_mlp.pth")
        if os.path.exists(ruta):
            try:
                ckpt = torch.load(ruta, map_location=torch.device("cpu"), weights_only=False)
                self.modelo = RedMLP()
                self.modelo.load_state_dict(ckpt["model_state_dict"])
                self.modelo.eval()
                self.scaler_mean = ckpt["scaler_mean"]
                self.scaler_std = ckpt["scaler_std"]
                self.modelo_entrenado = True
            except Exception as e:
                print(f"Aviso al cargar modelo previo: {e}")

    def _cargar_desde_csv(self, ruta: str) -> None:
        try:
            with open(ruta, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                count = 0
                for row in reader:
                    self.datos.append(
                        Sample(
                            ball1_dx=float(row["ball1_dx"]),
                            ball1_dy=float(row["ball1_dy"]),
                            ball1_vx=float(row["ball1_vx"]),
                            ball1_vy=float(row["ball1_vy"]),
                            ball2_dx=float(row["ball2_dx"]),
                            ball2_dy=float(row["ball2_dy"]),
                            ball2_vx=float(row["ball2_vx"]),
                            ball2_vy=float(row["ball2_vy"]),
                            chase_dx=float(row["chase_dx"]),
                            chase_dy=float(row["chase_dy"]),
                            player_x=float(row["player_x"]),
                            player_y=float(row["player_y"]),
                            move_dx=int(row["move_dx"]),
                            move_dy=int(row["move_dy"]),
                            accion=int(row["accion"]),
                        )
                    )
                    count += 1
                print(f"Cargadas {count} filas desde {ruta}")
        except Exception as e:
            print(f"Error leyendo {ruta}: {e}")

    # ---- Modelo MLP: Entrenamiento e Inferencia ----
    def entrenar_modelo(self) -> str:
        """
        Entrena el Perceptrón Multicapa con los datos recolectados.
        """
        # Si no hay datos en memoria, intentar cargar desde CSV si existe
        if len(self.datos) < 50:
            ruta_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), CSV_NAME)
            if os.path.exists(ruta_csv):
                self._cargar_desde_csv(ruta_csv)

        if len(self.datos) < 50:
            return f"Datos insuficientes ({len(self.datos)}). Juega en [M] al menos 20 seg."

        X_list = []
        y_list = []
        for s in self.datos:
            X_list.append([
                s.ball1_dx, s.ball1_dy, s.ball1_vx, s.ball1_vy,
                s.ball2_dx, s.ball2_dy, s.ball2_vx, s.ball2_vy,
                s.chase_dx, s.chase_dy,
                s.player_x, s.player_y,
            ])
            y_list.append(s.accion)

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.int64)

        # Normalización Z-score (esencial para estabilidad del MLP)
        self.scaler_mean = np.mean(X, axis=0)
        self.scaler_std = np.std(X, axis=0)
        self.scaler_std[self.scaler_std < 1e-4] = 1.0  # Evitar división por cero
        X_norm = (X - self.scaler_mean) / self.scaler_std

        # Convertir a tensores PyTorch
        X_tensor = torch.tensor(X_norm, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        batch_size = min(64, max(16, len(X) // 10))
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.modelo = RedMLP(in_features=12, hidden_dim=128, out_classes=9)
        self.modelo.train()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.modelo.parameters(), lr=0.003)

        epochs = 40
        for _ in range(epochs):
            total_loss = 0.0
            for bx, by in loader:
                optimizer.zero_grad()
                out = self.modelo(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        self.modelo.eval()
        self.modelo_entrenado = True

        # Guardar checkpoint para persistir el modelo entrenado
        ruta = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modelo_mlp.pth")
        torch.save({
            "model_state_dict": self.modelo.state_dict(),
            "scaler_mean": self.scaler_mean,
            "scaler_std": self.scaler_std
        }, ruta)

        loss_avg = total_loss / max(1, len(loader))
        return f"¡MLP Entrenado! {len(X)} muestras | Loss: {loss_avg:.4f} | ¡Listo para Auto [A]!"

    def decision_auto(self) -> Tuple[int, int]:
        """
        Calcula la dirección (dx, dy) usando la predicción del modelo MLP.
        """
        if not self.modelo_entrenado or self.modelo is None or self.scaler_mean is None or self.scaler_std is None:
            return 0, 0

        assert self.perseguidor is not None
        f: List[float] = []
        for i in range(2):
            if i < len(self.pelotas):
                f.extend(self._feat_pelota(self.pelotas[i]))
            else:
                f.extend([0.0, 0.0, 0.0, 0.0])

        x_raw = np.array([
            f[0], f[1], f[2], f[3],
            f[4], f[5], f[6], f[7],
            float(self.perseguidor.x - self.jugador.centerx),
            float(self.perseguidor.y - self.jugador.centery),
            float(self.jugador.centerx),
            float(self.jugador.centery),
        ], dtype=np.float32)

        # Normalizar con los mismos parámetros del entrenamiento
        x_norm = (x_raw - self.scaler_mean) / self.scaler_std
        t_in = torch.tensor(x_norm, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            logits = self.modelo(t_in)
            accion = int(torch.argmax(logits, dim=1).item())

        return accion_a_dirs(accion)

    # ---- menú ----
    def _dibujar_menu(self, msg: str = "") -> None:
        self.pantalla.fill(self.NEGRO)
        titulo = self.fuente.render("MENÚ — Esquivar con MLP", True, self.BLANCO)
        self.pantalla.blit(titulo, (self.w // 2 - titulo.get_width() // 2, int(35 * self.scale)))

        lineas = [
            "M / Botón [A]  — Manual (juega con Stick o Teclado para recolectar datos)",
            "T / Botón [X]  — Entrenar Red Neuronal (MLP)",
            "A / Botón [Y]  — Modo Auto (el MLP esquiva solo)",
            "C / Botón [RB] — Exportar dataset a CSV",
            "F              — Pantalla completa",
            "Q / Botón [B]  — Salir",
            "",
        ]
        x0 = int(60 * self.scale)
        y = int(95 * self.scale)
        for op in lineas:
            self.pantalla.blit(self.fuente_chica.render(op, True, self.BLANCO), (x0, y))
            y += self.fuente_chica.get_linesize() + 4

        y += 6
        # Estado del control
        ctrl_txt = f"Control: {self.joystick.get_name()}" if self.joystick else "Control: No detectado (Usa teclado)"
        ctrl_col = (100, 220, 255) if self.joystick else self.GRIS
        self.pantalla.blit(self.fuente_chica.render(ctrl_txt, True, ctrl_col), (x0, y))
        y += self.fuente_chica.get_linesize() + 4

        # Memoria y estado de la IA
        self.pantalla.blit(
            self.fuente_chica.render(f"Memoria datos: {len(self.datos)} muestras", True, self.GRIS),
            (x0, y),
        )
        y += self.fuente_chica.get_linesize() + 4
        estado_ia = "ENTRENADO (Listo para [A])" if self.modelo_entrenado else "NO ENTRENADO (Juega en [M] y presiona [T])"
        color_ia = (120, 255, 120) if self.modelo_entrenado else (255, 140, 140)
        self.pantalla.blit(
            self.fuente_chica.render(f"Modelo MLP: {estado_ia}", True, color_ia),
            (x0, y),
        )
        if msg:
            self.pantalla.blit(
                self.fuente_chica.render(msg, True, self.AMARILLO),
                (x0, y + self.fuente_chica.get_linesize() + 10),
            )
        pygame.display.flip()

    def mostrar_menu(self) -> None:
        msg = ""
        esperando = True
        while esperando and self.corriendo:
            self._dibujar_menu(msg)
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    self.corriendo = False
                    esperando = False
                elif e.type in (pygame.JOYDEVICEADDED, pygame.JOYDEVICEREMOVED):
                    self._detectar_control()
                elif e.type == pygame.JOYBUTTONDOWN:
                    if e.button == 0:  # Botón A -> Manual
                        self.modo_auto = False
                        self.datos.clear()
                        self.colisiones = 0
                        self._reset()
                        esperando = False
                    elif e.button == 3:  # Botón Y -> Auto
                        if not self.modelo_entrenado:
                            msg = "Primero debes entrenar el MLP con [T] o [X]."
                        else:
                            self.modo_auto = True
                            self._reset()
                            esperando = False
                    elif e.button == 2:  # Botón X -> Entrenar
                        msg = self.entrenar_modelo()
                    elif e.button in (4, 5):  # LB o RB -> Exportar CSV
                        msg = self.exportar_csv()
                    elif e.button == 1:  # Botón B -> Salir
                        self.corriendo = False
                        esperando = False
                elif e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_m:
                        self.modo_auto = False
                        self.datos.clear()
                        self.colisiones = 0
                        self._reset()
                        esperando = False
                    elif e.key == pygame.K_a:
                        if not self.modelo_entrenado:
                            msg = "Primero debes entrenar el MLP con [T] tras jugar en [M]."
                        else:
                            self.modo_auto = True
                            self._reset()
                            esperando = False
                    elif e.key == pygame.K_t:
                        msg = self.entrenar_modelo()
                    elif e.key == pygame.K_c:
                        msg = self.exportar_csv()
                    elif e.key == pygame.K_f:
                        self._toggle_fs()
                    elif e.key == pygame.K_q:
                        self.corriendo = False
                        esperando = False

    def _toggle_fs(self) -> None:
        self._fullscreen = not self._fullscreen
        if self._fullscreen:
            info = pygame.display.Info()
            self.pantalla = pygame.display.set_mode(
                (info.current_w, info.current_h), pygame.FULLSCREEN
            )
        else:
            self.pantalla = pygame.display.set_mode(self._ventana)
        self.w, self.h = self.pantalla.get_size()
        self.fondo = pygame.transform.smoothscale(self.fondo, (self.w, self.h))
        self._reset()

    def _dibujar(self) -> None:
        self.pantalla.blit(self.fondo, (0, 0))
        pygame.draw.line(
            self.pantalla, (255, 255, 255), (self.w // 2, self.h // 2 - 10), (self.w // 2, self.h // 2 + 10), 1
        )
        pygame.draw.line(
            self.pantalla, (255, 255, 255), (self.w // 2 - 10, self.h // 2), (self.w // 2 + 10, self.h // 2), 1
        )

        self.frame_tick += 1
        if self.frame_tick >= 8:
            self.frame_i = (self.frame_i + 1) % len(self.frames)
            self.frame_tick = 0
        self.pantalla.blit(self.frames[self.frame_i], self.jugador.topleft)

        for i, p in enumerate(self.pelotas):
            self.pantalla.blit(self.ball_img if i % 2 == 0 else self.ball_img2, p.rect.topleft)

        assert self.perseguidor is not None
        pygame.draw.circle(
            self.pantalla,
            (220, 60, 60),
            (int(self.perseguidor.x), int(self.perseguidor.y)),
            int(self.perseguidor.radio),
        )
        pygame.draw.circle(
            self.pantalla,
            (255, 200, 200),
            (int(self.perseguidor.x), int(self.perseguidor.y)),
            int(self.perseguidor.radio),
            3,
        )

        modo = "AUTO" if self.modo_auto else "MANUAL"
        hud = f"[{modo}] {len(self.datos)}  golpes={self.colisiones}"
        self.pantalla.blit(self.fuente_chica.render(hud, True, self.AMARILLO), (10, 8))

    def jugar(self) -> None:
        reloj = pygame.time.Clock()
        while self.corriendo:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    self.corriendo = False
                    return
                if e.type in (pygame.JOYDEVICEADDED, pygame.JOYDEVICEREMOVED):
                    self._detectar_control()
                if e.type == pygame.JOYBUTTONDOWN:
                    # Botón B (1) o Start (7 / 6) para pausar y volver al menú
                    if e.button in (1, 6, 7):
                        return
                if e.type == pygame.KEYDOWN:
                    if e.key in (pygame.K_ESCAPE, pygame.K_p):
                        return
                    if e.key == pygame.K_q:
                        self.corriendo = False
                        return
                    if e.key == pygame.K_f:
                        self._toggle_fs()

            keys = pygame.key.get_pressed()
            if self.modo_auto:
                dx, dy = self.decision_auto()
            else:
                dx, dy = self._obtener_direccion(keys)
                self.registrar(dx, dy)

            self._mover_jugador(dx, dy)
            self._mover_pelotas()
            self._mover_perseguidor()
            self.frames_vivos += 1

            if self._colision():
                self.colisiones += 1
                self._reset()

            self._dibujar()
            pygame.display.flip()
            reloj.tick(45)

    def loop(self) -> None:
        while self.corriendo:
            self.mostrar_menu()
            if self.corriendo:
                self.jugar()
        pygame.quit()


if __name__ == "__main__":
    Juego().loop()
