from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

# Защита от несовместимости версий
try:
    from moviepy.editor import VideoFileClip, concatenate_videoclips
except ModuleNotFoundError:
    from moviepy import VideoFileClip, concatenate_videoclips

ProgressCB = Optional[Callable[[float], None]]
FLAT_STD_THRESHOLD = 2.0  # порог «плоского» участка без текстуры


def format_timecode(seconds: float) -> str:
    """Секунды → строка ЧЧ:ММ:СС.сс."""
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:05.2f}"


@dataclass
class ROI:
    """Прямоугольная область интереса на кадре."""
    x: int
    y: int
    width: int
    height: int

    def slice(self) -> Tuple[slice, slice]:
        """Возвращает срезы numpy для вырезания области. (диапазон индексов)"""
        return (slice(self.y, self.y + self.height),
                slice(self.x, self.x + self.width))


@dataclass
class Interval:
    """Интервал времени [start, end] в секундах."""
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict:
        """Сериализация для JSON-отчёта."""
        return {
            "start_sec": round(self.start, 2),
            "end_sec": round(self.end, 2),
            "duration_sec": round(self.duration, 2),
            "start_timecode": format_timecode(self.start),
            "end_timecode": format_timecode(self.end),
        }


@dataclass
class DetectionConfig:
    """Параметры детекции и сегментации."""
    roi: Optional[ROI] = None
    similarity_threshold: float = 0.6
    min_pause_duration_sec: float = 5.0
    sample_step_sec: float = 0.5


# Работа с видео
def _open_capture(video_path: str) -> cv2.VideoCapture:
    """Открывает видеофайл."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(
            f"Не удалось открыть видео: {video_path}. Попробуйте латиницу."
        )
    return cap


def _get_fps(cap: cv2.VideoCapture) -> float:
    """Возвращает FPS с проверкой на корректность."""
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        return 25.0
    return fps


def get_video_duration(video_path: str) -> float:
    """Длительность видео в секундах."""
    cap = _open_capture(video_path)
    try:
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / _get_fps(cap)
    finally:
        cap.release()


def grab_frame(video_path: str, time_sec: float) -> np.ndarray:
    """Извлекает кадр на указанной секунде."""
    cap = _open_capture(video_path)
    try:
        fps = _get_fps(cap)
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(time_sec * fps)))
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok:
        raise ValueError(f"Не удалось прочитать кадр на {time_sec:.1f} с.")
    return frame


# Детекция логотипа
def load_template(path: str) -> np.ndarray:
    """Загружает образец логотипа (через np.fromfile — работает с кириллицей)."""
    try:
        buf = np.fromfile(path, dtype=np.uint8)
        template = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception:
        template = None
    if template is None:
        raise FileNotFoundError(f"Не удалось загрузить образец: {path}")
    return template


def extract_roi(frame: np.ndarray, roi: ROI) -> np.ndarray:
    """Извлекает область ROI из кадра."""
    y_slice, x_slice = roi.slice()
    return frame[y_slice, x_slice]


def compute_similarity(frame_roi: np.ndarray, template: np.ndarray) -> float:
    """Оценка схожести ROI с образцом: [-1, 1]."""
    if (frame_roi.shape[0] < template.shape[0] or
            frame_roi.shape[1] < template.shape[1]):
        raise ValueError("Область поиска меньше образца — увеличьте ROI.")
    # «Плоский» участок без текстуры не может совпасть с содержательным образцом
    if (frame_roi.std() < FLAT_STD_THRESHOLD
            and template.std() >= FLAT_STD_THRESHOLD):
        return -1.0
    result = cv2.matchTemplate(frame_roi, template, cv2.TM_CCOEFF_NORMED)
    return float(cv2.minMaxLoc(result)[1])


def analyze_video(
    video_path: str,
    template: np.ndarray,
    roi: ROI,
    sample_step_sec: float = 0.5,
    progress_cb: ProgressCB = None,
) -> Tuple[List[float], List[float]]:
    """Проходит по видео и возвращает (times, scores) для каждой проверяемой точки."""
    cap = _open_capture(video_path)
    try:
        fps = _get_fps(cap)
        duration = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / fps
        step = max(1, round(fps * sample_step_sec))
        times: List[float] = []
        scores: List[float] = []
        idx = 0
        while True:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)  # перескок через step кадров
            ok, frame = cap.read()
            if not ok:
                break
            times.append(idx / fps)
            scores.append(compute_similarity(extract_roi(frame, roi), template))
            if progress_cb:
                progress_cb(min(1.0, (idx / fps) / duration))
            idx += step
    finally:
        cap.release()
    return times, scores


# Сегментация
def binarize(scores: List[float], threshold: float) -> List[bool]:
    """Оценки - True/False по порогу (True = логотип есть)."""
    return [s >= threshold for s in scores]


def find_absence_intervals(
    times: List[float],
    presence: List[bool],
    min_pause_sec: float,
) -> List[Interval]:
    """Находит интервалы без логотипа длиннее min_pause_sec."""
    raw: List[Interval] = []
    start: Optional[float] = None
    for t, present in zip(times, presence):
        if not present and start is None:
            start = t
        elif present and start is not None:
            raw.append(Interval(start, t))
            start = None
    if start is not None:
        raw.append(Interval(start, times[-1]))
    return [iv for iv in raw if iv.duration >= min_pause_sec]


def build_keep_intervals(
    ad_intervals: List[Interval],
    total_duration: float,
) -> List[Interval]:
    """Строит сегменты для сохранения — дополнение к рекламным паузам."""
    keep: List[Interval] = []
    cursor = 0.0
    for ad in sorted(ad_intervals, key=lambda iv: iv.start):
        if ad.start > cursor:
            keep.append(Interval(cursor, ad.start))
        cursor = max(cursor, ad.end)
    if cursor < total_duration:
        keep.append(Interval(cursor, total_duration))
    return keep


# Монтаж
def _subclip(clip, start: float, end: float):
    """Совместимость moviepy 1.x/2.x для обрезки клипа."""
    return (clip.subclipped if hasattr(clip, "subclipped") else clip.subclip)(start, end)


def cut_ads(
    video_path: str,
    keep_intervals: List[Interval],
    output_path: str,
    progress_cb: ProgressCB = None,
) -> float:
    """Вырезает рекламу и сохраняет итоговое видео. Возвращает его длительность."""
    if not keep_intervals:
        raise ValueError("Список сегментов для сохранения пуст.")
    source = VideoFileClip(video_path)
    pieces = []
    try:
        for i, iv in enumerate(keep_intervals):
            end = min(iv.end, source.duration)  # защита от выхода за границы
            if end > iv.start:
                pieces.append(_subclip(source, iv.start, end))
            if progress_cb:
                progress_cb((i + 1) / len(keep_intervals))
        if not pieces:
            raise ValueError("Все сегменты оказались нулевой длины.")
        final = concatenate_videoclips(pieces)
        try:
            final.write_videofile(
                output_path, codec="libx264", audio_codec="aac", logger=None,
            )
            return final.duration
        finally:
            final.close()
    finally:
        for p in pieces:
            p.close()
        source.close()


# Отчёт
def save_report(
    path: str,
    ad_intervals: List[Interval],
    keep_intervals: List[Interval],
    total_duration: float,
    final_duration: float,
) -> None:
    """Сохраняет отчёт о рекламных паузах в JSON."""
    report = {
        "total_duration_sec": round(total_duration, 2),
        "final_duration_sec": round(final_duration, 2),
        "removed_duration_sec": round(total_duration - final_duration, 2),
        "ad_breaks": [iv.to_dict() for iv in ad_intervals],
        "kept_segments": [iv.to_dict() for iv in keep_intervals],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)