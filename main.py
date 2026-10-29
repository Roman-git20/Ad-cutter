from __future__ import annotations

import base64
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np
from nicegui import events, run, ui

import core


@dataclass
class State:
    """Состояние GUI: пути к файлам, кадр, ROI и результаты анализа."""
    video_path: Optional[str] = None
    template_path: Optional[str] = None
    template: Optional[np.ndarray] = None
    roi: Optional[core.ROI] = None
    frame: Optional[np.ndarray] = None
    drag_start: Optional[Tuple[int, int]] = None
    last_rect: Optional[Tuple[int, int, int, int]] = None
    times: Optional[List[float]] = None
    scores: Optional[List[float]] = None
    ad_intervals: List[core.Interval] = field(default_factory=list)
    keep_intervals: List[core.Interval] = field(default_factory=list)
    video_duration: float = 0.0


# Утилиты
def frame_to_data_url(frame_bgr: np.ndarray) -> str:
    """Кодирует BGR-кадр в PNG data URL для отображения в браузере."""
    ok, buf = cv2.imencode(".png", frame_bgr)
    if not ok:
        raise RuntimeError("Не удалось закодировать кадр.")
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


def save_image_unicode(path: str, image: np.ndarray) -> bool:
    """Сохраняет изображение с поддержкой кириллицы в пути."""
    ok, buf = cv2.imencode(os.path.splitext(path)[1], image)
    if ok:
        buf.tofile(path)
    return ok


def roi_with_margin(x: int, y: int, w: int, h: int,
                    fw: int, fh: int, margin: int = 15) -> core.ROI:
    """Строит ROI с запасом margin пикселей, не выходя за границы кадра."""
    sx, sy = max(0, x - margin), max(0, y - margin)
    return core.ROI(sx, sy, min(fw - sx, w + 2 * margin),
                    min(fh - sy, h + 2 * margin))


def rect_from_drag(start: Tuple[int, int], end: Tuple[int, int],
                   fw: int, fh: int) -> Tuple[int, int, int, int]:
    """Преобразует drag мыши в (x, y, w, h), обрезая по границам кадра."""
    x1, y1 = start
    x2, y2 = end
    x, y = max(0, min(x1, x2)), max(0, min(y1, y2))
    return x, y, min(fw - x, abs(x2 - x1)), min(fh - y, abs(y2 - y1))


def download_file(path: str) -> None:
    """Совместимость ui.download для NiceGUI 1.x и 2.x."""
    if hasattr(ui.download, "file"):
        ui.download.file(path)
    else:
        ui.download(path)


# Страница
@ui.page("/")
def main_page() -> None:
    ui.page_title("Ad Cutter")
    state = State()

    # Повторяющиеся CSS-классы вынесены в переменные
    card_cls = "w-full"
    title_cls = "text-lg font-bold text-blue-8"
    hint_cls = "text-xs text-gray-500"
    btn_full = "w-full"

    def notify(message: str, kind: str = "info") -> None:
        """Всплывающее уведомление в правом верхнем углу."""
        ui.notify(message, type=kind, position="top-right")

    # Шапка
    with ui.header().classes("bg-blue-9 text-white items-center"):
        ui.label("🎬 Ad Cutter").classes("text-xl font-bold")
        ui.label("Вырезание рекламы по исчезновению логотипа").classes(
            "text-sm opacity-80 ml-4")

    with ui.row().classes("w-full max-w-7xl mx-auto gap-4 p-4 items-start"):


        # Левая панель — управление
        with ui.column().classes("w-96 gap-3"):

            # 1. Загрузка файлов
            with ui.card().classes(card_cls):
                ui.label("📁 1. Файлы").classes(title_cls)
                ui.label("Видео:").classes("text-sm mt-2")
                video_upload = ui.upload(
                    label="Загрузить видео (.mp4, .avi, .mkv)",
                    auto_upload=True, max_files=1,
                ).props('accept=".mp4,.avi,.mkv,.mov" flat').classes(btn_full)
                video_name = ui.label("Файл не выбран").classes(hint_cls)

                ui.label("Кадр на секунде:").classes("text-sm mt-2")
                time_input = ui.number(
                    value=5.0, min=0, step=0.5, format="%.1f").classes(btn_full)

                show_frame_btn = ui.button("👁 Показать кадр").classes(btn_full).props("color=primary")

                ui.separator().classes("my-2")
                ui.label("Образец логотипа:").classes("text-sm font-bold mt-1")

                ui.label("Способ 1: выделите на кадре и сохраните").classes(hint_cls)
                save_template_btn = ui.button(
                    "💾 Сохранить выделение как образец"
                ).classes(btn_full).props("color=positive").set_enabled(False)

                ui.label("Способ 2: загрузите готовый файл").classes("text-xs text-gray-500 mt-2")
                template_upload = ui.upload(
                    label="Загрузить PNG/JPG с логотипом",
                    auto_upload=True, max_files=1,
                ).props('accept=".png,.jpg,.jpeg" flat').classes(btn_full)

                match_status = ui.label("").classes("text-xs mt-1 font-bold whitespace-pre-line")

                set_roi_btn = ui.button(
                    "🎯 Задать область поиска (ROI)"
                ).classes(btn_full).props("color=secondary").set_enabled(False)

                template_status = ui.label("Образец не задан").classes(
                    "text-xs text-gray-500 mt-1 whitespace-pre-line")

            # 2. Параметры детекции
            with ui.card().classes(card_cls):
                ui.label("⚙️ 2. Параметры детекции").classes(title_cls)

                threshold_input = ui.number(
                    "Порог схожести (0..1)", value=0.6, min=0.0, max=1.0,
                    step=0.05, format="%.2f",
                ).classes(btn_full)

                min_pause_input = ui.number(
                    "Минимальная длительность паузы (сек)", value=5.0, min=0.0,
                    step=0.5, format="%.1f",
                ).classes(btn_full)

                step_input = ui.number(
                    "Шаг проверки кадров (сек)", value=0.5, min=0.05,
                    step=0.1, format="%.2f",
                ).classes(btn_full)

            # 3. Анализ и нарезка
            with ui.card().classes(card_cls):
                ui.label("🚀 3. Анализ и вырезание").classes(title_cls)

                analyze_btn = ui.button("🔍 Анализировать видео").classes(btn_full).props("color=primary").set_enabled(False)
                analyze_progress = ui.linear_progress(value=0, show_value=False).classes(btn_full)
                summary_label = ui.label("").classes("text-sm whitespace-pre-line")

                cut_btn = ui.button("✂️ Вырезать рекламу").classes(btn_full).props("color=orange").set_enabled(False)
                cut_progress = ui.linear_progress(value=0, show_value=False).classes(btn_full)

                download_container = ui.column().classes("w-full mt-2")

        # Правая панель — предпросмотр и результаты
        with ui.column().classes("flex-1 gap-3 min-w-0"):

            # Кадр с выделением ROI
            with ui.card().classes(card_cls):
                ui.label("Кадр видео").classes(title_cls)
                ui.label("Зажмите левую кнопку мыши и выделите область логотипа.").classes(hint_cls)
                frame_placeholder = ui.column().classes("w-full")  # сюда добавим frame_view позже
                coords_label = ui.label("Выделение: —").classes(hint_cls)

            # График
            with ui.card().classes(card_cls):
                ui.label("📈 График детекции").classes(title_cls)
                plot_container = ui.column().classes("w-full")

            # Таблица пауз
            with ui.card().classes(card_cls):
                ui.label("📋 Найденные рекламные паузы").classes(title_cls)
                table = ui.table(
                    columns=[
                        {"name": "idx", "label": "#", "field": "idx", "align": "left"},
                        {"name": "start", "label": "Начало", "field": "start", "align": "left"},
                        {"name": "end", "label": "Конец", "field": "end", "align": "left"},
                        {"name": "duration", "label": "Длительность", "field": "duration", "align": "left"},
                    ],
                    rows=[], row_key="idx",
                ).classes("w-full").props("dense flat bordered")


    # Логика
    def redraw_plot() -> None:
        """Перерисовывает график схожести с линией порога и зонами рекламы."""
        plot_container.clear()
        with plot_container:
            if state.times is None or len(state.times) == 0:
                ui.label("График появится после анализа видео").classes("text-gray-400 italic p-4")
                return
            with ui.matplotlib(figsize=(9, 3.2)).figure as fig:
                ax = fig.gca()
                ax.plot(state.times, state.scores, linewidth=1, color="#1976d2", label="Схожесть")
                thr = float(threshold_input.value)
                ax.axhline(thr, color="red", linestyle="--", linewidth=1.5, label=f"Порог ({thr:.2f})")
                # Подсветка зон, где схожесть ниже порога — там реклама
                ax.fill_between(state.times, 0, state.scores,
                                where=[s < thr for s in state.scores],
                                color="red", alpha=0.2)
                ax.set_ylim(-0.05, 1.05)
                ax.set_xlabel("Время, сек")
                ax.set_ylabel("Схожесть")
                ax.grid(True, alpha=0.3)
                ax.legend(loc="lower right", fontsize=8)

    def _current_roi() -> Optional[core.ROI]:
        """Возвращает актуальный ROI: явно заданный или из последнего выделения."""
        if state.roi is not None:
            return state.roi
        if state.last_rect is not None:
            x, y, w, h = state.last_rect
            return core.ROI(x, y, w, h)
        return None

    def _check_match() -> None:
        """Проверяет схожесть загруженного логотипа с областью на текущем кадре."""
        if state.template is None or state.frame is None:
            return
        roi = _current_roi()
        if roi is None:
            match_status.set_text("🎯 Выделите мышью область поиска на кадре, чтобы проверить совпадение")
            match_status.classes(replace="text-xs mt-1 font-bold text-blue-6")
            return
        try:
            score = core.compute_similarity(core.extract_roi(state.frame, roi), state.template)
        except ValueError as ex:
            match_status.set_text(f"❌ Ошибка проверки: {ex}")
            match_status.classes(replace="text-xs mt-1 font-bold text-red-6")
            return
        pct = score * 100
        if score >= 0.75:  # хорошее совпадение
            match_status.set_text(f"✅ Совпадение: {pct:.1f}% — отлично")
            match_status.classes(replace="text-xs mt-1 font-bold text-green-6")
            notify(f"Совпадение: {pct:.1f}%", "positive")
        elif score >= 0.5:  # приемлемо, но можно лучше
            match_status.set_text(
                f"⚠️ Совпадение: {pct:.1f}% — приемлемо.\n"
                f"Попробуйте выделить плотнее или создать образец вручную.")
            match_status.classes(replace="text-xs mt-1 font-bold text-orange-6")
            notify(f"Совпадение: {pct:.1f}%", "warning")
        else:  # плохое — рекомендуем пересоздать образец
            match_status.set_text(
                f"❌ Совпадение: {pct:.1f}% — НЕ подходит!\n"
                f"Создайте образец вручную: покажите кадр → выделите логотип → 💾")
            match_status.classes(replace="text-xs mt-1 font-bold text-red-6")
            notify(f"❌ Плохое совпадение: {pct:.1f}%", "negative")

    def _reset_analysis_state() -> None:
        """Сбрасывает результаты анализа при загрузке нового видео."""
        state.times = state.scores = None
        state.ad_intervals, state.keep_intervals = [], []
        state.frame = state.template = state.roi = state.last_rect = None

    # Обработчики загрузки
    async def on_video_upload(e: events.UploadEventArguments) -> None:
        """Загрузка видео, чтение метаданных и сброс состояния."""
        os.makedirs("data", exist_ok=True)
        path = os.path.join("data", e.file.name)
        with open(path, "wb") as f:
            f.write(await e.file.read())
        try:
            duration = core.get_video_duration(path)
        except Exception as ex:
            notify(f"Ошибка: {ex}", "negative")
            return
        state.video_path = path
        state.video_duration = duration
        _reset_analysis_state()
        video_name.set_text(f"📹 {e.file.name} ({core.format_timecode(duration)})")
        time_input.max = duration
        template_status.set_text("Образец не задан")
        match_status.set_text("")
        frame_view.set_source("")
        coords_label.set_text("Выделение: —")
        table.rows = []
        summary_label.set_text("")
        redraw_plot()
        _refresh_buttons()
        notify(f"Видео загружено: {core.format_timecode(duration)}", "positive")

    video_upload.on_upload(on_video_upload)

    async def on_template_upload(e: events.UploadEventArguments) -> None:
        """Загрузка готового образца логотипа + сразу проверка совпадения."""
        os.makedirs("data", exist_ok=True)
        path = os.path.join("data", "loaded_" + e.file.name)
        with open(path, "wb") as f:
            f.write(await e.file.read())
        try:
            state.template = core.load_template(path)
        except Exception as ex:
            notify(f"Ошибка загрузки логотипа: {ex}", "negative")
            return
        state.template_path = path
        th, tw = state.template.shape[:2]
        template_status.set_text(f"✅ Загружен: {e.file.name} ({tw}×{th} пикс.)")
        if state.frame is None:
            match_status.set_text("⚠️ Покажите кадр видео и выделите область поиска")
            match_status.classes(replace="text-xs mt-1 font-bold text-orange-6")
        else:
            _check_match()
        notify(f"Логотип загружен: {tw}×{th} пикс.", "positive")
        _refresh_buttons()

    template_upload.on_upload(on_template_upload)

    # Кадр и выделение
    def show_frame() -> None:
        """Показывает кадр на указанной секунде."""
        if state.video_path is None:
            notify("Сначала загрузите видео", "warning")
            return
        try:
            state.frame = core.grab_frame(state.video_path, float(time_input.value))
        except Exception as ex:
            notify(f"Ошибка: {ex}", "negative")
            return
        frame_view.set_source(frame_to_data_url(state.frame))
        coords_label.set_text("Выделите мышью логотип на кадре")
        state.last_rect = None
        if state.template is not None:
            _check_match()
        _refresh_buttons()

    show_frame_btn.on_click(show_frame)

    def on_mouse(e: events.MouseEventArguments) -> None:
        """Обработка перетаскивания мыши для выделения области."""
        if state.frame is None:
            return
        fh, fw = state.frame.shape[:2]
        if e.type == "mousedown":  # начало выделения
            state.drag_start = (int(e.image_x), int(e.image_y))
            state.last_rect = None
        elif e.type in ("mousemove", "mouseup") and state.drag_start is not None:
            end = (int(e.image_x), int(e.image_y))
            x, y, w, h = rect_from_drag(state.drag_start, end, fw, fh)
            if e.type == "mousemove":  # показываем текущие координаты
                coords_label.set_text(f"Выделение: x={x}, y={y}, w={w}, h={h}")
            else:  # mouseup — фиксируем выделение
                if w > 5 and h > 5:
                    state.last_rect = (x, y, w, h)
                    coords_label.set_text(f"✅ Выделено: {w}×{h} пикс.")
                else:
                    coords_label.set_text("⚠️ Слишком маленькая область")
                state.drag_start = None
                _refresh_buttons()

    # frame_view создаётся здесь, потому что on_mouse уже определён выше
    with frame_placeholder:
        frame_view = ui.interactive_image(
            "",
            cross=False,
            on_mouse=lambda e: on_mouse(e),
            events=["mousedown", "mousemove", "mouseup"],
        ).classes("w-full max-h-[400px] bg-gray-100 border")

    # Создание образца и ROI

    def create_template() -> None:
        """Способ 1: сохраняет выделение как образец и строит ROI с запасом."""
        if state.frame is None or state.last_rect is None:
            notify("Сначала покажите кадр и выделите логотип", "warning")
            return
        x, y, w, h = state.last_rect
        fh, fw = state.frame.shape[:2]
        os.makedirs("data", exist_ok=True)
        path = os.path.join("data", "logo_sample.png")
        save_image_unicode(path, state.frame[y:y + h, x:x + w])
        state.template_path = path
        state.template = core.load_template(path)
        state.roi = roi_with_margin(x, y, w, h, fw, fh)
        template_status.set_text(
            f"✅ Образец {w}×{h}, ROI {state.roi.width}×{state.roi.height}")
        notify(f"Образец сохранён: {w}×{h} пикс.", "positive")
        _check_match()
        _refresh_buttons()

    save_template_btn.on_click(create_template)

    def set_roi_from_selection() -> None:
        """Способ 2: задаёт ROI по выделению для уже загруженного логотипа."""
        if state.frame is None or state.last_rect is None:
            notify("Сначала покажите кадр и выделите область поиска", "warning")
            return
        if state.template is None:
            notify("Сначала загрузите образец логотипа", "warning")
            return
        x, y, w, h = state.last_rect
        fh, fw = state.frame.shape[:2]
        state.roi = roi_with_margin(x, y, w, h, fw, fh)
        template_status.set_text(
            f"✅ Образец задан\n🎯 ROI: x={state.roi.x}, y={state.roi.y}, "
            f"{state.roi.width}×{state.roi.height}")
        notify(f"ROI задан: {state.roi.width}×{state.roi.height} пикс.", "positive")
        _check_match()
        _refresh_buttons()

    set_roi_btn.on_click(set_roi_from_selection)

    # Анализ
    async def start_analysis() -> None:
        """Запускает анализ видео в фоновом потоке."""
        if state.video_path is None or state.template is None or state.roi is None:
            notify("Задайте видео, образец логотипа и ROI", "warning")
            return
        analyze_btn.disable()
        analyze_progress.value = 0
        notify("Анализ запущен...", "info")
        try:
            times, scores = await run.io_bound(  # не блокирует UI
                core.analyze_video, state.video_path, state.template,
                state.roi, float(step_input.value),
            )
            state.times, state.scores = times, scores
            analyze_progress.value = 1.0
            notify(f"Анализ завершён: {len(times)} точек", "positive")
            recompute_segments()
            redraw_plot()
        except Exception as ex:
            notify(f"Ошибка анализа: {ex}", "negative")
        finally:
            analyze_btn.enable()
            _refresh_buttons()

    analyze_btn.on_click(start_analysis)

    def recompute_segments() -> None:
        """Пересчитывает интервалы рекламы по текущим порогу и мин. длительности."""
        if state.times is None:
            return
        presence = core.binarize(state.scores, float(threshold_input.value))
        state.ad_intervals = core.find_absence_intervals(
            state.times, presence, float(min_pause_input.value))
        total = state.times[-1]
        state.keep_intervals = core.build_keep_intervals(state.ad_intervals, total)
        # Обновление таблицы
        table.rows = [
            {"idx": i + 1,
             "start": core.format_timecode(iv.start),
             "end": core.format_timecode(iv.end),
             "duration": f"{iv.duration:.1f} с"}
            for i, iv in enumerate(state.ad_intervals)
        ]
        # Сводка
        total_ad = sum(iv.duration for iv in state.ad_intervals)
        summary_label.set_text(
            f"Исходная: {core.format_timecode(total)}\n"
            f"Пауз: {len(state.ad_intervals)}  |  Реклама: {core.format_timecode(total_ad)}\n"
            f"Итоговая: {core.format_timecode(total - total_ad)}")
        _refresh_buttons()

    # Нарезка
    async def start_cut() -> None:
        """Запускает вырезание рекламы и сохранение результата в фоне."""
        if not state.keep_intervals:
            notify("Нет сегментов для сохранения", "warning")
            return
        os.makedirs("output", exist_ok=True)
        output_path = os.path.join("output", "result_no_ads.mp4")
        cut_btn.disable()
        cut_progress.value = 0
        notify("Начинаю вырезание...", "info")
        try:
            final_duration = await run.io_bound(
                core.cut_ads, state.video_path, state.keep_intervals, output_path,
            )
            cut_progress.value = 1.0
            # Отчёт сохраняем отдельно от видео
            report_path = os.path.join("output", "report.json")
            core.save_report(report_path, state.ad_intervals, state.keep_intervals,
                             state.times[-1], final_duration)
            notify(f"✅ Готово! Длительность: {core.format_timecode(final_duration)}",
                   "positive")
            # Кнопки скачивания появляются только после успешной нарезки
            download_container.clear()
            with download_container:
                ui.label("📥 Скачать результат:").classes("text-sm font-bold mt-2")
                ui.button("⬇️ Скачать видео без рекламы",
                          on_click=lambda: download_file(output_path)
                          ).classes(btn_full).props("color=positive")
                ui.button("📄 Скачать отчёт (JSON)",
                          on_click=lambda: download_file(report_path)
                          ).classes(btn_full).props("color=secondary")
        except Exception as ex:
            notify(f"Ошибка: {ex}", "negative")
        finally:
            cut_btn.enable()
            _refresh_buttons()

    cut_btn.on_click(start_cut)

    # Управление доступностью кнопок

    def _refresh_buttons() -> None:
        """Обновляет enabled-состояние кнопок по текущему состоянию."""
        show_frame_btn.set_enabled(state.video_path is not None)
        save_template_btn.set_enabled(
            state.frame is not None and state.last_rect is not None)
        set_roi_btn.set_enabled(
            state.frame is not None and state.last_rect is not None
            and state.template is not None)
        analyze_btn.set_enabled(
            state.video_path is not None and state.template is not None
            and state.roi is not None)
        cut_btn.set_enabled(len(state.keep_intervals) > 0)

    # Пересчёт при изменении параметров
    threshold_input.on_value_change(
        lambda: (recompute_segments(), redraw_plot()) if state.times is not None else None)
    min_pause_input.on_value_change(
        lambda: recompute_segments() if state.times is not None else None)

    _refresh_buttons()
    redraw_plot()


# Точка входа
if __name__ in {"__main__", "__mp_main__"}:
    ui.run(
        title="Logo Ad Cutter",
        host="0.0.0.0",
        port=8080,
        reload=False,
        show=False,
        native=False,
    )
