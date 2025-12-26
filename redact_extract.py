import os
import sys
import time
import argparse
import threading
import multiprocessing as mp
from tqdm import tqdm
import pymupdf  # PyMuPDF


# ============================================================
# PAUSE CONTROLLER
# ============================================================

class PauseController:
    def __init__(self):
        self.pause_event = threading.Event()
        self.pause_event.set()
        self.stop = False

    def toggle(self):
        if self.pause_event.is_set():
            self.pause_event.clear()
            print("\n⏸ Paused (press 'p' to resume)")
        else:
            self.pause_event.set()
            print("\n▶ Resumed")

    def wait(self):
        self.pause_event.wait()


def keyboard_listener(ctrl: PauseController):
    while not ctrl.stop:
        key = sys.stdin.read(1).lower()
        if key == "p":
            ctrl.toggle()
        elif key == "q":
            ctrl.stop = True
            ctrl.pause_event.set()
            print("\n⛔ Abort requested")
            break


# ============================================================
# TEXT PROCESSING
# ============================================================

def group_words_into_lines(words, line_tol):
    words.sort(key=lambda w: (w[1], w[0]))
    lines, current, current_top = [], [], None

    for w in words:
        top = w[1]
        if current_top is None or abs(top - current_top) <= line_tol:
            current.append(w)
            current_top = top if current_top is None else (current_top + top) * 0.5
        else:
            lines.append(current)
            current = [w]
            current_top = top

    if current:
        lines.append(current)
    return lines


def build_line_text(line_words, space_unit, min_spaces):
    line_words.sort(key=lambda w: w[0])
    parts = [line_words[0][3]]
    prev_x1 = line_words[0][2]
    fs = line_words[0][4]

    for x0, _, x1, text, _ in line_words[1:]:
        gap = x0 - prev_x1
        if gap > 0:
            parts.append(" " * max(min_spaces, int(gap / space_unit)))
        parts.append(text)
        prev_x1 = max(prev_x1, x1)

    return "".join(parts), line_words[0][0], line_words[0][1], fs


# ============================================================
# MULTIPROCESS WORKER (TOP-LEVEL)
# ============================================================

def process_page(args):
    pdf_path, page_no, line_tol, space_unit, min_spaces = args
    doc = pymupdf.open(pdf_path)
    page = doc[page_no]

    words_raw = page.get_text("words")
    page_width = page.rect.width
    words = []

    for x0, y0, x1, y1, text, *_ in words_raw:
        if not text.strip():
            continue
        fs = max(6.0, y1 - y0)
        if fs < 5 or x0 > page_width:
            continue
        words.append((x0, y0, x1, text, fs))

    doc.close()

    lines = group_words_into_lines(words, line_tol)
    out = []

    for lw in lines:
        text, x0, top, fs = build_line_text(lw, space_unit, min_spaces)
        if text.strip():
            out.append((text, x0, top + fs * 0.85, fs))

    return page_no, out


# ============================================================
# MAIN PIPELINE
# ============================================================

def make_side_by_side(input_pdf, output_pdf, line_tol, space_unit, min_spaces):
    pause_ctrl = PauseController()
    threading.Thread(target=keyboard_listener, args=(pause_ctrl,), daemon=True).start()

    src = pymupdf.open(input_pdf)
    out = pymupdf.open()
    total_pages = len(src)
    results = [None] * total_pages

    worker_args = [
        (input_pdf, i, line_tol, space_unit, min_spaces)
        for i in range(total_pages)
    ]

    # ------------------ EXTRACT ------------------
    start = time.time()
    done = 0

    with mp.Pool(mp.cpu_count()) as pool:
        with tqdm(total=total_pages, desc="Extracting", unit="page") as pbar:
            for page_no, data in pool.imap_unordered(process_page, worker_args):
                if pause_ctrl.stop:
                    pool.terminate()
                    pool.join()
                    raise KeyboardInterrupt("Aborted")

                pause_ctrl.wait()
                results[page_no] = data
                done += 1
                pbar.update(1)

                if done % 10 == 0 or done == total_pages:
                    eta = int(((time.time() - start) / done) * (total_pages - done))
                    pbar.set_postfix_str(f"ETA {eta//60:02d}:{eta%60:02d}")

    # ------------------ RENDER ------------------
    start = time.time()
    done = 0

    with tqdm(total=total_pages, desc="Rendering", unit="page") as pbar:
        for i, src_page in enumerate(src):
            if pause_ctrl.stop:
                raise KeyboardInterrupt("Aborted")

            pause_ctrl.wait()

            rect = src_page.rect
            w, h = rect.width, rect.height
            new_page = out.new_page(width=2 * w, height=h)
            new_page.show_pdf_page(pymupdf.Rect(0, 0, w, h), src, i)

            for txt, x0, y, fs in results[i]:
                new_page.insert_text(
                    pymupdf.Point(w + x0, y),
                    txt,
                    fontsize=fs,
                    fontname="helv",
                    overlay=True
                )

            done += 1
            pbar.update(1)
            eta = int(((time.time() - start) / done) * (total_pages - done))
            pbar.set_postfix_str(f"ETA {eta//60:02d}:{eta%60:02d}")

    out.save(output_pdf)
    src.close()
    out.close()
    pause_ctrl.stop = True


# ============================================================
# CLI
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_pdf")
    ap.add_argument("-o", "--output")
    ap.add_argument("--line-tol", type=float, default=2.0)
    ap.add_argument("--space-unit", type=float, default=3.0)
    ap.add_argument("--min-spaces", type=int, default=1)
    args = ap.parse_args()

    if not os.path.exists(args.input_pdf):
        raise FileNotFoundError(args.input_pdf)

    output = args.output or args.input_pdf.replace(".pdf", "_side_by_side.pdf")

    make_side_by_side(
        args.input_pdf,
        output,
        args.line_tol,
        args.space_unit,
        args.min_spaces
    )

    print(f"\nWrote: {output}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
