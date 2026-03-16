"""
SmartOMR Grader - Module cham diem bai thi
==========================================
Chuc nang:
- Load dap an chuan tu file (.txt / .json / .csv)
- So sanh bai lam voi dap an
- Tinh diem: tong diem, tru diem, nhom cau
- Xuat bao cao chi tiet
- Ve anh annotated mau dung/sai

Dinh dang file dap an (.txt):
    # SmartOMR Answer Key
    # name = Ten de thi
    # total_score = 10
    # correct_score = 0.1        (moi cau dung = 0.1 diem)
    # wrong_penalty = 0          (tru diem neu sai, 0 = khong tru)
    1: A
    2: B
    3: C
    ...
    120: D

Dinh dang JSON:
    {
      "config": {
        "name": "Ten de thi",
        "total_score": 10,
        "correct_score": 0.1,
        "wrong_penalty": 0
      },
      "answers": {
        "1": "A", "2": "B", ...
      }
    }
"""

import os
import json
import csv
import cv2

CHOICE_LABELS = ["A", "B", "C", "D"]

# Mau annotate: BGR
COLOR_CORRECT = (0, 200, 0)       # Xanh la -> dung
COLOR_WRONG   = (0, 0, 220)       # Do -> sai
COLOR_CORRECT_ANS = (255, 150, 0) # Xanh duong -> dap an dung (khi thi sinh sai)
COLOR_BLANK   = (0, 165, 255)     # Cam -> bo trong (dap an dung)
COLOR_NEUTRAL = (180, 180, 180)   # Xam -> khong co dap an


# ============================================================
# LOAD ANSWER KEY
# ============================================================

def load_answer_key(filepath):
    """
    Load dap an chuan tu file.

    Ho tro .txt, .json, .csv

    Returns:
        dict: {
            'answers': {1: 'A', 2: 'B', ...},
            'config': {
                'name': str,
                'total_score': float,
                'correct_score': float,
                'wrong_penalty': float
            }
        }
    """
    ext = os.path.splitext(filepath)[1].lower()
    if ext == '.json':
        return _load_json(filepath)
    elif ext == '.csv':
        return _load_csv(filepath)
    else:
        return _load_txt(filepath)


def _parse_config_comment(line, config):
    """Parse dong comment config: # key = value"""
    if '=' not in line or not line.startswith('#'):
        return
    parts = line[1:].split('=', 1)
    key = parts[0].strip().lower().replace(' ', '_')
    val = parts[1].strip()
    if key in ('total_score', 'correct_score', 'wrong_penalty'):
        try:
            config[key] = float(val)
        except ValueError:
            pass
    elif key == 'name':
        config['name'] = val


def _fill_correct_score(answers, config):
    """Tu tinh correct_score neu chua co."""
    if config.get('correct_score') is None and answers:
        n = len([v for v in answers.values() if v is not None])
        if n > 0:
            config['correct_score'] = round(config['total_score'] / n, 6)


def _load_txt(filepath):
    answers = {}
    config = {
        'name': os.path.splitext(os.path.basename(filepath))[0],
        'total_score': 10.0,
        'correct_score': None,
        'wrong_penalty': 0.0
    }
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('#'):
                _parse_config_comment(line, config)
                continue
            # Parse: "1: A", "1. A", "1 A", "1=A"
            for sep in (':', '.', '=', '\t', ' '):
                if sep in line:
                    parts = line.split(sep, 1)
                    q_str = parts[0].strip()
                    a_str = parts[1].strip().upper() if len(parts) > 1 else ''
                    if q_str.isdigit():
                        q_num = int(q_str)
                        answers[q_num] = a_str if a_str in CHOICE_LABELS else None
                    break
    _fill_correct_score(answers, config)
    return {'answers': answers, 'config': config}


def _load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    raw_answers = data.get('answers', {})
    answers = {}
    for k, v in raw_answers.items():
        try:
            q_num = int(k)
            a = str(v).upper()
            answers[q_num] = a if a in CHOICE_LABELS else None
        except (ValueError, TypeError):
            pass
    config = {
        'name': os.path.splitext(os.path.basename(filepath))[0],
        'total_score': 10.0,
        'correct_score': None,
        'wrong_penalty': 0.0
    }
    for key in ('name', 'total_score', 'correct_score', 'wrong_penalty'):
        if key in data.get('config', {}):
            v = data['config'][key]
            config[key] = float(v) if key != 'name' else str(v)
    _fill_correct_score(answers, config)
    return {'answers': answers, 'config': config}


def _load_csv(filepath):
    answers = {}
    config = {
        'name': os.path.splitext(os.path.basename(filepath))[0],
        'total_score': 10.0,
        'correct_score': None,
        'wrong_penalty': 0.0
    }
    with open(filepath, 'r', encoding='utf-8', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or str(row[0]).startswith('#'):
                continue
            if len(row) >= 2:
                q_str = str(row[0]).strip()
                a_str = str(row[1]).strip().upper()
                if q_str.isdigit():
                    answers[int(q_str)] = a_str if a_str in CHOICE_LABELS else None
    _fill_correct_score(answers, config)
    return {'answers': answers, 'config': config}


def save_answer_key(answers, filepath, config=None, name=None):
    """
    Luu file dap an.

    Args:
        answers: dict {q_num: 'A'/'B'/'C'/'D'/None}
        filepath: duong dan (.txt / .json)
        config: dict config (optional)
        name: ten de thi
    """
    n = len([v for v in answers.values() if v is not None])
    if config is None:
        config = {
            'name': name or 'De thi',
            'total_score': 10.0,
            'correct_score': round(10.0 / n, 4) if n else 0,
            'wrong_penalty': 0.0
        }
    ext = os.path.splitext(filepath)[1].lower()
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    if ext == '.json':
        data = {
            'config': config,
            'answers': {str(k): (v or '-') for k, v in sorted(answers.items())}
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    else:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("# SmartOMR Answer Key\n")
            f.write(f"# name = {config.get('name', 'De thi')}\n")
            f.write(f"# total_score = {config.get('total_score', 10)}\n")
            f.write(f"# correct_score = {config.get('correct_score', 0.1)}\n")
            f.write(f"# wrong_penalty = {config.get('wrong_penalty', 0)}\n")
            f.write("# Format: so_cau: dap_an  (A/B/C/D  hoac - khi bo trong)\n#\n")
            for q in sorted(answers.keys()):
                a = answers[q] or '-'
                f.write(f"{q}: {a}\n")
    print(f"  Saved answer key: {filepath}")


def create_template(output_path, n_questions=120, total_score=10.0):
    """Tao file dap an mau de dien."""
    answers = {q: None for q in range(1, n_questions + 1)}
    config = {
        'name': 'Ten de thi',
        'total_score': total_score,
        'correct_score': round(total_score / n_questions, 4),
        'wrong_penalty': 0.0
    }
    save_answer_key(answers, output_path, config=config)


# ============================================================
# GRADING
# ============================================================

def grade(student_answers, answer_key_data):
    """
    Cham diem bai thi.

    Args:
        student_answers: dict {q_num: 'A'/'B'/'C'/'D'/None}
        answer_key_data: dict tu load_answer_key()

    Returns:
        GradingResult
    """
    key_answers = answer_key_data['answers']
    config = answer_key_data['config']
    correct_score = float(config.get('correct_score') or 0)
    wrong_penalty = float(config.get('wrong_penalty') or 0)
    total_score_max = float(config.get('total_score', 10))

    details = {}
    n_correct = n_wrong = n_blank = n_no_key = 0

    all_qs = sorted(set(list(student_answers.keys()) + list(key_answers.keys())))
    for q in all_qs:
        s_ans = student_answers.get(q)       # None = bo trong
        c_ans = key_answers.get(q)            # None = khong co trong dap an

        if c_ans is None:
            status = 'no_key'; earned = 0.0; n_no_key += 1
        elif s_ans is None:
            status = 'blank';  earned = 0.0; n_blank += 1
        elif s_ans == c_ans:
            status = 'correct'; earned = correct_score; n_correct += 1
        else:
            status = 'wrong';  earned = -wrong_penalty; n_wrong += 1

        details[q] = {
            'student': s_ans,
            'correct': c_ans,
            'status': status,
            'earned': earned
        }

    raw = sum(d['earned'] for d in details.values())
    final = round(max(0.0, min(raw, total_score_max)), 4)

    return GradingResult(details, final, total_score_max,
                         n_correct, n_wrong, n_blank, n_no_key, config)


class GradingResult:
    """Ket qua cham diem."""

    def __init__(self, details, score, score_max,
                 n_correct, n_wrong, n_blank, n_no_key, config):
        self.details = details
        self.score = score
        self.score_max = score_max
        self.n_correct = n_correct
        self.n_wrong = n_wrong
        self.n_blank = n_blank
        self.n_no_key = n_no_key
        self.config = config

    @property
    def score_pct(self):
        return round(self.score / self.score_max * 100, 1) if self.score_max else 0.0

    @property
    def n_graded(self):
        return self.n_correct + self.n_wrong + self.n_blank

    def print_report(self):
        cfg = self.config
        sep = '=' * 60
        print(f"\n{sep}")
        print(f"  KET QUA CHAM DIEM")
        print(f"  De thi : {cfg.get('name', '')}")
        print(sep)
        print(f"  DIEM   : {self.score:.2f} / {self.score_max}  ({self.score_pct}%)")
        print(sep)
        penalty_str = f"  (-{cfg.get('wrong_penalty',0)}/cau)" if cfg.get('wrong_penalty', 0) > 0 else ""
        print(f"  Dung   : {self.n_correct:3d} cau  (+{cfg.get('correct_score', 0):.4f}/cau)")
        print(f"  Sai    : {self.n_wrong:3d} cau{penalty_str}")
        print(f"  Bo trong: {self.n_blank:3d} cau")
        print(sep)

        # Bang chi tiet
        n_q_per_col = 30
        n_cols = 4
        print()
        for col in range(n_cols):
            start = col * n_q_per_col + 1
            end = start + n_q_per_col
            max_q = max(self.details.keys()) if self.details else 0
            if start > max_q:
                break
            print(f"  Cot {col+1} (Cau {start}-{min(end-1, max_q)}):")
            row_buf = "  "
            for q in range(start, end):
                d = self.details.get(q)
                if d is None:
                    continue
                status = d['status']
                s = d['student'] or '-'
                mk = {'correct': '+', 'wrong': 'X', 'blank': 'o', 'no_key': ' '}.get(status, ' ')
                row_buf += f"{q:3d}:{s}{mk} "
                if (q - start + 1) % 10 == 0:
                    print(row_buf)
                    row_buf = "  "
            if row_buf.strip():
                print(row_buf)
        print()

        # Liet ke cau sai
        wrongs = [(q, d) for q, d in self.details.items() if d['status'] == 'wrong']
        blanks = [(q, d) for q, d in self.details.items() if d['status'] == 'blank']
        if wrongs or blanks:
            print(f"  Chi tiet sai/bo trong:")
            for q, d in sorted(wrongs):
                print(f"    Cau {q:3d}:  Chon {d['student']}  |  Dung: {d['correct']}  [SAI]")
            for q, d in sorted(blanks):
                print(f"    Cau {q:3d}:  Bo trong       |  Dung: {d['correct']}  [TRONG]")
        print(sep)

    def save_report(self, filepath):
        """Luu bao cao cham diem ra file .txt."""
        cfg = self.config
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("SmartOMR - Ket qua cham diem\n")
            f.write(f"De thi : {cfg.get('name', '')}\n")
            f.write(f"{'='*50}\n\n")
            f.write(f"DIEM   : {self.score:.2f} / {self.score_max}  ({self.score_pct}%)\n")
            f.write(f"Dung   : {self.n_correct} cau\n")
            f.write(f"Sai    : {self.n_wrong} cau\n")
            f.write(f"Bo trong: {self.n_blank} cau\n\n")
            f.write(f"{'='*50}\n")
            f.write(f"{'Cau':>4}  {'Bai lam':>7}  {'Dap an':>6}  {'Ket qua':>8}  {'Diem':>6}\n")
            f.write(f"{'-'*40}\n")
            for q in sorted(self.details.keys()):
                d = self.details[q]
                if d['status'] == 'no_key':
                    continue
                s = d['student'] or '-'
                c = d['correct'] or '-'
                label = {'correct': 'Dung', 'wrong': 'SAI',
                         'blank': 'Bo trong'}.get(d['status'], '')
                f.write(f"{q:4d}  {s:>7}  {c:>6}  {label:>8}  {d['earned']:>+6.3f}\n")
            f.write(f"\n{'='*50}\n")
            f.write(f"TONG DIEM: {self.score:.2f}/{self.score_max}  ({self.score_pct}%)\n")
        print(f"  Saved report: {filepath}")

    def to_dict(self):
        return {
            'score': self.score,
            'score_max': self.score_max,
            'score_pct': self.score_pct,
            'n_correct': self.n_correct,
            'n_wrong': self.n_wrong,
            'n_blank': self.n_blank,
            'config': self.config,
            'details': {str(q): d for q, d in self.details.items()}
        }


# ============================================================
# ANNOTATE ANH CO CHAM DIEM
# ============================================================

def draw_graded_annotated(image, grid, answers, grading_result,
                           num_questions=120, num_choices=4):
    """
    Ve anh annotated co to mau dung/sai.

    Mau:
      Xanh la  = cau dung (bubble dung)
      Do       = cau sai (bubble sai cua thi sinh)
      Xanh duong = dap an dung (hien ra khi thi sinh sai/bo trong)
      Cam      = cau bo trong (hien dap an dung)
      Xam      = khong co dap an
    """
    ann = image.copy()
    q_num = 1

    for col_idx in sorted(grid.keys()):
        for row in grid[col_idx]:
            if q_num > num_questions:
                break

            d = grading_result.details.get(q_num, {})
            status = d.get('status', 'no_key')
            student_ans = d.get('student')
            correct_ans = d.get('correct')
            s_idx = CHOICE_LABELS.index(student_ans) if student_ans in CHOICE_LABELS else -1
            c_idx = CHOICE_LABELS.index(correct_ans) if correct_ans in CHOICE_LABELS else -1

            for ci, (cx, cy, r) in enumerate(row[:num_choices]):
                if status == 'correct' and ci == s_idx:
                    cv2.circle(ann, (cx, cy), r + 6, COLOR_CORRECT, 3)
                    cv2.circle(ann, (cx, cy), r, COLOR_CORRECT, -1)

                elif status == 'wrong':
                    if ci == s_idx:
                        cv2.circle(ann, (cx, cy), r + 6, COLOR_WRONG, 3)
                        cv2.circle(ann, (cx, cy), r, COLOR_WRONG, -1)
                    elif ci == c_idx:
                        cv2.circle(ann, (cx, cy), r + 6, COLOR_CORRECT_ANS, 3)

                elif status == 'blank' and ci == c_idx:
                    cv2.circle(ann, (cx, cy), r + 5, COLOR_BLANK, 2)

                elif status == 'no_key':
                    if ci == s_idx:
                        cv2.circle(ann, (cx, cy), r + 3, COLOR_NEUTRAL, 2)
                    else:
                        cv2.circle(ann, (cx, cy), r + 1, (210, 210, 210), 1)

                else:
                    # blank (cau bi bo trong), khong ro bubble nao
                    cv2.circle(ann, (cx, cy), r + 1, (210, 210, 210), 1)

            # Nhan so cau + marker
            if row:
                mk = {'correct': '+', 'wrong': 'X', 'blank': 'o', 'no_key': ' '}.get(status, ' ')
                col_txt = {
                    'correct': COLOR_CORRECT, 'wrong': COLOR_WRONG,
                    'blank': COLOR_BLANK, 'no_key': COLOR_NEUTRAL
                }.get(status, COLOR_NEUTRAL)
                x_lbl = row[0][0] - row[0][2] - 60
                y_lbl = row[0][1] + 8
                cv2.putText(ann, f"{q_num}{mk}", (x_lbl, y_lbl),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, col_txt, 2)

            q_num += 1

    # Legend
    _draw_legend(ann)
    return ann


def _draw_legend(img):
    """Ve chu thich mau o goc trai duoi."""
    h, w = img.shape[:2]
    x0, y0 = 20, h - 100
    items = [
        (COLOR_CORRECT, "Dung"),
        (COLOR_WRONG,   "Sai (chon)"),
        (COLOR_CORRECT_ANS, "Dap an dung"),
        (COLOR_BLANK,   "Bo trong")
    ]
    for i, (color, label) in enumerate(items):
        y = y0 + i * 22
        cv2.circle(img, (x0 + 10, y), 8, color, -1)
        cv2.putText(img, label, (x0 + 25, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description='SmartOMR Grader')
    sub = p.add_subparsers(dest='command')

    pt = sub.add_parser('template', help='Tao file dap an mau')
    pt.add_argument('--output', '-o', default='answer_key.txt')
    pt.add_argument('--questions', '-n', type=int, default=120)
    pt.add_argument('--score', type=float, default=10.0)

    pc = sub.add_parser('check', help='Kiem tra file dap an')
    pc.add_argument('--key', '-k', required=True)

    args = p.parse_args()

    if args.command == 'template':
        create_template(args.output, args.questions, args.score)

    elif args.command == 'check':
        data = load_answer_key(args.key)
        cfg = data['config']
        filled = sum(1 for v in data['answers'].values() if v)
        print(f"  Ten de   : {cfg.get('name', '')}")
        print(f"  So cau   : {len(data['answers'])}")
        print(f"  Da dien  : {filled}/{len(data['answers'])}")
        print(f"  Diem toi da    : {cfg['total_score']}")
        print(f"  Moi cau dung   : {cfg['correct_score']:.4f}")
        print(f"  Tru diem sai   : {cfg['wrong_penalty']}")
        if filled < len(data['answers']):
            print("  [WARN] Chua dien du dap an!")
    else:
        p.print_help()
