# Báo cáo khoa học (ACM Conference Proceedings)

Thư mục này chứa báo cáo khoa học của đề tài
**Dự báo Giá Bất động sản Việt Nam và Hoa Kỳ bằng Học máy Tích hợp**,
được soạn theo mẫu chính thức **ACM Conference Proceedings (sample-sigconf)**
sử dụng lớp tài liệu `acmart`.

## Cấu trúc

| File | Mô tả |
| --- | --- |
| `report.tex` | File LaTeX chính (định dạng ACM `sigconf`). |
| `references.bib` | Tài liệu tham khảo (BibTeX). |
| `figures/` | Các hình minh hoạ (PDF) và bảng số liệu (`metrics.json`, `boosting_results.csv`). |
| `gen_figures.py` | Script Python sinh hình + chỉ số cho phần Việt Nam và Boosting cơ bản trên Ames. |
| `gen_topkaggle.py` | Script Python tái lập pipeline Top-Kaggle (Box-Cox + Stacking đa họ + Blending). |
| `Makefile` | Build tự động bằng `latexmk` + `xelatex`. |

## Yêu cầu môi trường

### LaTeX
- TeX Live $\geq 2021$ (`xelatex`, `latexmk`, `bibtex`).
- Các gói: `acmart` (texlive-publishers), `polyglossia`, `algorithm2e`,
  `subcaption`, `booktabs`, `amsmath`, `amssymb`.

Trên Ubuntu/Debian:

```bash
sudo apt-get install -y \
  texlive-latex-base texlive-latex-recommended texlive-latex-extra \
  texlive-publishers texlive-fonts-recommended texlive-fonts-extra \
  texlive-bibtex-extra texlive-xetex texlive-lang-other \
  texlive-science biber latexmk
```

### Python (chỉ cần khi muốn sinh lại hình)
- Python 3.10+
- `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `scipy`,
  `xgboost`, `lightgbm`, `catboost`.

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy \
            xgboost lightgbm catboost
```

## Cách build

Từ thư mục `report/`:

```bash
make            # sinh report.pdf (xelatex + bibtex)
make figures    # tuỳ chọn: tái sinh hình từ dữ liệu thô
make clean      # xoá file phụ trợ
```

Hoặc thủ công:

```bash
xelatex report.tex
bibtex  report
xelatex report.tex
xelatex report.tex
```

## Ghi chú về tiếng Việt

`acmart` đã nạp sẵn font Libertine, vốn hỗ trợ đầy đủ ký tự
tiếng Việt khi compile bằng **xelatex** (hoặc `lualatex`).
Việc khai báo ngôn ngữ thực hiện qua gói `polyglossia`:

```latex
\usepackage{polyglossia}
\setdefaultlanguage{vietnamese}
```

Nếu cần xuất bản chính thức trên ACM, hãy xoá tuỳ chọn `nonacm`
trong `\documentclass[sigconf,nonacm]{acmart}` và điền thông tin
bản quyền (`\setcopyright`, `\acmDOI`, …) theo email xác nhận từ
ACM.

## Nguồn dữ liệu

- **Việt Nam**: 1 000 tin đăng cào từ `batdongsan.com.vn` (xem
  `BDSVietNam/scraper_batdongsan.py`).
- **Hoa Kỳ**: bộ Ames Housing thuộc cuộc thi Kaggle House Prices
  (`Kaggle/train.csv`, `Kaggle/test.csv`).

## Tham chiếu mẫu ACM

Báo cáo dựa trên mẫu chính thức:
<https://www.overleaf.com/latex/templates/acm-conference-proceedings-primary-article-template/wbvnghjbzwpc>
(`sample-sigconf` của bộ `acmart`).
