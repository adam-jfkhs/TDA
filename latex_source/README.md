# LaTeX Thesis - Compilation Instructions

## What's In This Folder

```
thesis_latex/
├── thesis_main.tex           ← Main document (compile this)
├── sections/                 ← Individual section files
│   ├── sec02_methodology.tex (stub - from your v12)
│   ├── sec03_results.tex (stub - from your v12)
│   ├── sec04_analysis.tex (stub - from your v12)
│   ├── sec05_preliminary.tex (stub - from your v12)
│   ├── sec06_intraday.tex (stub - needs Phase 1 content)
│   ├── sec07_sector.tex ← COMPLETE (key finding, all feedback applied)
│   ├── sec08_variants.tex (stub - needs Phase 3 content)
│   ├── sec09_crossmarket.tex (stub - needs Phase 4 content)
│   ├── sec10_ml.tex ← COMPLETE (ML section, consistent guardrails)
│   ├── sec11_theory.tex (stub - needs Phase 6 content)
│   └── sec12_conclusion.tex (stub - needs conclusion content)
├── tables/                   ← Authoritative tables
│   ├── table_sector_authoritative.tex ← COMPLETE
│   ├── table_ml_authoritative.tex ← COMPLETE
│   └── (other tables - need to be created)
├── figures/                  ← Figure PDFs
│   └── (symlink to ../thesis_expansion/phase*/ folders)
└── references.bib            ← Bibliography file

```

## How to Compile

### Option 1: Using pdflatex (Recommended)

```bash
cd /home/user/TDA/thesis_latex

# Compile (run 2-3 times for cross-references)
pdflatex thesis_main.tex
pdflatex thesis_main.tex

# With bibliography
pdflatex thesis_main.tex
bibtex thesis_main
pdflatex thesis_main.tex
pdflatex thesis_main.tex

# Output: thesis_main.pdf
```

### Option 2: Using Overleaf (Easiest)

1. Create account at overleaf.com
2. Upload all files (zip this folder first)
3. Set main document to `thesis_main.tex`
4. Click "Recompile"
5. Download PDF

### Option 3: Using latexmk (Automated)

```bash
latexmk -pdf thesis_main.tex
```

## Current Status

### ✅ Complete Sections (Ready to Compile)
- Title page
- Abstract
- Executive Summary
- Introduction (partial)
- **Section 7: Sector-Specific** (COMPLETE, all feedback applied)
- **Section 10: ML Integration** (COMPLETE, consistent guardrails)
- Authoritative tables for Sections 7, 10

### ⚠️ Stub Sections (Need Content Added)
- Sections 2-6, 8-9, 11-12 (placeholder stubs inserted)
- Other authoritative tables

### 📋 What You Need To Do

#### Quick Test (Compile As-Is)
```bash
cd thesis_latex
pdflatex thesis_main.tex
```

This will compile with stub sections. You'll get a PDF showing the structure.

#### Add Remaining Content

**For each stub section** (e.g., `sec06_intraday.tex`):
1. Open the corresponding markdown file from `thesis_expansion/SECTION_6_TEXT.md`
2. Convert to LaTeX format (or just copy text, LaTeX handles plain text)
3. Replace stub content in `sections/sec06_intraday.tex`

**For figures**:
1. Create symlink to existing figures:
   ```bash
   ln -s ../../thesis_expansion/phase1_intraday figures/phase1
   ln -s ../../thesis_expansion/phase2_sector figures/phase2
   # etc.
   ```

2. Or copy figures:
   ```bash
   cp -r ../thesis_expansion/phase*/figure_*.pdf figures/
   ```

## Feedback Applied (Checklist)

- [x] Replace "breakthrough" with "key finding" / "central result"
- [x] Replace "profitable" with "positive risk-adjusted performance"
- [x] ONE authoritative table per phase (source of truth)
- [x] Explicit paragraph on WHY sector-specific works
- [x] Consistent ML guardrail sentence in Section 10
- [x] Conservative AUC interpretation (≈0.52 = "barely above random")
- [ ] Add "What I would do next" section (Section 12)
- [x] Avoid overstating claims
- [x] Keep failure-first structure
- [x] Maintain intellectual honesty

## Common LaTeX Commands

### Insert Figure
```latex
\begin{figure}[h]
\centering
\includegraphics[width=0.8\textwidth]{figures/phase2/figure_7_2.pdf}
\caption{Your caption here}
\label{fig:correlation-cv}
\end{figure}
```

### Reference Figure
```latex
See Figure~\ref{fig:correlation-cv} for details.
```

### Insert Table
```latex
\input{tables/table_sector_authoritative}
```

### Math Equations
```latex
\begin{equation}
\text{CV} = \frac{\sigma}{\mu}
\end{equation}
```

### Citations
```latex
\citep{Gidea2018}  % (Gidea & Katz, 2018)
\citet{Gidea2018}  % Gidea & Katz (2018)
```

## Troubleshooting

### Error: "File not found"
- Check that stub files exist in `sections/` folder
- Run the stub creation script below

### Error: "Missing \begin{document}"
- File encoding issue, resave as UTF-8

### Error: "Undefined control sequence"
- Check for special characters (%, $, &, #)
- Escape them: `\%`, `\$`, `\&`, `\#`

### Figures don't show
- Check file path in `\includegraphics{}`
- Use relative path: `figures/phase2/figure_7_2.pdf`

## Next Steps

1. **Test compile** (should work with stubs):
   ```bash
   cd thesis_latex
   pdflatex thesis_main.tex
   open thesis_main.pdf  # Mac
   # or xdg-open thesis_main.pdf  # Linux
   ```

2. **Add content** to remaining stubs (Sections 2-6, 8-9, 11-12)

3. **Copy figures** to `figures/` folder

4. **Create bibliography** (`references.bib` with all citations)

5. **Final compilation** with bibliography

## Questions?

See `../THESIS_INTEGRATION_COMPLETE_GUIDE.md` for detailed content guidelines.

See `../QUICK_REFERENCE_DATA.md` for all numerical values to verify.

---

**Quality Target**: 9.2/10 → 9.5/10 after editing

**Estimated Time**: 6-8 hours to complete all sections

**Output**: 80-90 page professional thesis PDF
