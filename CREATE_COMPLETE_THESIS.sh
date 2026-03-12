#!/bin/bash
# Script to create complete thesis in Word format
# Combines v12 content with new Phases 6-11

echo "=== TDA Thesis Builder ==="
echo "Creating complete thesis document..."

# Create output directory
mkdir -p thesis_output

# Step 1: Convert all markdown sections to Word
echo "Step 1: Converting markdown sections..."

cd thesis_expansion

# Convert each section
for file in SECTION_*.md; do
    if [ -f "$file" ]; then
        echo "Converting $file..."
        pandoc "$file" -o "${file%.md}.docx" \
            --reference-doc=../thesis_template.docx \
            --number-sections \
            --toc
    fi
done

echo ""
echo "✅ All sections converted to Word format"
echo ""
echo "FILES READY:"
echo "  - thesis_expansion/SECTION_1_INTRODUCTION.docx"
echo "  - thesis_expansion/SECTION_6_TEXT.docx (Phase 1)"
echo "  - thesis_expansion/SECTION_7_TEXT.docx (Phase 2 - BREAKTHROUGH)"
echo "  - thesis_expansion/SECTION_8_TEXT.docx (Phase 3)"
echo "  - thesis_expansion/SECTION_9_TEXT.docx (Phase 4)"
echo "  - thesis_expansion/SECTION_10_TEXT.docx (Phase 5)"
echo "  - thesis_expansion/SECTION_11_TEXT.docx (Phase 6)"
echo "  - thesis_expansion/SECTION_12_CONCLUSION.docx"
echo ""
echo "NEXT STEPS:"
echo "1. Open your v12 PDF"
echo "2. Create new Word document"
echo "3. Copy sections from v12 (keep Sections 2-5 as baseline)"
echo "4. Insert the .docx files above as new sections"
echo "5. Insert figures from phase*/figure_*.pdf folders"
echo ""
echo "See THESIS_INTEGRATION_COMPLETE_GUIDE.md for detailed instructions"
