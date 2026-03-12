#!/bin/bash
# Export all report assets after training

echo "Exporting report assets..."

python -c "
from src.report.export_assets import export_all_assets
export_all_assets()
print('Done!')
"

echo ""
echo "Report assets saved to: report_assets/"
