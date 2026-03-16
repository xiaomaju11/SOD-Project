from __future__ import annotations

from project_kernel_common import RESULTS_DIR, set_plot_defaults, write_summary
from project_kernel_part1 import generate_part1_results
from project_kernel_part2 import generate_part2_results
from project_kernel_part3 import generate_part3_results


def generate_all_results() -> dict[str, object]:
    set_plot_defaults()
    RESULTS_DIR.mkdir(exist_ok=True)

    part1_output = generate_part1_results()
    part2_output = generate_part2_results()
    part3_output = generate_part3_results(part1_output["problem"])

    summary_lines = (
        part1_output["summary_lines"]
        + part2_output["summary_lines"]
        + part3_output["summary_lines"]
    )
    summary_file = write_summary(summary_lines)

    return {
        "part1": part1_output,
        "part2": part2_output,
        "part3": part3_output,
        "summary_lines": summary_lines,
        "results_dir": RESULTS_DIR,
        "summary_file": summary_file,
        "pdf_files": sorted(path.name for path in RESULTS_DIR.glob("*.pdf")),
    }
