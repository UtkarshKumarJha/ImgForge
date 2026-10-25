from scripts.report_tables import ci_cell, per_seed


def test_ci_cell_parses_the_log_string():
    assert ci_cell({"f1_macro_ci": "[0.6391, 0.6993]"}) == "0.639-0.699"


def test_ci_cell_survives_a_missing_or_broken_value():
    assert ci_cell({}) == "—"
    assert ci_cell({"f1_macro_ci": "[nan, nan]"}) == "—"
    assert ci_cell({"f1_macro_ci": "[0.1]"}) == "—"


def test_per_seed_lists_every_seed_with_its_own_interval():
    grouped = {("efficientnet_b0", "ela"): [
        {"arch": "efficientnet_b0", "input_mode": "ela", "seed": "2",
         "f1_macro_tuned": "0.84", "f1_macro_ci": "[0.82, 0.86]"},
        {"arch": "efficientnet_b0", "input_mode": "ela", "seed": "1",
         "f1_macro_tuned": "0.83", "f1_macro_ci": "[0.81, 0.85]"},
    ]}
    lines = per_seed("efficientnet_b0", grouped)

    assert "| ela | 1 | 0.8300 | 0.810-0.850 |" in lines
    assert "| ela | 2 | 0.8400 | 0.820-0.860 |" in lines
    assert lines.index("| ela | 1 | 0.8300 | 0.810-0.850 |") < \
        lines.index("| ela | 2 | 0.8400 | 0.820-0.860 |")
