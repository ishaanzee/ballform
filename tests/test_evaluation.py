import json

import pytest

from app.evaluation import evaluate, match_shots, read_labels, summarize

HEADER = "job_id,release_s,outcome,shot_type,distance_ft,zone,shooter,source,notes\n"


def shot(release, outcome="made", shot_type="jump shot", distance=None, zone=None):
    return {"release_s": release, "outcome": outcome, "shot_type": shot_type,
            "game": {"metrics": {"shot_distance_ft": distance, "shot_zone": zone}, "players": {}}}


def write_job(jobs, job_id, shots):
    (jobs / job_id).mkdir(parents=True)
    (jobs / job_id / "result.json").write_text(json.dumps({"shots": shots}))


def test_labels_are_validated_with_the_line_number(tmp_path):
    path = tmp_path / "labels.csv"
    path.write_text(HEADER + "a,1.0,made,jump shot,,,,,\na,2.0,swish,,,,,,\n")
    with pytest.raises(ValueError, match="line 3: outcome"):
        read_labels(path)


def test_matching_is_one_to_one_by_release_time(tmp_path):
    labels = read_labels_from(tmp_path, "a,1.0,made,,,,,,\na,1.4,missed,,,,,,\n")
    matches = match_shots(labels, [shot(1.3), shot(5.0)], tolerance_s=.75)
    paired = [(m.label.release_s, m.shot["release_s"]) for m in matches if m.label and m.shot]
    assert paired == [(1.4, 1.3)]
    assert [m.label.release_s for m in matches if not m.shot] == [1.0]
    assert [m.shot["release_s"] for m in matches if not m.label] == [5.0]


def read_labels_from(tmp_path, rows):
    path = tmp_path / "labels.csv"
    path.write_text(HEADER + rows)
    return read_labels(path)


def test_summary_scores_outcomes_types_distance_and_zones(tmp_path):
    write_job(tmp_path, "a", [shot(1.0, "likely made", "layup or dunk", 3.0, "paint"),
                              shot(4.0, "made", "jump shot", 25.0, "above_break_three"),
                              shot(8.0, "unknown", "jump shot", 18.0, "midrange")])
    write_job(tmp_path, "empty", [shot(2.0)])
    labels = read_labels_from(tmp_path, "a,1.1,made,dunk,2.0,paint,,,\n"
                              "a,4.1,missed,jump shot,24.0,above_break_three,,,\n"
                              "a,8.0,made,jump shot,20.0,midrange,,,\n"
                              "empty,,,,,,,,\n")
    summary = summarize(evaluate(labels, tmp_path, .75))
    assert summary["detection"] == {"recall": "3/3 (100%)", "precision": "3/4 (75%)"}
    assert summary["outcome"]["coverage"] == "2/3 (67%)"
    assert summary["outcome"]["accuracy_when_called"] == "1/2 (50%)"
    assert summary["outcome"]["misses_called_made"] == 1
    assert summary["shot_type"]["accuracy"] == "3/3 (100%)"
    assert summary["distance_ft"]["mean_abs_error"] == pytest.approx(4 / 3, abs=.01)
    assert summary["distance_ft"]["bias"] == pytest.approx(0.0)
    assert summary["zone"]["accuracy"] == "3/3 (100%)"


def test_a_missing_job_is_reported_not_fatal(tmp_path):
    summary = summarize(evaluate(read_labels_from(tmp_path, "gone,1.0,made,,,,,,\n"), tmp_path, .75))
    assert summary["clips_failed"] == 1 and summary["labeled_shots"] == 0
