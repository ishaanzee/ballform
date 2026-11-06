import json

import pytest

from app.evaluation import (_court_landmarks, clip_rows, compare, evaluate, main, match_shots, mcnemar_p, read_labels,
                            shot_rows, summarize, wilson)

HEADER = "job_id,release_s,outcome,shot_type,distance_ft,zone,shooter,source,notes\n"


def shot(release, outcome="made", shot_type="jump shot", distance=None, zone=None):
    return {"release_s": release, "outcome": outcome, "shot_type": shot_type,
            "game": {"metrics": {"shot_distance_ft": distance, "shot_zone": zone}, "players": {}}}


def write_job(jobs, job_id, shots, duration_s=None):
    (jobs / job_id).mkdir(parents=True)
    result = {"shots": shots, **({"video": {"duration_s": duration_s}} if duration_s else {})}
    (jobs / job_id / "result.json").write_text(json.dumps(result))


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


def test_shot_type_is_also_scored_as_shot_versus_rim(tmp_path):
    write_job(tmp_path, "a", [shot(1.0, shot_type="floater"), shot(3.0, shot_type="layup or dunk"),
                              shot(5.0, shot_type="floater"), shot(7.0, shot_type="tip")])
    labels = read_labels_from(tmp_path, "a,1.0,made,jump shot,,,,,\na,3.0,made,tip,,,,,\n"
                              "a,5.0,made,layup,,,,,\na,7.0,made,dunk,,,,,\n")
    summary = summarize(evaluate(labels, tmp_path, .75))["shot_type"]
    assert summary["accuracy"] == "0/4 (0%)"
    assert summary["shot_vs_rim"] == "3/4 (75%)"
    assert summary["shot_vs_rim_labeled_vs_predicted"] == {"shot": {"shot": 1}, "rim": {"rim": 2, "shot": 1}}


def test_automatic_court_landmarks_fill_in_only_where_settings_have_none(tmp_path):
    auto = {"standard": "nba", "frame": 30, "points": [], "source": "auto"}
    path = tmp_path / "court_landmarks.json"
    path.write_text(json.dumps({"a": {"court_landmarks": auto}, "b": {"rejected": "no frame agreed"}}))
    own = {"standard": "nba", "time_s": 1.0, "points": []}
    assert _court_landmarks("a", {}, path) == auto
    assert _court_landmarks("a", {"court_landmarks": own}, path) == own
    assert _court_landmarks("b", {}, path) is None
    assert _court_landmarks("c", {}, path) is None
    assert _court_landmarks("a", {}, tmp_path / "missing.json") is None


def test_a_missing_job_is_reported_not_fatal(tmp_path):
    summary = summarize(evaluate(read_labels_from(tmp_path, "gone,1.0,made,,,,,,\n"), tmp_path, .75))
    assert summary["clips_failed"] == 1 and summary["labeled_shots"] == 0


def test_committed_label_files_have_one_value_per_column():
    import csv
    from pathlib import Path

    for path in sorted((Path(__file__).resolve().parents[1] / "eval").glob("*.csv")):
        with path.open(newline="") as handle:
            rows = list(csv.reader(handle))
        widths = {line: len(row) for line, row in enumerate(rows, start=1) if len(row) != len(rows[0])}
        assert not widths, f"{path.name}: rows with the wrong number of columns (an unquoted comma?): {widths}"
        read_labels(path)


def test_the_court_map_cache_key_follows_the_landmarks_and_the_rerun(tmp_path):
    from app.evaluation import _court_map_key

    saved, video = tmp_path / "scoring_inputs.pkl.gz", tmp_path / "input.mp4"
    saved.write_bytes(b"a")
    video.write_bytes(b"v")
    key = _court_map_key({"points": [1, 2]}, saved, video)
    assert key == _court_map_key({"points": [1, 2]}, saved, video)
    assert key != _court_map_key({"points": [1, 3]}, saved, video)
    saved.write_bytes(b"ab")
    assert key != _court_map_key({"points": [1, 2]}, saved, video)


def test_the_game_comes_from_the_label_notes():
    from app.evaluation import game_of

    assert game_of("MIN@DEN 2025-12-25 ESPN event 401809242 Q4 9:55 'Rudy Gobert blocks ...'") == "MIN@DEN 2025-12-25"
    assert game_of("FIXED youtube:dQw4w9WgXcQ pickup run, baseline GoPro") == "youtube:dQw4w9WgXcQ"
    assert game_of("no shot: passes around the arc") == ""


def test_wilson_intervals_match_known_values():
    assert wilson(0, 0) is None
    assert wilson(0, 10) == pytest.approx((0.0, 0.2775), abs=1e-4)
    low, high = wilson(17, 17)
    assert low == pytest.approx(17 / (17 + 1.959964 ** 2)) and high == 1.0
    assert wilson(373, 428) == pytest.approx((0.8365, 0.8999), abs=1e-4)


def test_mcnemar_p_is_the_exact_two_sided_binomial_test():
    assert mcnemar_p(0, 0) == 1.0
    assert mcnemar_p(3, 3) == 1.0
    assert mcnemar_p(5, 0) == pytest.approx(.0625)
    assert mcnemar_p(0, 6) == pytest.approx(.03125)
    assert mcnemar_p(10, 2) == mcnemar_p(2, 10) == pytest.approx(79 / 2048)


def test_summary_adds_intervals_and_the_primary_metrics_and_keeps_the_old_keys(tmp_path):
    write_job(tmp_path, "a", [shot(1.0, "likely made", "layup or dunk"), shot(4.0, "made"), shot(8.0, "unknown")],
              duration_s=40.0)
    write_job(tmp_path, "empty", [shot(2.0)], duration_s=20.0)
    labels = read_labels_from(tmp_path, "a,1.1,made,dunk,,,,,\na,4.1,missed,jump shot,,,,,\n"
                              "a,8.0,made,jump shot,,,,,\nempty,,,,,,,,\n")
    summary = summarize(evaluate(labels, tmp_path, .75))
    assert summary["detection"] == {"recall": "3/3 (100%)", "precision": "3/4 (75%)"}
    assert summary["primary"] == {"found_and_called_right": "1/3 (33%)", "false_shots": 1,
                                  "false_shots_per_minute": 1.0, "footage_minutes": 1.0}
    found = summary["interval_95_pct"]
    assert found["detection"] == {"recall": [43.9, 100.0], "precision": [30.1, 95.4]}
    assert found["primary"]["found_and_called_right"] == [6.1, 79.2]
    assert "distance_ft" not in found  # 0/0 has no interval


def test_false_shots_per_minute_leaves_out_clips_of_unknown_length(tmp_path):
    write_job(tmp_path, "a", [shot(1.0), shot(5.0)], duration_s=30.0)
    write_job(tmp_path, "b", [shot(1.0)])  # no duration in the result and no video to read it from
    labels = read_labels_from(tmp_path, "a,1.0,made,,,,,,\nb,,,,,,,,\n")
    primary = summarize(evaluate(labels, tmp_path, .75))["primary"]
    assert primary["false_shots"] == 2
    assert primary["false_shots_per_minute"] == 2.0 and primary["footage_minutes"] == .5
    assert primary["clips_without_duration"] == 1


def report_of(jobs, labels):
    clips = evaluate(labels, jobs, .75)
    return {"summary": summarize(clips), "shots": shot_rows(clips), "clips": clip_rows(clips)}


def test_compare_counts_the_shots_that_changed_and_pairs_false_shots_by_time(tmp_path):
    labels = read_labels_from(tmp_path, "a,1.0,made,jump shot,,,,,\na,3.0,missed,layup,,,,,\n"
                              "a,5.0,made,dunk,,,,,\na,7.0,missed,jump shot,,,,,\nempty,,,,,,,,\n")
    write_job(tmp_path / "before", "a", [shot(1.0, "made"), shot(3.0, "made", "layup"),
                                         shot(7.0, "missed", "floater"), shot(9.0)])
    write_job(tmp_path / "before", "empty", [shot(2.0)])
    write_job(tmp_path / "after", "a", [shot(1.0, "missed"), shot(3.0, "missed", "layup"), shot(5.0, "made", "dunk"),
                                        shot(7.0, "missed"), shot(11.0)])
    write_job(tmp_path / "after", "empty", [shot(2.1)])  # the same false shot, released a little later
    before, after = report_of(tmp_path / "before", labels), report_of(tmp_path / "after", labels)

    paired = compare(before, after)
    assert paired["labeled_shots_in_both"] == 4 and paired["clips_compared"] == 2
    changes = {name: (m["gains"], m["losses"], m["before"], m["after"]) for name, m in paired["metrics"].items()}
    assert changes == {"found": (1, 0, 3, 4), "outcome_right_found_in_both": (1, 1, 2, 2),
                       "found_and_called_right": (2, 1, 2, 3), "fine_type_right_found_in_both": (1, 0, 2, 3),
                       "false_shots": (1, 1, 2, 2)}
    assert paired["metrics"]["found_and_called_right"]["p"] == pytest.approx(1.0)
    backwards = compare(after, before)["metrics"]
    assert (backwards["found"]["gains"], backwards["found"]["losses"]) == (0, 1)
    assert compare(after, after)["metrics"]["false_shots"] == {"before": 2, "after": 2, "gains": 0, "losses": 0,
                                                               "p": 1.0}


def test_compare_counts_false_shots_only_in_clips_both_reports_scored(tmp_path):
    labels = read_labels_from(tmp_path, "a,1.0,made,,,,,,\nempty,,,,,,,,\n")
    write_job(tmp_path / "before", "a", [shot(1.0)])
    write_job(tmp_path / "before", "empty", [])
    write_job(tmp_path / "after", "a", [shot(1.0)])
    write_job(tmp_path / "after", "empty", [shot(2.0)])
    before, after = report_of(tmp_path / "before", labels), report_of(tmp_path / "after", labels)
    assert compare(before, after)["metrics"]["false_shots"]["losses"] == 1
    # An older report has no clip list, and a clip with no shot row at all is missing from it.
    old = {"summary": before["summary"], "shots": before["shots"]}
    paired = compare(old, after)
    assert paired["metrics"]["false_shots"]["losses"] == 0 and paired["false_shots_not_compared"] == 1


def test_summary_only_reports_and_comparing_saved_reports(tmp_path, capsys):
    labels = tmp_path / "labels.csv"
    labels.write_text(HEADER + "a,1.0,made,,,,,,\na,4.0,missed,,,,,,\n")
    write_job(tmp_path / "jobs", "a", [shot(1.0, "made"), shot(6.0)], duration_s=30.0)
    run = ["--labels", str(labels), "--jobs", str(tmp_path / "jobs")]
    full, short = tmp_path / "full.json", tmp_path / "short.json"
    assert main([*run, "--out", str(full)]) == 0
    assert set(json.loads(full.read_text())) == {"summary", "by_game", "shots", "clips"}
    capsys.readouterr()

    assert main([*run, "--out", str(short), "--summary-only", "--compare", str(full)]) == 0
    printed = capsys.readouterr().out
    assert "found_and_called_right 1/2 (50%) [9-91]" in printed and "shots (label / predicted)" not in printed
    assert "found_and_called_right         gains 0   losses 0   p 1.000" in printed
    report = json.loads(short.read_text())
    assert set(report) == {"summary", "paired"} and report["paired"]["labeled_shots_in_both"] == 2

    assert main(["--compare-reports", str(full), str(full)]) == 0
    assert "false_shots                    gains 0   losses 0   p 1.000   1 -> 1" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(["--compare-reports", str(full), str(short)])  # a summary-only report has no shots to pair


def test_summary_only_still_names_the_clips_that_failed(tmp_path, capsys):
    labels = tmp_path / "labels.csv"
    labels.write_text(HEADER + "a,1.0,made,,,,,,\ngone,2.0,missed,,,,,,\n")
    write_job(tmp_path / "jobs", "a", [shot(1.0)], duration_s=10.0)
    assert main(["--labels", str(labels), "--jobs", str(tmp_path / "jobs"), "--out", str(tmp_path / "short.json"),
                 "--summary-only"]) == 0
    printed = capsys.readouterr().out
    assert "2 clips (1 failed)" in printed and "  gone: FileNotFoundError" in printed
    assert "a @1.0s" not in printed  # still no shot lines


def test_a_clip_without_a_saved_length_is_timed_from_its_video(tmp_path):
    import cv2
    import numpy as np

    write_job(tmp_path, "a", [shot(1.0)])
    writer = cv2.VideoWriter(str(tmp_path / "a" / "input.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 48))
    for _ in range(45):
        writer.write(np.zeros((48, 64, 3), np.uint8))
    writer.release()
    assert evaluate(read_labels_from(tmp_path, "a,1.0,made,,,,,,\n"), tmp_path, .75)[0].duration_s == 1.5


def test_compare_leaves_out_a_clip_that_failed_in_one_run(tmp_path):
    labels = read_labels_from(tmp_path, "a,1.0,made,,,,,,\nb,2.0,missed,,,,,,\n")
    write_job(tmp_path / "before", "a", [shot(1.0), shot(5.0)])
    write_job(tmp_path / "before", "b", [shot(2.0, "missed")])
    write_job(tmp_path / "after", "b", [shot(2.0, "missed")])  # a has no result, so it fails
    paired = compare(report_of(tmp_path / "before", labels), report_of(tmp_path / "after", labels))
    assert (paired["labeled_shots_in_both"], paired["labeled_shots_only_before"], paired["clips_compared"]) == (1, 1, 1)
    assert paired["false_shots_not_compared"] == 1 and paired["metrics"]["false_shots"]["gains"] == 0
    assert paired["metrics"]["found_and_called_right"] == {"before": 1, "after": 1, "of": 1, "gains": 0, "losses": 0,
                                                           "p": 1.0}
