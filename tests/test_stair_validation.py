from copy import deepcopy

from scripts.validate_stairs import case_quality, config_for_case, evaluate_convergence, load_base_preset, physical_signature, times_for, CASE_SPECS


def completed_record(name="baseline"):
    events = [{"step": i + 1, "verified": True, "ambiguous": False, "endpoint_label": label,
               "first_sustained_time": time - 0.1, "endpoint_supports": [{"label": label, "time": time}]}
              for i, (label, time) in enumerate(zip(["last", "first", "last"], [0.65, 1.05, 1.99]))]
    return {"name": name, "status": "completed", "simulation_time": 2.4, "duration": 2.4,
            "config": {}, "support": {"confirmed_flip_count": 3, "first_three_endpoint_support_events": events}}


def test_convergence_uses_endpoint_support_time():
    baseline = completed_record()
    half_dt = deepcopy(baseline)
    half_dt["name"] = "half_dt"
    mesh = deepcopy(baseline)
    mesh["name"] = "mesh"
    mesh["support"]["first_three_endpoint_support_events"][2]["endpoint_supports"][0]["time"] = 2.2
    assert times_for(baseline) == [0.65, 1.05, 1.99]
    result = evaluate_convergence([baseline, half_dt, mesh], "extended")
    assert not result["passed"]
    assert not result["support_time_checks"][1]["events"][2]["passed"]


def test_three_contacts_do_not_replace_verified_unambiguous_supports():
    record = completed_record()
    assert case_quality(record)["passed"]
    record["support"]["first_three_endpoint_support_events"][1]["ambiguous"] = True
    assert not case_quality(record)["passed"]
    record["support"]["first_three_endpoint_support_events"][1]["ambiguous"] = False
    record["support"]["first_three_endpoint_support_events"][1]["verified"] = False
    assert not case_quality(record)["passed"]


def test_incomplete_exports_cannot_pass_acceptance():
    record = completed_record()
    record["hdf5_error"] = "failed to close file"
    assert not case_quality(record)["passed"]
    record["hdf5_error"] = None
    record["frame_errors"] = ["Missing or changed evidence artifact"]
    assert not case_quality(record)["passed"]


def test_refinement_keeps_physics_and_sampling_identical():
    base = load_base_preset()
    records = [{"config": config_for_case(base, spec).model_dump(mode="json")} for spec in CASE_SPECS]
    assert all(physical_signature(record) == physical_signature(records[0]) for record in records)
    records[1]["config"]["numerics"]["contact_time_constant"] *= 2
    assert physical_signature(records[1]) != physical_signature(records[0])
