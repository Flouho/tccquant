from tccquant.policies import policy_from_name


def test_preset_policies_exist():
    for name in ["W4A16", "W4A8", "W8A8", "W8A16"]:
        p = policy_from_name(name)
        assert p.name == name
